---
id: 4dcb087d-c5a7-4dae-a9c0-dd1cf43afe20
title: 'DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level'
date: '2025-04-09T19:51:30.081055Z'
original_slug: ainews-deepcoder-a-fully-open-source-14b-coder-at
description: >-
  **Together AI** and **Agentica** have released **DeepCoder-14B**, an
  open-source 14B parameter coding model that rivals **OpenAI's o3-mini** and
  **o1** on coding benchmarks, achieving **60.6% on LiveCodeBench** and **1936
  on CodeForces**. **Google DeepMind** launched **Gemini 2.5 Pro** with
  experimental "Flash" versions available to subscribers, while **Moonshot AI**
  released **Kimi-VL-A3B**, a multimodal model with **128K context**
  outperforming **GPT-4o** on vision and math benchmarks. **Meta AI** introduced
  **Llama 4 Scout** and **Maverick**, with a larger **Behemoth** model in
  training, and **Runway** released **Gen-4 Turbo** with 10x better results than
  Gen-3. **Google** also announced **Imagen 3**, a high-quality text-to-image
  model now in Vertex AI, and **Veo 2**. The open-source nature of
  DeepCoder-14B, including dataset, code, and training recipe, highlights
  advances in reinforcement learning training pipelines and sampler bottleneck
  solutions.
companies:
  - together-ai
  - agentica
  - openai
  - google-deepmind
  - moonshot-ai
  - meta-ai-fair
  - runway
models:
  - deepcoder-14b
  - o3-mini
  - o1
  - gemini-2.5-pro
  - kimi-vl-a3b
  - gpt-4o
  - llama-4-scout
  - maverick
  - behemoth
  - gen-4-turbo
  - imagen-3
  - veo-2
topics:
  - reinforcement-learning
  - code-generation
  - multimodality
  - model-training
  - model-releases
  - model-optimization
  - fine-tuning
  - sampler-bottleneck
  - pipelining
  - open-source
people:
  - _philschmid
  - reach_vb
  - yuchenj_uw
  - togethercompute
  - epochairesearch
  - danielhanchen
  - c_valenzuelab
---


<!-- buttondown-editor-mode: plaintext -->**GPRO+ is all you need.**

> AI News for 4/7/2025-4/8/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**229** channels, and **7279** messages) for you. Estimated reading time saved (at 200wpm): **692 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

After the DeepSeek R1 launch ([our coverage here](https://buttondown.com/ainews/archive/ainews-deepseek-r1-o1-level-open-weights-model/)), a raft of "R1 but more open" clone attempts emerged, of which it seems [only HuggingFace's OpenR1](https://github.com/huggingface/open-r1) is still posting active updates, if you discount [the distillation work](https://www.youtube.com/watch?v=jrf76uNs77k&t=1036s). However, today Together and [the Agentica Project](https://agentica-project.com/) (previously of [the DeepScaleR work](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)) have come out with a 14B code-focused reasoning model that scores at O3-mini level:

![image.png](https://assets.buttondown.email/images/2bd31332-ba53-4690-b8b0-05b9d310f013.png?w=960&fit=max)

Usually these projects are easy to game and therefore unremarkable, but this project distinguishes it self by being fully open source - dataset, code, recipe and all, meaning the educational value is high, particularly given the prior work of its collaborators. 

Specifically for RL training, they note the sampler bottleneck:

![image.png](https://assets.buttondown.email/images/d2d565ad-746c-451a-806a-2cf7f74f1488.png?w=960&fit=max)

so they have very good thoughts on pipelining:

![image.png](https://assets.buttondown.email/images/1c7ec18e-12c3-4305-841d-0aedd82e15d5.png?w=960&fit=max)

and they also propose an update to DeepSeek's GRPO:

![image.png](https://assets.buttondown.email/images/0d839059-e7f3-48ea-8ee1-07b5108182af.png?w=960&fit=max)


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Model Releases and Updates**

- **Gemini 2.5 Pro, including its "Flash" experimental versions, are now available for subscribers**, according to [@Google](https://twitter.com/Google/status/1909747273149395425) and [@_philschmid](https://twitter.com/_philschmid/status/1909737527386255649). It can be accessed within the Deep Research feature of the Gemini app, as noted by [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1909943627218129004).  [@lepikhin](https://twitter.com/lepikhin/status/1909748715340152967) mentioned the team worked hard to serve all the traffic.
- **Moonshot AI has released Kimi-VL-A3B**, a multimodal LM with 128K context under the MIT license, outperforming GPT4o on vision + math benchmarks according to [@reach_vb](https://twitter.com/reach_vb/status/1910046715714937130), with models available on [Hugging Face](https://twitter.com/reach_vb/status/1909706444028670311) and integrated with Transformers. [@_akhaliq](https://twitter.com/_akhaliq/status/1910047935686991904) also noted the release.
- **Together AI and Agentica have collaborated to release DeepCoder-14B**, an open-source coding model that rivals OpenAI's o3-mini and o1 on coding tasks, costing approximately $26,880 to train, according to [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1910004382848229702). The model, training code, dataset, and a detailed blog are available, noted by [@togethercompute](https://twitter.com/togethercompute/status/1909697122372378908). It achieves a **60.6% score on LiveCodeBench and 1936 on CodeForces**, performing on par with o3-mini (low) and o1 on competition-level coding tasks, as per [@togethercompute](https://twitter.com/togethercompute/status/1909697131645903065). It was trained using an open-source RL framework from ByteDance, as noted by [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1910008307202548074).
- **Meta AI has released Llama 4 Scout and Maverick**, with a larger version called Behemoth in training, as mentioned by [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1909699970594394173). Maverick mixes MoE layers & dense, while Scout uses L2 Norm on QK, according to [@danielhanchen](https://twitter.com/danielhanchen/status/1909726119500431685).
- **Runway has released Gen-4 Turbo**, which offers 10x better results than Gen-3 at the same price point, according to [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1909976566987161785).
- **Google has announced Imagen 3**, their highest quality text-to-image model, now in Vertex AI, which allows for easier removal of unwanted objects, as per [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1910009261075357902).
- **Google has announced Veo 2** which allows users to refine and enhance existing footage and direct shot composition in Vertex AI, according to [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1910009257405133179).

**Evaluations and Benchmarks**

- **OpenAI has released a new Evals API** for programmatically defining tests, automating evaluation runs, and iterating on prompts, integrating them into any workflow, as stated by [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1909721613853139353). [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1909721618676695270)  notes that good evals help improve the quality of model responses systematically.
- **Epoch AI Research evaluated Llama 4**, finding that Maverick and Scout scored 67% and 52% on GPQA Diamond, respectively, similar to Meta’s reported scores, according to [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1909700016249479506).
- **ZeroBench tests reveal that current vision-language models fail**, with GPT-4V and Gemini scoring 0% pass@1 and 0% 5/5 reliability on 100 hard visual reasoning questions, according to [@LiorOnAI](https://twitter.com/LiorOnAI/status/1910022443453800746).

**Agentic Systems and Tooling**

- **Auth0's Auth for GenAI now has native LlamaIndex support**, making it easier to build authentication into agent workflows, as announced by [@llama_index](https://twitter.com/llama_index/status/1909697035365961954).
- **MongoDB has released a repository with 100+ step-by-step notebooks on AI Agents and RAG**, covering chatbot construction to Airbnb agents, according to [@LiorOnAI](https://twitter.com/LiorOnAI/status/1909695352497910232).

**Industry Analysis**

- **Swyx believes the twittersphere is well-calibrated on individual developer tooling but not on how AI is improving every aspect of the SDLC**, which may be more impactful, making Sourcegraph well-positioned as an AI dev tooling company, according to [@swyx](https://twitter.com/swyx/status/1909695963498946903).
- **Nearcyan believes that consumers will not be prompting their own full apps** because most good apps require data and there is no real data portability for consumers, according to [@nearcyan](https://twitter.com/nearcyan/status/1909730703388115132).
- **Svpino argues it is essential to learn how to apply AI in one's craft**, as Shopify understands, and those who know what's up are asking people to learn and study, according to [@svpino](https://twitter.com/svpino/status/1909699728545349689).

**Humor/Memes**

- **Vikhyatk joked about lunch in downtown Seattle costing 16-20 H100-hours**, with caloric consumption dropping by 10x since converting $ to H100-hours, according to [@vikhyatk](https://twitter.com/vikhyatk/status/1909752681742422383).
- **Scaling01 joked that Gemini 3.0 will be too cheap to meter**, according to [@scaling01](https://twitter.com/scaling01/status/1909967686584455174).
- **Andrew Carr noted Gemini's run on Pokemon**, citing Gemini "I can't believe it took six tries, and now the game is asking if I want to humiliate myself further by giving this thing a nickname. No way. I don't want to name this symbol of my failure. I'll press B to decline", according to [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1909707900240773444).


---

# AI Reddit Recap

> Our pipelines had an outage yesterday. Sorry!

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp

**Theme 1: Model Mania: Gemini Reigns, Llama 4 Stumbles, New Contenders Emerge**

*   **Gemini 2.5 Pro Crowned King, But Lacks Reasoning Transparency**: Across multiple Discords (LMArena, OpenRouter, Perplexity AI, Nous Research AI, aider), **Gemini 2.5 Pro** earns high praise for general capabilities, creative writing, and even generating functional code from complex prompts, often cited as superior to competitors like **GPT-4.5** and **Claude 3.5 Sonnet**. However, users note its reasoning tokens aren't exposed via the Perplexity API, hindering its use as a reasoning model there, and it can still hallucinate even with deep research capabilities unless specifically grounded in **AI Studio**.
*   **Llama 4 Launch Leaves Users Lamenting**: The release of **Llama 4** (Scout, Maverick) met widespread disappointment (LM Studio, Manus.im, Yannick Kilcher, Nomic.ai), with users calling it *terrible*, *overhyped*, and potentially a step backward despite some decent Japanese language performance. Concerns center on *sloppy post-training*, questionable benchmark validity possibly due to overfitting or *gaming*, and higher **VRAM** requirements than expected for its performance level, leading many to wait for an overhaul or stick with alternatives like **Qwen's 14B**.
*   **Cogito & Nvidia Models Challenge the Status Quo**: New models are making waves, including **DeepCogito's v1 Preview** models (3B-70B), trained via **Iterated Distillation and Amplification (IDA)**, claiming to outperform **Llama, DeepSeek, and Qwen** equivalents and even **Llama 4 109B MoE**, offering both direct answering and self-reflection modes ([DeepCogito Research](https://www.deepcogito.com/research/cogito-v1-preview)). **Nvidia** also quietly released a SOTA-level reasoning model, [Llama-3.1-Nemotron-Ultra-253B-v1](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1), featuring a toggle to turn reasoning capabilities on or off ([Nvidia Blog Post](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)).

**Theme 2: Training & Fine-Tuning Frontiers**

*   **Unsloth Fine-Tuning Fixes and FP4 Finds**: **Unsloth AI** tackled **DDP** training issues on 3+ GPUs, recommending specific CUDA device visibility settings, while advocating for **bitsandbytes (bnb)** over GGUF for QLoRA training due to data efficiency. Users explored fine-tuning quantized models using **FP4** via tools like [Unsloth](https://github.com/unslothai/unsloth) for faster training, clarifying that while direct fine-tuning of quantized models isn't feasible, **LoRA** offers a viable path.
*   **Distributed Training Debates: DeepSpeed vs. FSDP & Untrusted Compute**: In **Torchtune**, the merits of integrating **DeepSpeed** were debated, with maintainers favoring native PyTorch **FSDP** for better composability, though offering support for community **DeepSpeed** recipes. Meanwhile, the **Panthalia** platform ([X.com Waitlist](https://x.com/panthaliaxyz/status/1909342585505669228)), inspired by the **Nous DeMo** paper, aims to verify untrusted, low-cost compute for **Distributed Data Parallel (DDP)** training using gradient compression ([Algorithm Docs](https://docs.panthalia.com/gradient-compression-algorithm)).
*   **Novel Techniques and Research Directions Discussed**: Researchers discussed the **Hierarchical Perceiver** patent by [Google DeepMind](https://www.freepatentsonline.com/y2025/0103856.html), potentially related to long context in Gemini, and debated **QKNorm** advancements ([Paper 1](https://arxiv.org/abs/2503.05453), [Paper 2](https://arxiv.org/abs/2502.00919)). Other discussions included the **MIPRO** algorithm for automated prompt engineering scaling across complex tasks ([TensorZero Blog](https://tensorzero.com/blog/from-ner-to-agents-does-automated-prompt-engineering-scale-to-complex-tasks)), and **OLMo** powering **DAPO** research for better RLHF answers ([DAPO Paper](https://arxiv.org/abs/2504.05118), [OLMo Paper](https://arxiv.org/abs/2504.04022)).

**Theme 3: Tools & Platforms: Updates, Bugs, and Battles**

*   **Platform Updates: New UIs, Rate Limits, and Rebrands**: **LMArena** launched its [Alpha UI](https://alpha.lmarena.ai/) for testing, while **OpenRouter** debuted a slick new frontend but tightened free model rate limits to **50 RPD** (unless users have $10+ credits), sparking user frustration. **Codeium** officially rebranded to **Windsurf** ([Rebrand Announcement](https://windsurf.com/blog/windsurf-rebrand-announcement)) following the success of its editor, launching a new [SubReddit](https://www.reddit.com/r/windsurf).
*   **Tool Troubles: Bugs Plague Cursor, Aider, and APIs**: **Cursor** users reported issues with the **C/C++ extension** requiring rollbacks ([Forum Thread](https://forum.cursor.com/t/c-c-extension-usage-restriction-message-appears-in-cursor/75902)), the **auto-select** feature choosing poor models, and potential bans for bypassing the trial. **Aider** users faced **/architect mode** edits being cut off and sought ways to disable auto-committing ([Aider Config Docs](https://aider.chat/docs/config/options.html)), while **Perplexity API** users noted discrepancies compared to the web UI and issues with **Sonar** prompts focusing on the system prompt ([Prompt Guide](https://docs.perplexity.ai/guides/prompt-guide)).
*   **Framework Frustrations and Fixes: Mojo, MAX, Granite**: **Mojo** developers discussed its borrowing paradigm ([Mojo vs Rust Blog](https://www.modular.com/blog/mojo-vs-rust)), `__moveinit__` vs `__copyinit__` ([Example Code](https://github.com/sstadick/mojo-demo/tree/main/examples)), and managing `Span` lifetimes. Users compared **MLX** and **MAX**, noting **MAX's** current inability to target Apple Silicon GPUs, while **Unsloth AI** users found a quick fix for a **GraniteModel** bug in Colab involving editing `config.json`.

**Theme 4: The AI Ecosystem: Research, Rumors, and Real-World Use**

*   **Research Ripples: Patents, Audits, and Unlearning**: **Google DeepMind's** attempt to patent the **Hierarchical Perceiver** ([Patent Link](https://www.freepatentsonline.com/y2025/0103856.html), [Paper Link](https://arxiv.org/abs/2202.10890)) sparked discussion about defensive patenting and long-context Gemini. Researchers sought AI professionals for an ethics-based auditing survey ([Survey Link](https://link.webropolsurveys.com/S/AF3FA6F02B26C642)), and **ICML** announced a machine unlearning workshop ([Workshop Website](https://mugenworkshop.github.io/)).
*   **Industry Insights & Intrigue: Google's Payroll, Tariffs, and Cybercrime**: A [TechCrunch article](https://techcrunch.com/2025/04/07/google-is-allegedly-paying-some-ai-staff-to-do-nothing-for-a-year-) alleged **Google** pays some departing AI staff for a year to prevent them joining competitors, raising questions about legality and impact. Concerns surfaced that potential **tariffs** on **NVDA GPUs** could slow AI progress, while others noted AI adoption by cybercriminals seems slower than expected, though a future "shock" remains possible.
*   **Applications & Integrations: MCP, Math, Auth, and Agents**: The **Model Context Protocol (MCP)** saw use cases discussed, including integrating with **Neo4j** graph databases for RAG using clients like [mcpomni-connect](https://pypi.org/project/mcpomni-connect/), and **Semgrep** rewrote its MCP server using SSE ([Cursor Demo](https://www.loom.com/share/8535d72e4cfc4e1eb1e03ea223a702df)). **AI4Math** discussions highlighted using LLMs with formal systems like **Lean** for theorem proving ([Kaiyu Yang Lecture](https://www.youtube.com/live/cLhWEyMQ4mQ)), while **Auth0's Auth for GenAI** integrated native **LlamaIndex** support ([Tweet](https://twitter.com/llama_index/status/1909697035365961954)). Mozilla AI released `any-agent` to simplify agent framework evaluation ([GitHub Repo](https://github.com/mozilla-ai/any-agent)).

**Theme 5: GPU & Hardware Hustle**

*   **Hardware Headaches: ROCm Woes and METAL Sync Glitches**: Users continued to struggle getting **ROCm** via **WSL** working on **AMD 7800XT** GPUs due to lack of official support ([AMD Docs](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html)) and WSL passthrough issues. In **tinygrad**, a user debugging a **METAL sync issue** bounty found that sharding problems in LLaMA might stem from **COPY** operations executing before **XFER** commands finished, causing incorrect data reads.
*   **Performance Puzzles & Optimizations**: **Tinygrad** users reported significant speedups on **AMD** hardware using **BEAM=2**, surpassing **Torch** performance. In **GPU MODE**, discussions centered on **Triton's** `tl.make_block_ptr` with **`boundary_check`** for handling out-of-bounds memory safely (at a slight performance cost) and **TorchTitan's** unique pre-compile strategy potentially avoiding `torch.compile` bugs ([TorchTitan Code](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313)), though numerical issues with `torch.compile` and **FSDP** persist.
*   **New Releases & Resources for GPU Gurus**: **Nvidia's PhysX** CUDA physics simulation kernels are now [open source](https://github.com/NVIDIA-Omniverse/PhysX/discussions/384), inviting community ports (like ROCm). **TorchAO v0.10.0** was released ([Release Notes](https://github.com/pytorch/ao/releases/tag/v0.10.0)), adding **MXFP8** training support for **Nvidia B200** and a module swap quantization API. For learning, the [geohotarchive YouTube channel](https://www.youtube.com/@geohotarchive/videos) and the **Programming Massively Parallel Processors (PMPP)** book (4th ed) were recommended.

---

# PART 1: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 2.5 Pro Declared A.I. Supreme**: Members are calling [Gemini 2.5 Pro](https://ai.google.com/models/gemini) the first *true* A.I., highlighting its superiority in creative writing and consistency over previous models.
   - While **Gemini 2.5 Pro** excels in general tasks, it has been noted that the unreleased **Nightwhisper** model is superior in coding capabilities.
- **OpenAI's Deep Research Gets Skeptical Eye**: Doubts emerge regarding **OpenAI's Deep Research** [project](https://openai.com/research/deep-research), despite claims of it being the *best agent for web searching*, with some stating *2.5 with tools is just on another level*.
   - The prevailing sentiment suggests that **Deep Research** is merely a rebranded version of **OpenAI's** existing **o3 model**.
- **DeepCoder-14B Debuts with Muted Applause**: **Together AI** and **Agentica** launched [DeepCoder-14B-Preview](https://www.together.ai/blog/deepcoder), a code reasoning model, *finetuned from Deepseek-R1-Distilled-Qwen-14B via distributed RL*.
   - However, this release was met with criticism, with one user deriding the marketing as *the dumbest most shameful marketing ever*, saying the gains aren't impressive considering this is just o3-mini.
- **NightWhisper's Coding Skills Tease**: Enthusiasm builds around the potential release of **NightWhisper**, celebrated for its coding capabilities demonstrated on the arena, despite its short webdev and lmarena availability.
   - There's speculation that **NightWhisper** might align with the upcoming **Google Ultra model**.
- **Alpha UI Opens for Crowd Testing**: The **Alpha UI** is now available for testing [here](https://alpha.lmarena.ai/) **without a password**.
   - Users are prompted to provide feedback and bug reports through the provided [Google Forms](https://forms.gle/8cngRN1Jw4AmCHDn7) and [Airtable](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) links, as frequent updates are expected for both **Desktop & Mobile**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Patches DDP Training**: Users reported issues with **HF Trainer and DDP** not working with 3 or more GPUs, and it was recommended ensuring CUDA visible devices are set to a specific GPU, but [Unsloth supports DDP](https://docs.unsloth.ai/).
   - After testing, it threw a ValueError, so members recommended ensuring CUDA visible devices are set to a specific GPU.
- **bnb Is the Way for LoRA Training**: It was advised to use **bnb** (bitsandbytes) for QLoRA training instead of GGUF, as it saves downloading 4x the data, and you can save and merge the adapter with the bnb model for later export to GGUF.
   - Users were considering between training a LoRA on **bnb 4-bit** or GGUF for a tiny model, and the consensus leaned towards the former.
- **Llama 4 Models Earn a Sloppy Reputation**: Members testing **Llama 4** (Scout and Maverick) found it to perform well in Japanese and to be capable base models despite sloppy post-training.
   - The general sentiment is to await a forthcoming post-training overhaul.
- **DeepCogito v1 Claims Lead in LLM Performance**: DeepCogito's claims their [v1 Preview models](https://www.deepcogito.com/research/cogito-v1-preview) outperform the best available open models of the same size, including counterparts from LLaMA, DeepSeek, and Qwen.
   - These models offer the ability to answer directly (standard LLM), or self-reflect before answering (like reasoning models).
- **GraniteModel Bug Affects Colab**: Users encountered a bug in the Colab notebook using **GraniteModel**, and suggested a quick fix that involves editing `granite_based/config.json` to replace **GraniteModel** with **GraniteForCausalLM** and rerun the cell.
   - The recommended method for editing the file on Colab is to download, edit locally, and then upload the modified version back to Colab.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Free Model Limits Squeezed on OpenRouter**: OpenRouter reduced the token limit for free models to **50**, triggering negative reactions as users expressed frustration over the lowered limit, with some feeling that it's like a *paywall*.
   - Accounts with at least **$10 in credits** will have their daily requests per day (**RPD**) boosted to **1000**, while those with **less than $10** will see a decrease from **200 RPD** to **50 RPD**.
- **Quasar Credit-Dependent Rate Limit Coming**: The update notes that **Quasar** will soon have a rate limit that is dependent on credits and there is no hourly rate limit, but the rate limit is **20 requests per minute**.
   - Members opened a [feedback thread](https://discord.com/channels/994043905957435544/1243614384297644072) for users to post their thoughts on the changes.
- **OpenRouter Debuts Slick New Frontend**: OpenRouter has a new frontend that looks sick with big ups to [clinemay](https://discord.com/channels/1091220969173028894/1195014798837043240/1358883684609953812)!
   - One user joked that it looked like *gpt-3.5 made this website in about 4 minutes*.
- **Gemini Crowned King of the Models**: **Gemini 2.5 Pro** is on a whole other level compared to the other models, making it the most powerful model up to day.
   - One user noted it was rated as *1. gemini 2.5 pro ... 10. everyone else*.
- **Nvidia Stealthily Unleashes Reasoning Model**: [Nvidia](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) silently dropped a SOTA-level reasoning model.
   - The new model is casually showing it's better than **Behemoth**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Daniel Mac Graphs Code with GraphDB**: A member shared [Daniel Mac's tweet](https://x.com/daniel_mac8/status/1908332949251948808) about using a **graph database** for code querying.
   - This sparked a discussion on the potential benefits of using graph databases for code analysis and understanding complex relationships within codebases.
- **Manus.im Devours Credits**: A user reported that [Manus.im](https://manus.im) failed to answer a question correctly and consumed **984** of their **1000 free credits** on a single prompt.
   - Alternatives like [Smithery.ai](https://smithery.ai/) and [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers) were suggested as potential solutions.
- **C/C++ Extension Error Strikes**: A user reported encountering an error related to the **C/C++ extension** after using Cursor since March 2023, noting that the extension may be limited to Microsoft products.
   - A workaround involving [rolling back to a previous version](https://forum.cursor.com/t/c-c-extension-usage-restriction-message-appears-in-cursor/75902) was suggested, with users sharing [other forum threads](https://forum.cursor.com/t/c-c-extension-broken/75182) discussing the issue.
- **Auto-Select Model Labeled as Scam**: Users are reporting that the **auto-select** model option is choosing low quality models, with one user claiming it *fucked up my codebase*.
   - Another user suggested that this behavior might be intentional, raising concerns about the reliability of the **auto-select** feature.
- **Cursor's Ban-Hammer Swings at Free Tier Bypassers**: A member reported that bypassing the trial version of Cursor could lead to a complete ban from using the tool, with a warning that *you won’t be able to use it at all soon*.
   - This sparked a debate about the fairness of Cursor's trial version restrictions and the consequences of attempting to circumvent them.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 4 Disappoints Users**: Users expressed disappointment with **Llama 4's** performance, some describing it as a step backwards, questioning benchmark validity.
   - While **Llama 4** provides speed/cost similar to **17B** models with similar results to **24-27B**, it requires more **VRAM**, making it pointless for simple users, while **Qwen's 14B** models are praised.
- **ROCm on WSL Still Doesn't Work on 7800XT**: A user reported that **ROCm** via **WSL** doesn't work with a **7800XT** due to lack of official support ([AMD documentation](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html)).
   - Another user suggested it *might* work since both cards are **RDNA3**, while the first user confirmed that it was *impossible* to get working due to **WSL passthrough** issues.
- **Fix Cogito Jinja Errors Quickly**: Users reported errors with **Jinja templates** when using **cogito-v1-preview-llama-3b**, and were advised to use **ChatGPT** to quickly fix the template.
   - The community model maintainer was notified about the wonky template and is expected to update the model soon.
- **Docker Gets Bashed**: After one member expressed wanting to be *'best friends'* with anyone who says bad things about **Docker**, another member jokingly asked *'Did Docker take out your family or something?'*
   - The first member humorously replied, *'My therapist said I shouldn't talk about it.'
- **Debating an Affordable Supercomputer Build**: One user proposed building a **16-node supercomputer** with either **RTX 4090 D GPUs** or a less powerful option, aiming for a **2T model with 1M context**.
   - Skeptics questioned the feasibility, highlighting the need for **RDMA**, fast interconnects, and skilled engineers.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Startups Snag Savings via Perplexity**: Perplexity AI introduces a [startup program](https://www.perplexity.ai/startups) offering **$5000** in Perplexity API credits and **6 months** of Perplexity Enterprise Pro for eligible startups.
   - Eligibility requires less than **$20M** in funding, being less than **5 years** old, and association with a Startup Partner.
- **Gemini 2.5 Reasoning Ruckus Reported**: Members noted that **Gemini 2.5 Pro** doesn't expose its reasoning tokens via the API, and therefore can't be included as a reasoning model on Perplexity, though it is a *high latency thinking model*.
   - Consequently, the reasoning isn't displayed via the API, unlike in **AI Studio**.
- **Deep Research High Hype but Hindered**: Users await the rollout of **Deep Research High**, which aims to use **150-200 sources** on average, yet one user reports *Perplexity's deep research got 23 sources, the free gemini deep research got over 500*.
   - Some members are frustrated by the lack of communication on the release timeline and the current version's summary output, instead of a truly deep research; check out the [DeepSeek Subreddit](https://www.rxddit.com/r/DeepSeek/s/zFUYlP8NeV).
- **Llama 4 faces Benchmark Faking Flak**: Concerns were raised regarding a [Perplexity AI search result](https://www.perplexity.ai/search/does-llama-4-fake-benchmarks-pw9wkBJ4TCOUtdZu8fmTdg#0) questioning if **Llama 4** is faking benchmarks.
   - This is part of a broader discussion regarding model benchmarking transparency and the methodologies used to evaluate **Llama 4**.
- **Perplexity API: Prompting Problems Persist**: A user reported that **Sonar** responses focus on the system prompt, rather than user queries, while a team member clarified that the system prompt isn't used during the search phase, advising the user to optimize the **user prompt** instead using [Prompt Guide](https://docs.perplexity.ai/guides/prompt-guide).
   - Also, some members discussed discrepancies between the **Perplexity API** and the **web UI** when summarizing web pages, with the **API sandbox** even giving way better results than the actual API when using **sonar-reasoning-pro**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Local Manus is on the Horizon**: Members speculated that a local version of **Manus** will be possible in the future, similar to other **AI models**.
   - This would allow users to run **Manus** on their own hardware, addressing concerns about credit usage and data privacy.
- **MCP Servers Deployed on Claude**: As of November 25, 2024, **MCP servers** are available on **Claude** and can be used with Claude code, as one member reported.
   - This integration enables users to leverage **MCP servers** within the Claude environment for enhanced functionality.
- **Llama 4 Hype Train Derails**: After testing on **Openrouter.AI**, users report that **Llama 4** is overhyped due to subpar responses.
   - Criticism extends to **Zucks**, who is accused of gaming the **benchmarks**, leading to inflated performance expectations.
- **Octopus Web Scraper Steals the Show**: A member reported that the free website scraper [Octopus](https://octoparse.com/) works effectively on Zillow and Realtor, offering a cost-effective alternative to Bardeen, which is priced at $130/month.
   - The high cost of Bardeen prompted suggestions to use **Manus** for building a custom scraper as a more economical solution.
- **Manus Credit Crunch Angers Users**: Users express dissatisfaction with the high cost of [Manus credits](https://www.manus.im/pricing), reporting that even simple tasks consume substantial credits, with one user exhausting 1000 free credits on a single Standard Complexity task.
   - To mitigate credit consumption, users suggest breaking tasks into smaller dialogue windows and considering **Proxy** as a cheaper alternative, pending updates to **Manus's** pricing and credit plans.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 vs Sonnet Prompting Power**: Users found **Gemini 2.5's** logic strong but instruction following poor, contrasting it with **Sonnet's** feature-rich coding that needs more prompting.
   - One user reported needing only 1 prompt with Gemini 2.5 compared to 3 prompts for Sonnet, even with Sonnet's advanced features like *multiple file input methods and batch processing*.
- **Aider's Auto-Commit Causing Havoc?**: A user seeks to disable **Aider's auto-committing** due to committing untested code, referencing the [Aider configuration options](https://aider.chat/docs/config/options.html).
   - Another user suggested providing a [model and key](https://aider.chat/docs/troubleshooting/models-and-keys.html) or Aider will guess based on available keys.
- **OpenRouter's Missing Sonar Pro Citations**: A user questioned missing citation links when using **Perplexity Sonar Pro** via **OpenRouter**, providing a visual reference [here](https://cdn.discordapp.com/attachments/1131200896827654144/1358926629170319490/image.png?ex=67f798cc&is=67f6474c&hm=fe2c340b866bec81e485bbed3c2d1fe17071b540d6ea5c803306211e3d9f2ceb&).
   - The discussion implies potential issues with citation link reliability when using certain models through OpenRouter.
- **Software Engineer Gap Year a Career Killer?**: An article argues that taking a gap year/holiday would be a poor decision for software engineers, citing insights about the current tech landscape, see [this article](https://ghuntley.com/dothings/).
   - The author suggests the fast-evolving nature of tech makes extended breaks detrimental for maintaining relevance.
- **Architect Mode Edits Getting Interrupted**: Users report **/architect mode** edits in Aider being cut off when adding new files, leading to potential loss of the editor state.
   - Avoiding the addition of new files during editing appears to allow the process to continue without interruption.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **AgentSpace unlocks NotebookLM for Enterprise**: Google's **AgentSpace** documentation reveals that **NotebookLM Enterprise** can now be set up with [Customer-Managed Encryption Keys (CMEK)](https://cloud.google.com/agentspace/notebooklm-enterprise/docs/set-up-notebooklm#cmek) for better data encryption control.
   - A user inquired about commercial-scale **NotebookLM**, and another member pointed out this new offering.
- **NotebookLM's Privacy Assurances Confirmed**: Both the **Enterprise** and **Plus** versions of **NotebookLM** ensure user data remains private and never enters the public domain, according to a member.
   - This clarification addresses misunderstandings about **Google's privacy policy** and terms, noting built-in mechanisms to prevent prompt injection.
- **User Correction Improves NotebookLM's Summary**: A user reported that **NotebookLM** initially misread a scholarly article, but corrected itself after a quotation and explanation were provided.
   - Repeating the same prompt in different **Google accounts** from the beginning yielded correct results, raising questions about training and privacy.
- **Discovery Mode Rollout Still in Progress**: Users are still awaiting the new **Discovery Mode** feature in **NotebookLM**, with the rollout expected to take up to **two weeks** from the release date.
   - A user humorously demanded *special treatment as a Google fanboy* to get early access.
- **Gemini Still Hallucinates with Deep Research**: Users report that **Gemini** *hallucinates* with **deep research**, even with internet access.
   - A member clarified that **Gemini** can connect to Google Search, but it requires specific grounding instructions in **AI Studio**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek R2 Primed for LlamaCon Release**: Members are urging **DeepSeek** to release **R2** on the same day as **LlamaCon** to capitalize on the hype, noting that training data for **MoE** differs from base models, citing [this paper](https://arxiv.org/abs/2410.19034).
   - The release could challenge other models and draw significant attention during the event.
- **Together AI Gets into the Training Game**: **Together AI** is entering the model training business, as evidenced by [this case study](https://www.together.ai/models/cogito-v1-preview-llama-70b) showcasing the **Cogito-v1-preview-llama-70B** model.
   - This move marks a shift towards providing comprehensive AI solutions, including training infrastructure and services.
- **Google Rumored to Pay AI Staff for Idleness**: According to [this TechCrunch article](https://techcrunch.com/2025/04/07/google-is-allegedly-paying-some-ai-staff-to-do-nothing-for-a-year-), **Google** is allegedly paying some **AI staff** to do nothing for a year rather than allowing them to join competitors.
   - A member critiqued this as a *basic management idea with horrifically bad second-order effects*, with another noting it could create legal perils by restricting what they do or build while under contract.
- **Tariffs Threaten NVDA GPU Availability**: Members speculated that if **tariffs** remain, the AI field may slow down due to the increased cost of **NVDA GPUs**.
   - This could impact development and research, as access to necessary hardware becomes financially constrained.
- **OLMo Powers DAPO Research**: Members discussed [a DAPO paper](https://arxiv.org/abs/2504.05118) as offering *'Extreme value'*, referencing [another paper built on OLMo](https://arxiv.org/abs/2504.04022).
   - The researchers noted a novel compute method that results in better answers for **RLHF** tasks.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **DeepMind's Hierarchical Patent Pursuit**: [Google DeepMind](https://www.freepatentsonline.com/y2025/0103856.html) is trying to patent the **Hierarchical Perceiver**, drawing comparisons between the patent diagrams and those in the original [research paper](https://arxiv.org/abs/2202.10890).
   - Speculation suggests this patent might be related to DeepMind's work on **ultra-long context lengths** in Gemini, possibly as a defensive measure.
- **Survey Seeks AI Auditing Experts**: A researcher seeks participation from AI professionals for a survey on ethics-based auditing of generative AI systems.
   - The [survey](https://link.webropolsurveys.com/S/AF3FA6F02B26C642) aims to gather insights on auditing or evaluating AI systems, especially generative models.
- **Debate Dawns Over Dubious Developments in QKNorm**: Members debated that the **QKNorm developments** are not the right way to go, referencing [this paper](https://arxiv.org/abs/2503.05453).
   - A member suggested a [better/earlier paper](https://arxiv.org/abs/2502.00919).
- **ICML Invites Investigation Into Unlearning**: A member shared that [ICML](https://icml.cc/Conferences/2024) will have a **machine unlearning workshop**.
   - The workshop's website can be found [here](https://mugenworkshop.github.io/).
- **LM Harness Hand-holding Heeded**: A member inquired about a **LM harness implementation for HotpotQA** to evaluate **Llama** and **GPT models**.
   - Guidance was requested on running evaluations against **HotpotQA**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama-4-Scout-17B Ready for llama.cpp**: [Llama-4-Scout-17B text-to-text](https://github.com/ggml-org/llama.cpp/pull/12791) support has been added to *llama.cpp*, and members are converting and quantizing the model.
   - This pre-release has generated excitement among users, eager to test its capabilities.
- **Gemini 2.5 Pro Generates functional Code Snippets**: **Gemini 2.5 Pro** is praised for generating functional code snippets from complex prompts, see the prompts and responses in [this message](https://cdn.discordapp.com/attachments/1149866623109439599/1358975415426879589/message.txt?ex=67f7c63b&is=67f674bb&hm=1c655347ddb71efc0e03a079e62d8e26286724363242370cf6f19b9e50cc1980&).
   - A user reports using **aider-chat** combined with **Gemini 2.5 Pro** to edit or create 15 files from a **300k token context**, including their frontend, API, and microservices.
- **HiDream-I1 Generates High-Quality Images**: **HiDream-I1** is a new open-source image generative foundation model with **17B parameters** using **Llama 3.1 8B** as a text encoder, released under the [MIT license](https://huggingface.co/HiDream-ai/HiDream-I1-Full).
   - It *produces exceptional results across multiple styles including photorealistic, cartoon, artistic, and more, achieving state-of-the-art HPS v2.1 score, which aligns with human preferences*.
- **Cogito Models use Iterated Distillation**: A new suite of **Cogito** models (**3B-70B**) outperform models like **Llama, DeepSeek, and Qwen**, trained using **Iterated Distillation and Amplification (IDA)**, which iteratively improves a model's capabilities.
   - Notably, the **70B model** allegedly surpasses the newly released **Llama 4 109B MoE model**, as outlined in [this research](https://www.deepcogito.com/research/cogito-v1-preview).
- **Panthalia Platform Aims to Verify Low-Cost Compute with DDP**: Inspired by the **Nous DeMo** paper, a platform has been developed to verify untrusted, low-cost compute for training models over the internet using distributed data parallel (DDP), with a waitlist available via [X.com](https://x.com/panthaliaxyz/status/1909342585505669228).
   - The platform uses a gradient compression algorithm, documented [here](https://docs.panthalia.com/gradient-compression-algorithm), with code available on [GitHub](https://github.com/ritser-labs/panthalia-worker/blob/main/spl/util/demo.py).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPUMODE's dataset requires PyTorch 2.5**: The **GPUMODE** "triton" dataset, used for [Inductor Created Data](https://huggingface.co/datasets/GPUMODE/Inductor_Created_Data_Permissive), was created using **PyTorch 2.5**, and the creator promised to update the readme.
   - Users may experience issues running the dataset on **PyTorch 2.6+**.
- **Triton Gets Boundary Checks**: A member suggested using `tl.make_block_ptr` with **`boundary_check`** and **`padding_option="zero"`** to create pointers that can fill with zeros for out-of-bounds memory accesses.
   - It was clarified that omitting `boundary_check` increases speed, but risks errors like *"device-side assert triggered"* due to potential buffer overruns.
- **TorchTitan Compiles Before Ops**: **TorchTitan** does a unique per-block compile before operations, potentially to circumvent some **torch compile bugs**; see [torchtitan/parallelize_llama.py#L313](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313).
   - Numerical issues may still exist when using `torch.compile` and **FSDP** together.
- **PhysX now Open Source**: NVIDIA's **CUDA physics simulation kernels** are now [open source](https://github.com/NVIDIA-Omniverse/PhysX/discussions/384), and some are already working on a **ROCm** version.
   - The **Triton-Distributed** learning note details fusing Triton with **NVSHMEM/ROC-SHMEM** to enable multi-GPU execution.
- **LiveDocs Documents Legit Logistics**: The creator of **LiveDocs** invites users to *document your code* with their upgraded service, now with more features available via signup at [www.asvatthi.com](http://www.asvatthi.com).
   - Included was an image of the interface, showing off various code documentation pages.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **FP4 Fine-Tuning Fuels Faster Finishes**: Users are exploring fine-tuning quantized models using **FP4** with tools like [Unsloth](https://github.com/unslothai/unsloth), which allows loading lower precision models for training and quantization.
   - While fine-tuning a quantized model is possible via **LoRA**, directly fine-tuning the quantized model itself is not.
- **Parasail Provides Premier Performance**: **Parasail**, a new inference provider, is looking to partner with Hugging Face after recently coming out of stealth, already serving **3B tokens a day** on Open Router and **5B+ a day** for private companies, as reported by [The Next Platform](https://www.nextplatform.com/2025/04/03/parasail-brokers-between-ai-compute-demand-and-supply/).
   - The Next Platform reported that Parasail brokers between AI compute demand and supply.
- **Llama.cpp Leaps to Llama 4**: The backend **Llama.cpp** has been updated to support **Llama 4**, according to the [GitHub releases](https://github.com/ggml-org/llama.cpp/releases).
   - This update allows for enhanced compatibility and performance with the latest Llama models.
- **AI Runner Desktop GUI Takes Flight**: A member released **AI Runner**, a desktop GUI for running AI models locally using HuggingFace libraries as described in [this YouTube video](https://youtu.be/IPn3TcQr7e0).
   - The tool enables users to create and manage chatbots with custom voices, personalities, and moods, and the bots are agents built with llama-index using ReAct tools to generate images with **Stable Diffusion** and real-time voice conversations (espeak, speecht5, or openvoice).
- **any-agent Library Simplifies Agent Framework Evaluation**: The Mozilla AI team released `any-agent`, a library designed to simplify trying different agent frameworks, with a [GitHub repository](https://github.com/mozilla-ai/any-agent) available for users to try and contribute.
   - The library supports frameworks like **smolagents**, **OpenAI**, **Langchain**, and **Llama Index**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Semgrep MCP Server Gets Docker Boost**: A member reports running the [Semgrep MCP server](https://mcp.semgrep.ai/sse) for over a month, hosted via **Docker** and **AWS EC2**.
   - This setup is a practical demonstration of deploying MCP in a cloud environment, with potential for wider adoption given its ease of use.
- **CORS Error Fixed in Semgrep MCP Server**: A reported **CORS error** when connecting with the [Cloudflare Playground](https://playground.ai.cloudflare.com/) was quickly resolved.
   - The tool was being tested with **Cursor**, suggesting real-world application and integration needs.
- **HTTP Request-Response Support in MCP for Enterprises**: Discussion emerged regarding the need for **HTTP request-response** support in MCP for enterprise customers, highlighted in [this pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206#issuecomment-2766559523).
   - The demand for this feature underscores MCP's growing adoption among enterprise organizations.
- **MCP Integrates with Graph DB for RAG**: A member inquired about using MCP in a **RAG** use case with a **Neo4j graph database**, focusing on vector search and custom **CQL search**.
   - Another member confirmed this is a good use case, linking to [mcpomni-connect](https://pypi.org/project/mcpomni-connect/) as a viable MCP client, showcasing MCP's versatility.
- **Semgrep Rewrites MCP Server with SSE**: A member rewrote [Semgrep's MCP server](https://github.com/semgrep/mcp) and shared demo videos using **SSE** in [Cursor](https://www.loom.com/share/8535d72e4cfc4e1eb1e03ea223a702df) and [Claude](https://www.loom.com/share/f4440cbbb5a24149ac17cc7ddcd95cfa?sid=f190a5d6-176f-4ceb-86a2-35e98e701411).
   - The server is using **SSE** because the [Python SDK](https://github.com/modelcontextprotocol/python-sdk/pull/416) doesn't support HTTP streaming yet.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Shopify's AI Quest Gains Momentum**: Shopify's AI mandate is gaining attention, as highlighted in [this tweet](https://fxtwitter.com/tobi/status/1909251946235437514).
   - The company is pushing towards AI integration across its platform, with internal discussions focusing on practical applications and strategic implications.
- **Anthropic API Credits Have Expiration Dates**: Anthropic API credits expire after one year, potentially for accounting simplification and to account for the rapidly evolving AI landscape.
   - Members suggest that this policy helps manage projections in a quickly changing field, providing a framework for resource allocation and future planning.
- **NVIDIA Reasoning Model Features On/Off Toggle**: NVIDIA has released a new model with the ability to turn reasoning on or off, detailed in [this blog post](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/) and available on [Hugging Face](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1).
   - This feature allows developers to experiment with different reasoning approaches and fine-tune their AI applications for specific tasks.
- **Cybercrime's AI Adoption Slower Than Expected**: Despite basic AI applications like FraudGPT, mass adoption of AI by cybercriminals is surprisingly slow, with speculation that a "cybercrime AI shock" may occur when they adopt it more broadly.
   - One member noted that LLMs may have only recently become good enough for use in cybercrime, indicating that the technology is still maturing in this context.
- **Gemini Streams Pokemon Gameplay**: The Gemini AI is now playing Pokémon, garnering attention as shown in [this tweet](https://fxtwitter.com/kiranvodrahalli/status/1909699142265557208).
   - This showcases the potential of AI in gaming and interactive entertainment, demonstrating its ability to engage in complex tasks within virtual environments.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Llama 4 Benchmarking Shortcomings Exposed**: A member asserted that **Llama 4 flops** on nongamed nonoverfitted benchmarks, sparking interest in the paper [arxiv.org/abs/2408.04220](https://arxiv.org/abs/2408.04220) and a related [YouTube talk](https://www.youtube.com/watch?v=klW65MWJ1PY).
   - Concerns arose that *Meta should have clarified that “Llama-4-Maverick-03-26-Experimental” was a customized model to optimize for human preference*, according to [this fxtwitter link](https://fxtwitter.com/lmarena_ai/status/1909397817434816562?t=Gdzbf-abkahHSxqhEeqAkw&s=19).
- **Decoding Bayesian Structural EM's Secrets**: A member highlighted that **Bayesian inference** has been combining weights and architecture for around a century, citing [Bayesian Structural EM](https://arxiv.org/pdf/1301.7373) as an example.
   - They argued that *you do not gain any expressivity from updating both the architecture and the weights that you couldn't get from just weights*, citing [DARTS](https://arxiv.org/pdf/1806.09055) or [ES-ENAS](https://arxiv.org/pdf/2101.07415) as further examples.
- **DNA of a Model: Procedural Model Representation**: A member introduced **procedural model representation**, where a small seed generates a large model (architecture + weights), envisioning downloading a 10MB model to generate a 100TB model.
   - The member described *downloading DNA to generate a human*, by swapping seeds to generate different models.
- **Cogito 14b Adopts Efficient Tool Template**: The **14b model** unexpectedly began utilizing a more efficient tool calling template than what was initially provided in the instructions, see the [Cogito model](https://ollama.com/library/cogito).
   - This suggests the model may have autonomously optimized its tool use, offering a potential area for further investigation.
- **DeepCogito Improves Iteratively**: A member shared a link from **Hacker News** about an **iterative improvement strategy** using test time compute for fine-tuning, from [DeepCogito](https://www.deepcogito.com/research/cogito-v1-preview).
   - Another member pointed to [this paper](https://arxiv.org/pdf/2408.04220) and shared an [Awesome talk](https://www.youtube.com/watch?v=klW65MWJ1PY) about adapting **pre-training text**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Granite 8B Impresses with RAG-ability**: Members reported that [IBM Granite 8B](https://www.ibm.com/blogs/research/granite-foundation-models/) is effective with **RAG** tasks, especially regarding providing references.
   - Other members concurred, having also found **Granite** to be effective.
- **Docling Does OCR Delicately**: A member recommended **docling** for **image OCR**, especially for non-text PDFs like scans, for running embeddings.
   - They highlighted its continuous operation for embeddings and integration into a database with indexed documents, enabling **RAG** through intersections.
- **Semantic Chunking Chunks Context**: A member shared a semantic chunking server, demonstrating its use with [clipboard examples](https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html).
   - They noted its compatibility with audio and image processing, suggesting **ComfyUI** for combining all modalities.
- **Llama 4th Gen Bashed Badly**: A member trashed the **Llama 4th gen model** for being *terrible compared to smaller models*.
   - Others agreed, noting [Reddit comments](https://www.reddit.com/r/LocalLLaMA/) speculated that it may have overfit on smaller "high quality" datasets, despite some benchmarks showing promise.
- **GPT4All: Run Locally!**: A member advised using **GPT4All** primarily for local operations to ensure privacy and avoid sending private information to remote APIs.
   - They detailed how to run embedding models locally and index files by chunking and embedding, referencing a [shell script example](https://gnu.support/files/tmp/clipboard-2025-04-09-01-48-48.html).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX struggles with Apple Silicon Deployment**: A member compared **MLX** and **MAX**, noting **MAX** currently cannot target Apple Silicon GPUs, unlike **MLX**, which poses challenges for direct comparison and deployment.
   - They suggested that while **MLX** is convenient for initial experiments, the practical limitations of deploying Apple's ecosystem in server settings necessitates rewriting to frameworks like **MAX**, **JAX**, or **PyTorch**.
- **Mojo Borrowing Paradigm Receives Praise**: A newcomer shared [a blog post comparing Mojo and Rust](https://www.modular.com/blog/mojo-vs-rust), observing that Mojo's *borrow by default* felt more intuitive, and wondered about how Mojo handles returning values from functions.
   - Discussion ensued on how Mojo handles returning values from functions.
- **Moveinit vs Copyinit deep dive**: A member clarified that when returning objects in Mojo, the presence of `__moveinit__` dictates whether the object is moved, otherwise `__copyinit__` is used, and provided [an example on Github](https://github.com/sstadick/mojo-demo/tree/main/examples).
   - The member also pointed to the [official Mojo documentation](https://docs.modular.com/) for a complete picture.
- **Span Lifetimes got you down? Rebind!**: A member inquired how to specify in Mojo that *"the lifetime of the return value is at least the lifetime of self"*, specifically for a `Span`.
   - Another member suggested using `rebind[Span[UInt8, __origin_of(self)]](Span(self.seq))` or making the trait generic over origin, but noted that trait parameters are not yet supported.
- **Self-Promotion Rules Trigger Moderator!**: A member flagged a post in the Discord channel as a violation of self-promotion rules.
   - A moderator agreed, confirming the post indeed violated the community's self-promotion guidelines.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Seeking Elegant Tensor Naming**: A member is seeking a more elegant way to name tensors for easier tracking when printing model parameters, instead of manually adding a *name* attribute in the Tensor class.
   - The member is seeking techniques to streamline tensor naming conventions for enhanced code readability.
- **GPU Programming and Compiler Dev Resources**: A member expressed interest in getting into **GPU programming** and **compiler development** for projects like tinygrad and requested learning resources or blog posts.
   - The member is planning to read [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/) and asked for book or blog post recommendations on compiler development for GPUs, with another member recommending the [geohotarchive YouTube channel](https://www.youtube.com/@geohotarchive/videos) as a resource for learning about tinygrad, and **PMPP (4th ed)** for GPU programming.
- **METAL Sync Glitch Shards LLaMA**: A member found unexpected behavior in sharding while reproducing a minimal example of a **METAL sync issue** from the bounty, suspecting that the **COPY** from **METAL:1** to **CPU** was executing before the **XFER** from **METAL** to **METAL:1** ended.
   - The user suggests this caused the CPU to read zeros instead of the correct shard during **LLaMA** inference.
- **AMD BEAM=2 Turbocharges Tinygrad**: A user reported impressive speed improvements using **AMD** with **BEAM=2**, achieving **64 it/s**, outperforming their previous best with Torch at **55+ it/s**.
   - Members noted that *BEAM=2 often beats torch*.
- **LLaMA Sharding Loses Device Info**: A user encountered an **AssertionError** while running **llama.py** with `--shard 4`, indicating that the device info was lost after sampling.
   - A potential fix was proposed to move the tensor, as seen on [GitHub](https://github.com/tinygrad/tinygrad/pull/9761/files).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama 4 Powers New RAG Workflow**: A quickstart tutorial demonstrates building a **RAG workflow** from scratch using **Llama 4**, showcasing how to set up core steps around ingestion, retrieval, and generation using LlamaIndex workflows, as shown in [this tweet](https://twitter.com/llama_index/status/1909635186079453494).
   - The tutorial focuses on core steps around ingestion, retrieval, and generation.
- **Auth0 and LlamaIndex Join Forces on Auth for GenAI**: **Auth0's Auth for GenAI** now ships with native LlamaIndex support, making it easier to build auth into agent workflows, as announced in [this tweet](https://twitter.com/llama_index/status/1909697035365961954).
   - This integration simplifies incorporating authentication into agent-based applications.
- **Gemini 2.5 Pro Shuttered, Points to Unified SDK**: Members discovered that **Gemini 2.5 Pro** is deprecated and to use **Google's latest unified SDK** instead, as noted in the [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/examples/llm/google_genai/).
   - It was brought up that the **Google SDK** doesn't validate model names, but assumes provided name is valid, so it may be important to double check.
- **StructuredPlannerAgent Gets the Axe**: The documentation for `StructuredPlannerAgent` was removed because it is no longer maintained due to a cleanup of the agent docs, with a backlink provided for historical reference: [StructuredPlannerAgent](https://docs.llamaindex.ai/en/v0.12.15/examples/agent/structured_planner/).
   - Instead of `StructuredPlannerAgent`, it was suggested to use an agent with a **planning tool** that does some **Chain of Thought (CoT)** reasoning, or using the **LLM** itself to create a plan before using agent(s).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Members Inquire on Event Recordings**: A member inquired about the availability of event recordings for those unable to attend live, but no response was given.
   - The member expressed interest, so in the future, **posting event recordings** would benefit absent members.
- **Newbies Seek Structured Output Guidance**: A new member requested examples of how to get structured output (e.g., a list of books) using **Cohere**, and were directed to the [Cohere documentation](https://docs.cohere.com).
   - The user admitted to being inexperienced with **Cohere**, and more examples of **structured output** may be warranted in the official documentation.
- **Pydantic Schemas Integrated via cURL**: A member sought ways to use **Pydantic schemas** directly in `response_format` with Cohere and avoid using the Cohere Python package.
   - They received a [link to the Cohere Chat API reference](https://docs.cohere.com/reference/chat) and a **cURL** example for requests to `https://api.cohere.com/v2/chat`, mirroring the approach in the **OpenAI SDK**.
- **Cohere Side-Steps Vector DB Recommendations**: Explicit recommendations for **vector DBs** have historically been avoided because Cohere's models are designed to function effectively with *all* **vector DBs**.
   - This approach ensures broad compatibility and a neutral stance towards the **vector database ecosystem**, meaning no special optimizations are needed for any particular **vector DB**.
- **Aditya Enters the Cohere Community**: Aditya, with a background in **machine vision and control**, introduced themself while taking a sabbatical to explore web/AI with the [openchain.earth](https://openchain.earth) project.
   - Aditya is using **VS Code**, **Github Co-Pilot**, **Flutter**, **MongoDB**, **JS**, and **Python** (Evaluating), looking to learn more about integrating **Cohere's AI** into their projects.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Contributor Tag Sought After**: A member requested a Contributor tag on Discord, sharing their [GitHub username](https://github.com/nathan-az).
   - The user lightheartedly mentioned their Discord profile picture featuring the character Gus from *Psych*.
- **DeepSpeed Integration Debated for TorchTune**: A member inquired about integrating [DeepSpeed](https://www.deepspeed.ai/) as a backend into TorchTune and created [an issue](https://github.com/pytorch/torchtune/issues/2569) to discuss the possibility.
   - A maintainer asked for more context, noting that **FSDP supports all the sharding options from DeepSpeed**.
- **TorchTune Favors FSDP Over DeepSpeed**: TorchTune leans towards **FSDP** due to its better composition with other PyTorch distributed features, with the belief that *supporting both versions well is not feasible*.
   - Users who migrated to TorchTune to avoid the complexities of composing DeepSpeed, PyTorch, and Megatron prefer sticking to native PyTorch.
- **Recipe for DeepSpeed with TorchTune?**: A maintainer suggested creating a community recipe that imports TorchTune and hosts a DeepSpeed recipe, offering to feature it if a repo is made.
   - This allows users interested in **DeepSpeed** to leverage it with TorchTune while keeping the core framework focused on native PyTorch.
- **Tweaking FSDPModule for zero1-2 Training**: Since TorchTune defaults to the equivalent of **zero3**, documentation or more recipes on how to tweak recipes using the **FSDPModule** methods for **zero1-2** training are appreciated.
   - It's believed that **zero 1-3** are all possible with very minor tweaks to the collectives.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MIPRO Algorithm Scaled on Complex Tasks**: An [article](https://tensorzero.com/blog/from-ner-to-agents-does-automated-prompt-engineering-scale-to-complex-tasks) tested the **MIPRO automated prompt engineering algorithm** across tasks of varied complexity, from named entity recognition to text-based game navigation.
   - The study leveraged tasks like **CoNLL++, HoVer, BabyAI**, and **τ-bench** (customer support with agentic tool use).
- **Larger Models Leverage MIPRO More**: The study found that **larger models benefit more from MIPRO optimization** in complex settings, potentially because they handle longer multi-turn demonstrations more effectively.
   - The quality of feedback significantly impacts the MIPRO optimization process, with meaningful improvements seen even from **noisy AI-generated feedback**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Kaiyu Yang Explores Formal Math Reasoning**: Guest speaker **Kaiyu Yang** presented on *"Language models for autoformalization and theorem proving"* on a livestream, available at [this link](https://www.youtube.com/live/cLhWEyMQ4mQ).
   - The lecture covered using LLMs for formal mathematical reasoning, including **theorem proving** and **autoformalization**.
- **AI4Math Becomes Crucial for AI Systems**: **AI for Mathematics (AI4Math)** is crucial for AI-driven system design and verification, mirroring NLP techniques, especially training LLMs on curated math datasets.
   - A complementary approach involves formal mathematical reasoning grounded in systems like **Lean**, which verify reasoning correctness and provide feedback.
- **Dr. Yang Enhances AI in Math**: **Dr. Kaiyu Yang**, a Research Scientist at Meta FAIR, focuses on enhancing AI's mathematical reasoning by integrating formal systems like **Lean**.
   - His work explores using LLMs for tasks like theorem proving (generating formal proofs) and autoformalization (translating informal to formal).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Manifold Research Deep Dive**: The **Manifold Research Group** is hosting their [Community Research Call #4](https://lu.ma/wlne416w) this Saturday (4/12 @ 9 AM PST), offering a look into their latest projects.
   - Discussions will include **Multimodal AI**, **self-assembling space robotics**, and **robotic metacognition**, inviting collaboration in frontier science.
- **Swarm Space Robotics Takes Flight**: A PhD student at **Manifold Research Group**, who specializes in robotic swarms in space, extended an invitation to the research call.
   - The research call seeks to encourage collaboration and probe frontier science in the field of space robotics.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium Rebrands to Windsurf After Editor Success**: Codeium rebranded to **Windsurf** after the successful launch of the **Windsurf Editor** in November 2024, explained in their [rebrand announcement](https://windsurf.com/blog/windsurf-rebrand-announcement).
   - The new name represents a blend of human and machine capabilities to create powerful experiences.
- **Windsurf Floats a New SubReddit**: Windsurf launched a new [SubReddit](https://www.reddit.com/r/windsurf) to build a community, coinciding with changes to their Discord server.
   - These changes included refreshed pages and channel renaming to reflect the new **Windsurf** branding.
- **Codeium Extensions Get a New Plugin**: With the rebrand, **Codeium Extensions** are now officially **Windsurf Plugins** and more innovation is promised.
   - The company reiterated their dedication to enhancing the **Windsurf Editor** continually.



---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1358879398237044846)** (1134 messages🔥🔥🔥): 

> `Gemini 2.5 Pro, OpenAI's Deep Research, Google's AI Strategy, DeepCoder-14B Preview Model, NightWhisper Model` 


- **Gemini 2.5 Pro Hailed as Superior Model**: Members are calling [Gemini 2.5 Pro](https://ai.google.com/models/gemini) the first *true* A.I., noting its superiority in creative writing and consistency over other models.
   - Some users have observed that while **Gemini 2.5 Pro** excels in general tasks, **Nightwhisper** is superior in coding.
- **OpenAI's Deep Research Under Scrutiny**: Users are questioning OpenAI's [Deep Research](https://openai.com/research/deep-research), noting its potential as the best agent for web searching, with one stating that *2.5 with tools is just on another level*.
   - However, the general consensus is that Deep Research is just OpenAI's existing o3 model.
- **Together AI Launches DeepCoder-14B Preview Model**: **Together AI** and **Agentica** jointly released [DeepCoder-14B-Preview](https://www.together.ai/blog/deepcoder), a code reasoning model, *finetuned from Deepseek-R1-Distilled-Qwen-14B via distributed RL*.
   - A user pointed out the *dumbest most shameful marketing ever* used, saying the gains aren't impressive considering this is just o3-mini.
- **NightWhisper Model's coding prowess praised**: Users are eagerly awaiting the potential release of **NightWhisper**, highlighting its demonstrated coding capabilities on the arena, despite its brief availability on webdev and lmarena.
   - Some speculate it's the same as the upcoming Google Ultra model.
- **O3 model variations get mixed reviews**: Members are comparing OpenAI's **O3 Mini** and **O3** models, with one noting that *O1 is more adept in deciding how long to think than O3 mini*.
   - One user with access to O3 medium described it as better at language-related problems than O1, but still weaker than Gemini 2.5 Pro for code.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1359224745370976438)** (1 messages): 

> `Alpha UI, Desktop & Mobile, Bugs, Leaderboard` 


- **Alpha UI Open for Testing**: The **Alpha UI** is now open for testing **without a password** at [https://alpha.lmarena.ai/](https://alpha.lmarena.ai/).
   - Users are encouraged to submit feedback and bug reports via the provided [Google Forms](https://forms.gle/8cngRN1Jw4AmCHDn7) and [Airtable](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) links.
- **Updates Coming Fast for Alpha UI**: The announcement mentions that the **Alpha UI** is an early version with limited features, but updates are coming quickly for **Desktop & Mobile**.
   - For the latest models and leaderboard data, users should refer to the main site, suggesting that the alpha version may not be fully up-to-date.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1358881886088204319)** (586 messages🔥🔥🔥): 

> `Unsloth DDP Support, GGUF vs bnb LoRA training, Llama 4 Analysis, cogito-v1 preview LLMs` 


- **Unsloth Addresses DDP Training Issues**: A user reported issues with **HF Trainer and DDP** not working with 3 or more GPUs, but working fine with 2, but [Unsloth supports DDP](https://docs.unsloth.ai/)
   - After testing, it threw a ValueError, and a member recommended ensuring CUDA visible devices are set to a specific GPU.
- **bnb Is the Way to Go**: A user inquired about whether to train a LoRA on **bnb 4-bit** or GGUF for a tiny model, to which it was advised to use **bnb** (bitsandbytes) for QLoRA training, as it saves downloading 4x the data.
   - Once the adapter is trained, it can be saved and merged with the bnb model, then exported to GGUF.
- **Llama 4 Models Get a Sloppy Reputation**: A member tested **Llama 4** (Scout and Maverick) and mentioned that it performs well in Japanese and seems to be capable base models with sloppily-put-together post-training.
   - Another member commented that they will be waiting for the post-training overhaul.
- **DeepCogito's v1 Preview LLMs Boast Strong Claims**: A user shared [DeepCogito's v1 Preview models](https://www.deepcogito.com/research/cogito-v1-preview), claiming their models outperform the best available open models of the same size, including counterparts from LLaMA, DeepSeek, and Qwen.
   - They claim each model can answer directly (standard LLM), or self-reflect before answering (like reasoning models).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1358950021415964873)** (21 messages🔥): 

> `iMatrix Dynamic Uploads, Apple BFloat, Model Pruning, Online DPO` 


- **iMatrix Dynamic Uploads Land on HF**: Members uploaded iMatrix dynamic versions for [Llama-4-Scout-17B-16E-Instruct-GGUF](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF) to HuggingFace.
- **B in BFloat stands for Brain**: The "B" in **bfloat** means "brain" and the dataype was developed at Google Brain, according to [Apple's documentation](https://developer.apple.com/documentation/metal/mtldatatype/bfloat?changes=_5_5&language=objc).
- **Schizo Theory**: A member shared his *"schizo theory is that companies like openai / claude / gemini use user inputs to prune their models"*.
   - He believes the *"Which one of these do you prefer"-like responses* are for *collecting user preference data for training their models*.
- **Online DPO learns you too well**: One member noted that online DPO starts to understand you better than you understand yourself.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1358885784073011441)** (175 messages🔥🔥): 

> `GraniteModel bug, Unsloth on MacOS, Multi-GPU Support, Gemma 3 12b issues, GRPO training` 


- ****GraniteModel Bug Bites Colab Users!****: Users encountered a bug in the Colab notebook using **GraniteModel**, but a quick fix involves editing `granite_based/config.json` to replace **GraniteModel** with **GraniteForCausalLM** and rerun the cell.
   - The recommended method for editing the file on Colab is to download, edit locally, and then upload the modified version back to Colab.
- ****MacOS Misses Out on Unsloth's GPU Goodness****: Unsloth currently **only supports GPUs**, leading to an `NotImplementedError` for MacOS users without NVIDIA GPUs.
   - However, there's a potential solution via [this pull request](https://github.com/unslothai/unsloth/pull/1289) that aims to address MacOS compatibility.
- ****Multi-GPU Support Coming Soon!****: Users are eagerly awaiting multi-GPU support for fine-tuning in Unsloth.
   - The response from the team is that it's *"soon (tm)"*.
- ****Gemma 3 12b Faces Loading Fails****: Users reported that `push_to_hub_merged` isn't uploading all the necessary files to HF, so they cannot use `AutoModelForCausalLM.from_pretrained("modelname/here")` and get an error `OSError: modelname/here does not appear to have a file named pytorch_model.bin`.
   - One member suggested that if you're using > 1B gemma it's a vision language model technically, so some things are slightly different. Users are suggested to try `FastModel` vs `FastLanguageModel` for gemma3.
- ****GRPO Training Tips Sought for Massive Models****: A user sought advice on training a **24B** model with a **16k** context length using GRPO, managing only a batch size of 1 on an H200 with 141GB VRAM and asked about Unsloth pro plan multi GPU support.
   - Suggestions included increasing gradient accumulation, with the possibility of multi-GRPO support via other frameworks, and discussions around distributed GRPO concepts for sampling efficiency.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1359245049044668467)** (2 messages): 

> `Location clarification` 


- **Location not France**: A member asked another member if they were from France.
   - The member responded clarifying that they are from **Dutch/Holland**.
- **Location Confirmed**: The member confirmed they are from Dutch/Holland.
   - This clarifies their origin in response to the initial question.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1358921030390911268)** (36 messages🔥): 

> `LLMs knowledge storage alternatives, RAG for memory offloading, Vector DBs and privacy, Retrieval augmented training, DeepSeek-V3` 


- **LLMs Mull Knowledge Storage Alternatives**: Members discussed the potential of LLMs **offloading knowledge retrieval to RAG pipelines** to reduce the size and increase the speed of the models, and training the attention heads to learn in conjunction with a vector database.
   - It was suggested that OpenAI could provide **generalized vector DB knowledge lookup** over private datasets that open LLM kernels could plug into for added context.
- **RAG Reimagined: Retrieval Portion Evolved**: Discussions revolved around splitting LLMs into a **knowledge model** and a **chat model**, where the chat model focuses on intelligence and reasoning and tool calls to the knowledge model.
   - While likened to RAG, the focus is on a kernel that works with experts or specialized vector DBs built on the same embeddings, effectively increasing vocab size in some sense.
- **Vector DB Ventures: Privacy Benefits Beckon**: A member noted that OpenAI could potentially benefit from giving away an open kernel for free: *"Look at your benchmarks before our attention vector lookups. Now look at your benchmarks after our attention vector lookups."
   - This could also lead to a privacy benefit by **only offloading static knowledge memory lookup**.
- **Rewarding Retraining: Forget What's Efficiently Remembered**: A participant suggested *"retrieval augmented training"*, **rewarding the model to forget** what it can efficiently remember via vector search.
   - This approach could lead to more efficient models by leveraging external knowledge sources during training.
- **DeepCoder Optimization Detailed**: A member shared a link to a [Together AI blog post](https://www.together.ai/blog/deepcoder) about **DeepCoder optimization**, highlighting its potential for optimizing the vLLM pipeline.
   - The optimization minimizes the wait for sampling by doing an initial sample and training, while simultaneously sampling again.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1358909049588154488)** (5 messages): 

> `Rate Limits, Credits, Quasar Rate Limit, Feedback on Rate Limiting` 


- **OpenRouter adjusts Free Model Rate Limits**: Accounts with at least **$10 in credits** will have their daily requests per day (**RPD**) boosted to **1000**, while those with **less than $10** will see a decrease from **200 RPD** to **50 RPD**.
- **Quasar to get Credit-Dependent Rate Limit**: The update also notes that **Quasar** will soon have a rate limit that is dependent on credits.
- **Feedback on Free Model Rate Limits**: A member opened a [feedback thread](https://discord.com/channels/994043905957435544/1243614384297644072) for users to post their thoughts on the changes.
- **Hourly rate limits not available**: There is no hourly rate limit, but the rate limit is **20 requests per minute**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1359349816118743130)** (2 messages): 

> `Olympia.chat, Shopify, SaaS Marketing, Turnkey Operation` 


- **Olympia.chat Seeks New Leadership**: The founder of [Olympia.chat](https://olympia.chat) has taken a role as Principal Engineer at **Shopify**, and the company is seeking an experienced site operator to take over technical maintenance and **SaaS marketing**.
   - The **profitable** site generates over **$3k USD per month**, and the founders are flexible about terms for a potential takeover, offering a **turnkey operation** with all IP included.
- **Olympia.chat's Financial Performance**: Despite peaking at nearly **$8k last year**, Olympia.chat currently generates over **$3k USD per month** consistently.
   - Lack of funding led to a halt in marketing efforts, impacting customer churn.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1358879753461043202)** (758 messages🔥🔥🔥): 

> `OpenRouter Frontend, Quasar Open Sourced, Free Model Rate Limits, API Keys Please, Gemini` 


- **OpenRouter Drops Sick New Frontend**: OpenRouter has a new frontend that looks sick, big ups [clinemay](https://discord.com/channels/1091220969173028894/1195014798837043240/1358883684609953812)!
   - One user joked that it looked like *gpt-3.5 made this website in about 4 minutes*.
- **Gemini models are top tier**: **Gemini 2.5 Pro** is on a whole other level compared to the other models, making it the most powerful model up to day.
   - One user noted it was rated as **1. gemini 2.5 pro** ... **10. everyone else**.
- **Free Model Limits Tightened, Community Reacts**: OpenRouter reduced the token limit for free models to **50**, triggering mixed reactions from users, with some expressing frustration over the lowered limit.
   - Some users feel that it's like a *paywall*.
- **API Keys Made Easier**: Users can now easily get an **API key** once they make an account, add credits then in the top right dropdown go to keys and create one there.
   - A community member said: *I was asking about the app so i could try to help you put the key in the right spot but not sure how Godot works iwth that*.
- **Nvidia Silently Drops SOTA-Level Reasoning Model Llama 3.1**: [Nvidia](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) silently dropped a SOTA-level reasoning model.
   - The new model casually showing it's better than **Behemoth**.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1358879030627143810)** (762 messages🔥🔥🔥): 

> `Augment, Vector DB vs graph DB, Manus.im, Cursor C/C++ extension error, Model selection` 


- **Daniel Mac goes Graph DB for Code**: A member shared a link to [Daniel Mac's tweet](https://x.com/daniel_mac8/status/1908332949251948808) about using a **graph database** for code querying.
- **Manus.im burns through credits**: A user reported that [Manus.im](https://manus.im) failed to answer a question correctly and burned through **984** of **1000 free credits** on a single prompt.
   - Another member suggested exploring alternatives like [Smithery.ai](https://smithery.ai/) or [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers).
- **C/C++ Extension Error**: A user reported receiving an error related to the **C/C++ extension** after using Cursor since its launch in March 2023, with the extension possibly limited to use with Microsoft products.
   - A workaround involved [rolling back to a previous version](https://forum.cursor.com/t/c-c-extension-usage-restriction-message-appears-in-cursor/75902) and users shared [other forum threads](https://forum.cursor.com/t/c-c-extension-broken/75182) discussing the issue.
- **Auto-Select is a scam**: Users are reporting that the **auto-select** model option is selecting trash models.
   - One user claimed it *fucked up my codebase*, while another suggested that it's intentionally designed this way.
- **Cursor's Free Tier Gets Heat**: A member reported that bypassing the trial version of Cursor could result in getting the user completely banned from using Cursor.
   - One user noted: *Gonna ban you now, but just so you know, I hope you didn’t like using Cursor because you won’t be able to use it at all soon.*


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1358908572959900011)** (158 messages🔥🔥): 

> `Llama 4 Disappointment, GPU requirements and model sizes, LM Studio and Ollama, Jinja templates` 


- **Llama 4 performance leaves users disappointed**: Users express disappointment with **Llama 4's** performance, describing it as *bad* and *10 steps backwards*, questioning the validity of benchmarks.
   - Others suggest that larger models may have quality control issues due to **random data**, **too many connections**, or **poisoned datasets**, while Qwen's **14B** models are praised.
- **LLM size and hardware implications**: A discussion arose regarding the relationship between **VRAM consumption** and model dilution, with some noting that models consuming less VRAM often appear more distilled or diluted to reduce size.
   - A user clarified that **Llama 4** gives similar results to **24-27B** models, but has speed and cost of **17B** model, but requires more vram making it pointless for simple users.
- **LM Studio's remote GPU compatibility is debated**: Users discussed connecting **LM Studio** to remote instances of **Ollama**, but it was confirmed that **LM Studio is not compatible with Ollama**.
   - Furthermore, the potential for connecting LM Studio with a remote GPU cluster was raised, alongside a discussion regarding the use of **Snapdragon X Series** NPUs and their (lack of) support with LM Studio and llama.cpp.
- **Cogito models' Jinja errors fixed with ChatGPT**: Users reported errors with **Jinja templates** when using **cogito-v1-preview-llama-3b**, and were advised to use **ChatGPT** to quickly fix the template.
   - The community model maintainer was notified about the wonky template and is expected to update the model.
- **Decoding MOE models for dummies**: A user asked, *what is an MoE model?*
   - A helpful member explained that **Mixture of Experts (MoE)** models can be faster than dense models, as only parts of the model are active per token, although the whole model must be in VRAM.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1358890575981449236)** (398 messages🔥🔥): 

> `Docker Bad, AMD ROCm WSL Woes, Memory Limits and Motherboards, Umbrella Rack SuperComputer, Fast Reading Skills` 


- **Docker Gets Roasted**: After one member expressed wanting to be *"best friends"* with anyone who says bad things about **Docker**, another member jokingly asked *"Did Docker take out your family or something?"*.
   - The first member humorously replied, *"My therapist said I shouldn't talk about it."
- **ROCm on WSL still problematic for 7800XT**: A user reported that ROCm via WSL doesn't work with a **7800XT** due to the lack of official support as seen in the [AMD documentation](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html).
   - Despite this, another user suggested it *might* work since both cards are **RDNA3** architecture, while the first user confirmed that it was *impossible* to get working half a year ago due to WSL passthrough issues.
- **Memory Limits Debated**: In a discussion about **RAM** limits, a user stated that a **Ryzen 7000** has a weak memory controller and that the **BIOS** limit is **192GB** on consumer hardware, while mainboards can fit **256GB**.
   - Another user pointed out that **AMD's website** states a **128GB** limit, to which the first user responded that people have been running **192GB** for years, and they attributed the discrepancy to server hardware having different quality targets.
- **Assembling an NND Umbrella Rack SuperComputer**: One user proposed building a **16-node supercomputer** with either **RTX 4090 D GPUs** (totaling 3TB VRAM) or a less powerful option (1.5TB VRAM), aiming for a **2T model with 1M context** within a budget cheaper than an **Nvidia DGX B300**.
   - Skeptics questioned the feasibility, with one user bluntly stating, *"this isn't how you do any of this...."*, highlighting the need for **RDMA**, fast interconnects, and skilled engineers, emphasizing that the user's goal was not possible on their current hardware.
- **Language Model fine-tuning educational project**: One member asked about a fun and educational project involving beefy hardware (**2 RTX ADA 6000s**, **512GB of RAM**) and asked if it's a good idea to learn to fine-tune a small instance of something like **phi4**.
   - Another member suggested **pretraining an LLM** from scratch or **fine-tuning an LLM** and pointed to a coding dataset from Nvidia ([huggingface.co](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)) and suggested that fine-tuning base models, not instruct ones would be better.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1359276120368742472)** (1 messages): 

> `Perplexity for Startups program, API Credits, Enterprise Pro` 


- **Perplexity Launches Startup Program**: Perplexity AI is launching a [startup program](https://www.perplexity.ai/startups) offering resources to help startups reduce research time and focus on building.
   - The program provides **$5000** in Perplexity API credits and **6 months** of Perplexity Enterprise Pro for the entire team; eligibility requires less than **$20M** in funding, being less than **5 years** old, and association with a Startup Partner.
- **Startup Program Details**: The Perplexity for Startups program aims to provide eligible startups with the resources they need to accelerate their development.
   - Eligible startups can receive **$5000** in API credits and a **6-month** subscription to Perplexity Enterprise Pro, enabling access to advanced AI capabilities for their entire team.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1358880405138178180)** (453 messages🔥🔥🔥): 

> `Gemini 2.5 Pro performance, Deep Research High rollout, Perplexity Discover tab, Manus Invites are still needed, AI image generation on Android` 


- **Gemini 2.5 Pro's Reasoning Outputs Missing**: Members discussed that **Gemini 2.5 Pro** doesn't expose its reasoning tokens, and therefore can't be included as a reasoning model on Perplexity, though it is a *high latency thinking model*.
   - Because **Gemini 2.5 Pro** reasoning tokens aren’t sent,  Perplexity via the API doesn't show the reasoning like you would via AI Studio, but is still a *high latency thinking model*.
- **Deep Research High Rolling Out Slowly**: Members are eagerly awaiting the rollout of **Deep Research High**, which is expected to use **150-200 sources** on average, however, one user reports *Perplexity's deep research got 23 sources, the free gemini deep research got over 500*.
   - Some members voiced frustration over the lack of communication regarding the release timeline, and the fact that the current version outputs a summary rather than conducting truly deep research.  Check out the [DeepSeek Subreddit](https://www.rxddit.com/r/DeepSeek/s/zFUYlP8NeV).
- **Gemini 2.5 Pro gives great performance on Perplexity**: Users noted the addition of **Gemini 2.5 Pro** in Perplexity, with one user finding that **Gemini 2.5 Pro**'s single story beat the other 3 stories by **GPT 4.5**, and another stating that it's now powering deep research, delivering detailed reports like [this one](https://cdn.discordapp.com/attachments/1047649527299055688/1359301884208480266/DR_2.5_Pro.pdf?ex=67f7a4c7&is=67f65347&hm=400f4c8d943d0887565453ecb42690e59499500547b64395af83ad45cadd3916&).
   - However, one user noted that answers are often truncated at **500-800 tokens**, despite the model generating **16,098 tokens** of detailed report.
- **Users report Perplexity auto-enabling Pro mode**: Several users have reported that Perplexity is auto-enabling Pro mode on free users to waste their daily limits.
   - One user said the non-pro model seems to be *balls*.
- **Reported issues when Uploading PDF files**: Pro user tried uploading 8 .pdf files and after 5 minutes of loading it either uploads one or two that instantly disappear with an error pop up saying *file upload failed*.
   - Files sizes ranges from **114kb to 9,502kb**


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1359158847511597308)** (1 messages): 

> `Llama 4, Benchmark Faking` 


- **Benchmark Faking Allegations against Llama 4**: A user shared a [Perplexity AI search result](https://www.perplexity.ai/search/does-llama-4-fake-benchmarks-pw9wkBJ4TCOUtdZu8fmTdg#0) questioning whether **Llama 4** is faking benchmarks.
   - The shared link provides a discussion and potential evidence related to the alleged benchmark manipulation by **Llama 4**.
- **Ongoing Debate on Model Benchmarking**: The conversation highlights the broader issue of transparency and reliability in AI model benchmarking, a recurring theme in the AI community.
   - Concerns were raised about the methodologies used to evaluate **Llama 4** and the potential for misleading results.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1358887341116100780)** (29 messages🔥): 

> `Perplexity API News Fetching, Perplexity API Sonar Prompting, Perplexity API Search Discrepancies, Perplexity API Citations, Perplexity API Sandbox` 


- ****Perplexity API** News Fetching**: A user requested a news API feature to fetch news based on queries or topics, similar to [particle.news](https://particle.news), and the team responded that they already have partnerships to surface news via their API.
   - A team member suggested building a news API feature using **Sonar's** existing functionalities and adding it to the [API cookbook](https://github.com/ppl-ai/api-cookbook).
- ****Perplexity API** Sonar Prompting**: A user reported issues with **Sonar**, where responses were focused around the system prompt rather than dynamically handling user queries.
   - A team member clarified that the system prompt isn't used during the search phase, advising the user to optimize the **user prompt** instead, referencing the [Prompt Guide](https://docs.perplexity.ai/guides/prompt-guide).
- ****Perplexity API** Search Discrepancies**: A user reported discrepancies between the **Perplexity API** and the **web UI** when summarizing web pages, with some links not being retrieved by the API and the results being less structured.
   - The user is seeking assistance to resolve the issues, as the **Sonar-pro** model yields different results between the **API** and the **web UI**.
- ****Perplexity API** Sandbox superiority?**: A user reported that the **API sandbox** is giving way better results than the actual API when using **sonar-reasoning-pro**.
   - The user is seeking advice on how to make the **API** give the same results as the **sandbox**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1358886363457388676)** (463 messages🔥🔥🔥): 

> `High Effort Mode, Manus Local Version, Genspark vs Manus, Llama 4 hype, Manus Credit Usage` 


- **Local Manus Coming Soon**: Members discussed that a local version of **Manus** will be possible in the future, like most other **AI models**.
- **MCP Servers Available**: Members noted that **MCP servers** are available on **Claude** since Nov 25/2024, and can be used with Claude code.
   - Some members expressed skepticism, citing successful past attempts at what they termed *cursed model merging*.
- **Llama 4 is Overhyped**: Users share that **Llama 4** is overhyped after testing it on **Openrouter.AI** and receiving subpar responses.
   - Others claim **Zucks** is receiving criticism because he supposedly gamed the **benchmarks**.
- **Octopus Web Scraper Works**: A member reported that [Octopus](https://octoparse.com/), a free website scraper, works pretty well on Zillow and Realtor, while Bardeen costs $130/month.
   - Another member said $130/month seems expensive when you could use **Manus** to build your own.
- **Manus credits are too expensive**: Several users complain that [Manus credits](https://www.manus.im/pricing) are too expensive, with one user reporting that a single Standard Complexity task used up all 1000 free credits and wishing the pricing and credit plans get updated.
   - Some users shared it is better to break it into smaller tasks with new dialogue windows, and also recommend **Proxy** as a cheaper alternative.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1358879728995668009)** (237 messages🔥🔥): 

> `Gemini 2.5 vs Sonnet Thinking, Aider's auto-testing, Gemini 2.5 Pro vs exp, OpenRouter citation links, AI resume builder` 


- **Gemini 2.5 and Sonnet Faceoff**: Members discuss **Gemini 2.5's** strong logic but poor instruction following versus **Sonnet's** feature-rich but less accurate coding, ultimately requiring fewer prompts for a working program.
   - One user reported that **Gemini 2.5** only needed 1 prompt, while **Sonnet** needed 3 prompts, despite Sonnet including *multiple file input methods, optional drag and drop, batch processing, file queue management, explicit conversion start, explicit cancellation, resizable window, etc*.
- **Aider's Auto-Testing Troubles**: A user is looking to enable **Aider's auto-testing** and potentially disable **auto-committing** due to issues with committing untested code, with a pointer to the [Aider configuration options](https://aider.chat/docs/config/options.html).
   - Another user suggests to provide a [model and key](https://aider.chat/docs/troubleshooting/models-and-keys.html) or Aider will guess what you want based on whatever of your keys it can find.
- **Gemini 2.5 Pro exp Rate Limits**: Users compare **Gemini 2.5 Pro exp** to **Gemini 2.5 Pro preview**, noting different rate limits, and one reports getting charged for the seemingly free `pro-exp` model.
   - Despite one user feeling like exp is weaker, another user had *cancelled Sonnet within an hour of using it*, while another user got rate limit issues on both, especially through openrouter.
- **OpenRouter Missing Citation Links**: A user asks if missing citation links are normal when using services like **Perplexity Sonar Pro** through **OpenRouter**, attaching an [image](https://cdn.discordapp.com/attachments/1131200896827654149/1358926629170319490/image.png?ex=67f798cc&is=67f6474c&hm=fe2c340b866bec81e485bbed3c2d1fe17071b540d6ea5c803306211e3d9f2ceb&).
- **DIY AI Resume Builder Idea**: A user is seeking an **LLM-powered tool** to analyze resumes against job listings, suggesting wording modifications, and another user suggests building one's own tool.
   - A user suggests if they had some programming experience that this could be built, and they could also use it to test out Gemini 2.5 pro.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1358969628058976366)** (8 messages🔥): 

> `Architect mode interruptions, Aider Response Time, Aider Cursor Rules` 


- **Architect Mode Edits Getting Cut Off?**: Users are reporting **/architect mode** edits getting interrupted when asked to add new files, potentially losing the editor.
   - Saying no to adding new files appears to allow the edit to continue.
- **Aider Response Time Questioned**: Users are reporting that **Aider v0.81.1** with `openrouter/deepseek/deepseek-r1` and `openrouter/anthropic/claude-3.5-sonnet` is as slow as ChatGPT.
   - One user waited *"5 freaking minutes"* for a schema file to be created, only to receive a `litellm.APIError` due to a connection issue.
- **Comparing Aider Conventions to Cursor Rules**: Users are asking if Aider conventions are similar to [Cursor rules](https://roman.pt/posts/cursor-under-the-hood/).
   - A member clarified that *aider "conventions" isn't really a thing*, but just added context sent to the LLM. They are added manually or with `--read CONVENTIONS.md`.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1359034687086923989)** (8 messages🔥): 

> `Software Engineer Gap Year, LLMs as AI Coworkers, Programming LLMs for Successful Outcomes` 


- **Gap Year Not a Good Idea for Software Engineers?**: An article suggests that taking a gap year/holiday would be an incredibly bad decision/time for software engineers, pointing to insights about the current tech landscape, see [this article](https://ghuntley.com/dothings/).
- **LLMs as 1000 AI Coworkers**: Anni Betts from Anthropic suggests that software engineers should think beyond having *"an AI coworker"* and instead consider having *"1000 AI coworkers that went ham on your entire issue backlog at once"*.
   - According to the author this can be done by [programming the LLMs](https://ghuntley.com/stdlib/) and building a *"stdlib"* that manufactures successful LLM outcomes.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1358936328233943241)** (10 messages🔥): 

> `NotebookLM Commercial Options, NotebookLM privacy assurances, NotebookLM Misreading Scholarly Articles` 


- **Google's AgentSpace unlocks NotebookLM for Enterprise**: A user inquired about a commercial-scale version of **NotebookLM** with data privacy and specific programming capabilities, and another member linked to [Google's AgentSpace NotebookLM Enterprise documentation](https://cloud.google.com/agentspace/notebooklm-enterprise/docs/set-up-notebooklm#cmek) that enables **CMEK**.
   - The documentation outlines how to set up NotebookLM with **Customer-Managed Encryption Keys (CMEK)**, offering greater control over data encryption.
- **Privacy Assurances Provided by NotebookLM**: A member explained that both the **Enterprise** and **Plus** versions of **NotebookLM** offer privacy assurances, emphasizing that user data is never in the public domain, regardless of the version used.
   - They clarified this point to address a fundamental misunderstanding of **Google's privacy policy** and **terms of service**, and further suggested that the platform has mechanisms to prevent prompt injection attempts.
- **NotebookLM's improved Summary after user correction**: A user noticed that **NotebookLM** initially misread a critical point in a scholarly article's summary but corrected itself after the user provided a quotation and explanation.
   - Repeating the same prompt with the same article in different **Google accounts** yielded the correct summary from the beginning, raising questions about whether the model uses previous queries for training and whether the privacy statement is accurate, according to the user.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1358885964021498016)** (204 messages🔥🔥): 

> `Discovery Mode rollout, Google Cloud Next and Google I/O, NotebookLM Legal Use cases, New Gemini features with deep research, Podcast Audio Overviews` 


- **Discovery Mode Still Rolling Out Slowly**: Users report waiting for the new **Discovery Mode** feature, with the rollout expected to take up to **two weeks** from release date.
   - One user jokingly demanded *special treatment as a Google fanboy*, requesting to be an alpha tester.
- **Google Cloud Next and Google I/O Promise Surprises**: The upcoming **Google Cloud Next** and **Google I/O** events are anticipated to reveal new features, though details remain tightly guarded.
   - One user humorously compared Cloud Next to *Christmas*, with Google acting as Santa.
- **NLM for Legal Use Cases and Printing Concerns**: A user sought advice on using NotebookLM for extracting specific information from legal documents, aiming to get article numbers and relevant text, seeking assistance on printing the entire answer with all the links included.
   - Another member suggested breaking content into **10-20 notebooks**, each with its own specific content, to ask the same question immediately in each notebook.
- **Gemini Still Hallucinates with Deep Research**: Some users report experiencing *hallucinations* with **Gemini's deep research**, despite it having access to the internet.
   - One member clarified that **Gemini** can connect to Google Search, but it doesn't do it if you don't specify you wanna ground it and recommends testing this in **AI Studio**.
- **Podcast Audio Overviews Coming to NotebookLM**: It was reported that the new **2.5 Pro deep research** version will have the capability to make **audio overviews**, but it is not working for all users.
   - A Google employee clarified that complex topics with several different angles covered in the sources around a central topic result in longer podcasts.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1358880373685354668)** (92 messages🔥🔥): 

> `DeepSeek R2 Release, LlamaCon, Llama-4-Maverick, Style Control Ranking, HF version of Llama-4-Maverick` 


- **DeepSeek R2 must release for LlamaCon**: Members are encouraging those with connections at **DeepSeek** to release **R2** on the same day as **LlamaCon** to leverage the hype, citing that the training data needed for **MoE** is different than base models according to research from [arxiv.org](https://arxiv.org/abs/2410.19034).
- **LM Arena policy updates**: Early analysis shows **style and model response tone** was an important factor (demonstrated in style control ranking), and the HF version of **Llama-4-Maverick** is being added to Arena, but **Meta** should have made it clearer that *Llama-4-Maverick-03-26-Experimental* was a customized model to optimize for human preference, so leaderboard policies are being updated to reinforce commitment to **fair, reproducible evaluations**.
   - Members reacted saying it was a *yapping emoji slopfest*.
- **Cogito models released under open license**: Strong **LLMs of sizes 3B, 8B, 14B, 32B and 70B** are being released under open license, with each model outperforming the best available open models of the same size from **LLaMA, DeepSeek, and Qwen**, across most standard benchmarks and the **70B model** outperforming the newly released **Llama 4 109B MoE model**.
   - These **LLMs** are trained using **Iterated Distillation and Amplification (IDA)**, an scalable and efficient alignment strategy for superintelligence using iterative self-improvement from [DeepCogito](https://huggingface.co/collections/deepcogito/cogito-v1-preview-67eb105721081abe4ce2ee53).
- **Together AI moves into Training**: **Together AI** is getting into the training business, as showcased by this [case study](https://www.together.ai/models/cogito-v1-preview-llama-70b).
- **Google Gemini 2.5 Pro Deep Research Announced**: **Google Gemini 2.5 Pro Deep Research** was announced according to [9to5Google](https://9to5google.com/2025/04/08/gemini-2-5-pro-deep-research/) with a member reporting that Gemini 2.5 deep research is roughly on par with **OpenAI Plus** with an audio overview podcast option thing.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1359001640761294910)** (30 messages🔥): 

> `OpenAI Image Gen Capabilities, Logprob Reward, Arxiv Publishing, Arxiv Moderation, Phi-CTNL` 


- **OpenAI's New Image Gen Capabilities**: A member inquired about write-ups describing how **OpenAI** unlocked new image generation capabilities, suggesting it wasn't a new model but *latent capabilities*.
   - Another member suggested it was achieved using an objective similar to [this one](https://discord.com/channels/1179127597926469703/1208183216843005962/1358810240627376259) and incorporating a *logprob reward* as seen in [this paper](https://arxiv.org/abs/2503.19618).
- **Arxiv Posting Process Revealed**: Members discussed the process for posting on **Arxiv**, noting that a *small vouch* is needed, which differs from the old physics days of just dumping content.
   - They added that a *small random chance of moderation* exists, where even nonsense papers can get rejected, but people mostly just post whatever anyway.
- **Champion Phi-CTNL Paper has 20 Citations**: A member shared a link to [this paper](https://arxiv.org/abs/2309.08632) describing a champion model that has 20 citations, exclaiming *Godlike.*
   - Another member noted the *brutal* model name **phi-CTNL**, speculating on the reaction if **Meta** cites it in a future **Llama 4** paper.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1359173566054142232)** (15 messages🔥): 

> `Google AI Staff, AI Sabbatical, NVDA Tariffs, ASI, Google's management vibes` 


- **Google allegedly pays AI staff to do nothing**: A [TechCrunch article](https://techcrunch.com/2025/04/07/google-is-allegedly-paying-some-ai-staff-to-do-nothing-for-a-year-) discusses how **Google** is allegedly paying some **AI staff** to do nothing for a year rather than join rivals.
   - A member described it as the *most basic idea from management where all the second order effects are horrifically bad*.
- **AI Engineer eyes Sabbatical to look at trees**: One member expressed that after AI settles down in a year or two, they'd happily take a **sabbatical** and write a book.
   - Another member said that *from the corporate side (McKinsey etc.) they give their researchers very long time off to make sure they stay engaged otherwise they found you just lose everyone over time*.
- **Tariffs May Accelerate AI Slowdown**: A member suggested that if **tariffs** stay, the AI field will settle down in a month because people can't afford **NVDA GPUs**.
   - They stated *if tariffs stay it'll settle down in a month as you can't afford NVDA GPUs*.
- **Google Paying Quitters Creates Legal Peril**: A member clarified that **Google** is paying people who have quit for another year but forcing them not to work.
   - In theory, anything they do in that year belongs to **Google**, so they can't start working on their startup or something without legal peril.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1359020372875284723)** (12 messages🔥): 

> `Google Cloud Next, Qwen 3 Launch, GPT 4.5 preferences, Claude Code Credits, Tim Apple` 


- ****Google Cloud Next** to Drop New Models**: A member shared that Google will drop new models on **Cloud Next**, which starts Wednesday, according to [this X post](https://x.com/OfficialLoganK/status/1909443890366890200).
   - This may mean the launch of **Qwen 3**.
- ****GPT 4.5** Preferences Underway**: A member alluded to **GPT 4.5** preferences being collected by OpenAI, linking to [this X post](https://x.com/phill__1/status/1909623249563959551).
   - They were looking for the *High Taste Tester LMarena* to weigh in.
- ****Anthropic** offers **Claude Code Credits****: A member shared a link to **Anthropic** offering [$50 in Claude Code credits](https://www.anthropic.com/contact-sales/claude-code-credits) to 1,000 people for trying **Claude Code**.
   - According to the post, that may be enough credit *to change one var name*.


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1358978645632352450)** (5 messages): 

> `Jiankui He's X ad revenue` 


- ****Jiankui He's** AdSense fortune**: A user joked about **Jiankui He** making money from the [X creator ad share](https://x.com/Jiankui_He/status/1909417417396437200).
   - Another user joked that he makes *$20* from that, or *$20K* if **Elon Musk** wants to "fuck him."
- **AdSense Revenue Speculation**: Speculation arose regarding the potential ad revenue earned by **Jiankui He** on X.
   - Estimates ranged from a modest *$20* to a more substantial *$20,000*, contingent on **Elon Musk's** intervention.


  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1358999578468417588)** (24 messages🔥): 

> `DAPO papers, OLMo, Tulu 3, BoN Sampling` 


- **DAPO papers offer "Extreme value"**: Members in the channel discussed [a DAPO paper](https://arxiv.org/abs/2504.05118) as offering *"Extreme value"*.
   - They also referenced [another paper built on OLMo](https://arxiv.org/abs/2504.04022).
- **Tulu 3's work makes it into another paper**: A paper using **Tulu 3's work** was mentioned and linked: [https://arxiv.org/abs/2504.03790](https://arxiv.org/abs/2504.03790).
- **Alpha in research = talk to other researchers**: A member stated that *"biggest alpha in research is just talking to other researchers"* and shared some insights from a paper, noting it *"uses a very different method of inference time compute"*.
   - They also stated that it *"shows that **BoN sampling** is effectively just changing the beta factor in **RLHF** (lowering the KL penalty)"*, and that *"you can design the inference time compute differently, so that you aren't hacking RL (in this case using an RM as guidance) and get far better answers"*.
- **BoN sampling to sub in future work?**: A member asked whether **BoN sampling** could substitute in future work.
   - Another member responded that *"it’s more complicated to implement but sure why not if it’s flop equivalent"*.
- **Ash Vaswani Tweet on Undergrad Technical Report**: A user shared a [tweet from Ash Vaswani](https://x.com/ashVaswani/status/1909642828554387675) and stating that a linked paper *"didn't give very good vibes though"*.
   - The member stated that the paper *"felt very undergrad technical report"*, but declined to tweet negatively about it.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1358891406877266041)** (3 messages): 

> `Karan Dalal Post, Yuxi Liu Essay` 


- **Karan Dalal's Post Goes Viral**: A member shared a link to [Karan Dalal's post on fxtwitter](https://fxtwitter.com/karansdalal/status/1909312851795411093?s=61), generating excited discussion.
   - The original poster reacted with simply, *"WTF"*.
- **Yuxi Liu's Essay Gains Attention**: A member posted a link to [Yuxi Liu's essay](https://yuxi-liu-wired.github.io/essays/posts/cyc/) prompting immediate discussion.
   - No specific details about the essay were mentioned.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

natolambert: My post looks generous next to Marcus’s, oh my
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1358920108726030640)** (106 messages🔥🔥): 

> `Adam second-moment estimate buffers, Google DeepMind Patents, Hierarchical Perceiver, AI Auditing Survey, GFlowNets` 


- **Adam Buffers Reconstruction Discussions Emerge**: Members discussed the utility of **Adam second-moment estimate buffers** and how to efficiently reconstruct them for open-source models, balancing accuracy with computational cost, for potential method improvements.
   - It was noted that setting **beta2** to a high value (e.g., 0.999999) and the learning rate to zero could improve accuracy, though the final epoch of pretraining presents challenges.
- **DeepMind Patents Hierarchical Perceiver**: Members noted that [Google DeepMind](https://www.freepatentsonline.com/y2025/0103856.html) is trying to patent the **Hierarchical Perceiver**, drawing comparisons between the patent diagrams and those in the original [research paper](https://arxiv.org/abs/2202.10890).
   - Some speculated this patent could be related to DeepMind's work on **ultra-long context lengths** in Gemini, with discussions on whether it's a defensive measure or indicative of current usage after its original lack of uptake.
- **Licensing Faceoff: Apache 2.0 Prevails over MIT**: The conversation mentioned a preference for the **Apache 2.0 license** over MIT, citing its defenses against patent-based lawfare in machine learning.
   - It was highlighted that institutional inertia and GitHub org settings favored Apache 2.0, with the sentiment that *outside of GPLv2 weirdness or wanting to engage in lawfare shenanigans, there's no reason to argue for MIT over Apache 2.0*.
- **DeepMind Rumored to Sandbag Model Releases**: Members discussed a rumor, per a [Reddit thread](https://old.reddit.com/r/LocalLLaMA/comments/1jp1555/deepmind_will_delay_sharing_research_to_remain/), that **DeepMind** may be delaying the release of research to maintain a competitive edge.
   - One participant clarified that *sandbagging* refers to *holding back in ability*, not releasing purposely bad versions of models to mislead others.
- **Survey seeks AI Auditing Experts**: A researcher from the University of Turku, Finland, is conducting a survey on ethics-based auditing of generative AI systems and is seeking participation from professionals with practical experience in AI auditing, model evaluation, risk/compliance, or ethical alignment of AI principles.
   - The [survey](https://link.webropolsurveys.com/S/AF3FA6F02B26C642) aims to gather insights on auditing or evaluating AI systems, especially generative models.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1358891732053528637)** (35 messages🔥): 

> `QKNorm, Soft RL, Llama 4 Memorization, Critical Batch Size, Reward, Value, Q-value letters` 


- ****QKNorm** Developments Deemed Dubious**: A member suggested a [better/earlier paper](https://arxiv.org/abs/2502.00919) and stated that the **QKNorm developments** are not the right way to go, referencing [this paper](https://arxiv.org/abs/2503.05453).
- ****Soft RL** Goal Gleaned**: A member summarized that the goal of **Soft RL** *is to learn a policy that not only knows a good response to every query, but ideally knows all good responses to every query.*
   - They linked to [test-time-training.github.io/video-dit/](https://test-time-training.github.io/video-dit/) and [this tweet](https://x.com/karansdalal/status/1909393559981375574) while mentioning thread block clusters.
- ****Llama 4** Lacks on MATH-Perturb**: In a discussion about measuring memorization of test sets in the **Llama 4** models, a member stated that *it performs pretty badly on the MATH-Perturb dataset* and linked to [this tweet](https://x.com/KaixuanHuang1/status/1909387970773234088).
- ****Critical Batch Size** Critiqued**: Regarding the statement that *very large batch sizes are not good for convergence*, a member cited the standard **McCandlish paper** on critical batch sizes to back up the statement and linked to [this paper](https://www.cerebras.ai/blog/training-multi-billion-parameter-models-on-a-single-cerebras-system-is-easy).
- ****R** stands for Return, Remarks a Redditor**: A member joked that *one day llm researchers will use the correct letters out of R, V, and Q to represent reward, state-value, and state-action values respectively but not today* while linking to [this paper](https://web3.arxiv.org/abs/2503.19037).
   - Another member responded *Trick question, R stands for Return* alongside a link to [this paper](https://arxiv.org/abs/2504.01928v1).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1358894123033170021)** (9 messages🔥): 

> `Baranuik and Balestriero's works, ReLU networks, Boris Hanin's ReLU networks paper, ICML machine unlearning workshop` 


- **Hyperplane Happy Neural Nets Hedge Overfitting**: It was noted that because **ReLU neural nets** work by carving the input space along hyperplanes, they have an implicit bias against overfitting that gets better in high dimension.
   - It takes at least *d+1* hyperplanes to enclose a bounded set, so a perfectly overfitted model enclosing each datapoint in a separate bounded set would need at least *n*(d+1) neurons.
- **Hanin's Hyperplane Handling Helpful Hints**: A member shared a [link to Boris Hanin's paper](https://arxiv.org/abs/1906.00904) which demonstrates some mathematical properties of **ReLU networks**, specifically studying the geometry of their constant regions.
   - Another member expressed their love for a specific figure in the paper.
- **ICML Invites Insightful Investigation Into Unlearning**: A member shared that [ICML](https://icml.cc/Conferences/2024) will have a **machine unlearning workshop**.
   - The workshop's website can be found [here](https://mugenworkshop.github.io/).


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1359005284605100172)** (1 messages): 

> `LM Harness, HotpotQA, Llama Eval, GPT Models` 


- **Guidance Needed: LM Harness for HotpotQA**: A member inquired about a **LM harness implementation for HotpotQA** to evaluate **Llama** and **GPT models**.
   - They requested guidance on running evaluations against **HotpotQA**.
- **Llama and GPT models under eval**: Members are evaluating **Llama** and **GPT models**.
   - They require a **LM harness implementation for HotpotQA** to do so.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1358927244734500874)** (127 messages🔥🔥): 

> `Llama-4-Scout-17B, Gemini 2.5 Pro Code Generation, aider-chat & Gemini 2.5 Pro, HiDream-I1 Image Model, DeepCogito LLMs & IDA` 


- **Llama-4-Scout-17B Gets Ready for llama.cpp**: [Llama-4-Scout-17B text-to-text](https://github.com/ggml-org/llama.cpp/pull/12791) support has been added to *llama.cpp*, with members working on converting and quantizing the model.
   - This pre-release is generating excitement among users eager to test its capabilities.
- **Gemini 2.5 Pro Generates Solid Code Snippets**: **Gemini 2.5 Pro** is being lauded for its ability to generate functional code snippets from complex prompts. See the prompt and responses in [this message](https://cdn.discordapp.com/attachments/1149866623109439599/1358975415426879589/message.txt?ex=67f7c63b&is=67f674bb&hm=1c655347ddb71efc0e03a079e62d8e26286724363242370cf6f19b9e50cc1980&).
- **aider-chat plus Gemini 2.5 Pro Creates AGI Prototype**: A user reported using **aider-chat** combined with **Gemini 2.5 Pro** to edit or create 15 files from a **300k token context**, including their frontend, API, and microservices.
   - The user feels like they now have all the files to deploy a production AGI prototype.
- **HiDream-I1 Image Model Generates High-Quality Images**: **HiDream-I1** is a new open-source image generative foundation model with **17B parameters** using **Llama 3.1 8B** as a text encoder, released under the [MIT license](https://huggingface.co/HiDream-ai/HiDream-I1-Full) that achieves state-of-the-art image generation quality within seconds.
   - It *produces exceptional results across multiple styles including photorealistic, cartoon, artistic, and more, achieving state-of-the-art HPS v2.1 score, which aligns with human preferences*.
- **DeepCogito Models use Iterated Distillation and Amplification**: A new suite of **Cogito** models (**3B-70B**) claim to outperform same-size models like **Llama, DeepSeek, and Qwen**, and are trained using **Iterated Distillation and Amplification (IDA)**, which iteratively improves a model's capabilities through cycles of amplification and distillation as outlined in [this research](https://www.deepcogito.com/research/cogito-v1-preview).
   - Notably, the **70B model** allegedly surpasses the newly released **Llama 4 109B MoE model**.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1358919204622831677)** (4 messages): 

> `LayerNorm Implementation, Llama4 Context Window, H100 Usage` 


- **LayerNorm Stats Calculated Per Sample**: A member implemented **LayerNorm**, noting the key difference from **BatchNorm** is computing statistics *per sample* (**axis=1**) instead of *per batch*, with **keepdims=True** to avoid operand issues.
   - They also removed running averages since mean and variance depend on the number of features, not batch size, and attached [an image](https://cdn.discordapp.com/attachments/1154120232051408927/1358919204270641342/image.png?ex=67f791e1&is=67f64061&hm=3aeae5b8f48b37b2e22dacbdc7c0fe25279704c08caf5ff4cdbf3df8a01acf2a&) showcasing it.
- **Llama4 needs H100?**: A member inquired about testing **Llama4** with a **10M context window** on a single **H100**.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1358913098085302273)** (18 messages🔥): 

> `Distributed data parallel training, Untrusted low-cost compute, Nous DeMo paper, Gradient compression algorithm, P2P interruptible compute` 


- **Panthalia Platform Verifies Low-Cost Compute with DDP**: A platform has been developed to verify untrusted, low-cost compute for training models over the internet using distributed data parallel (DDP), inspired by the **Nous DeMo** paper for compression, with a waitlist available via [X.com](https://x.com/panthaliaxyz/status/1909342585505669228).
   - The platform uses a gradient compression algorithm, documented [here](https://docs.panthalia.com/gradient-compression-algorithm), with code available on [GitHub](https://github.com/ritser-labs/panthalia-worker/blob/main/spl/util/demo.py).
- **Panthalia Aims to Resell H100 Compute at $0.60/hr**: In its early stages, **Panthalia** aims to resell low-cost provider compute at interruptible prices, such as **$0.60/hr for an H100** and **$0.13/hr for a 4090**, leveraging **DDP** and **DeMo-style compression**.
   - The weights are stored on reliable servers, enabling scaling for initial users, and long-term plans include building a supply of **P2P interruptible compute**.
- **Panthalia Enables User-Defined Training and Plugins**: The platform supports model sizes limited only by device capacity, using **DeMo compression** to achieve significant reduction in size, with a [plugin system](https://docs.panthalia.com/buying-compute/create-a-plugin) allowing users to define their own models, training methods (**QLoRA**), and distributed training algorithms (**DeMo vs. DiLoCo**).
   - Users can download weights, and compute units can be standardized within subnets to ensure validation, with Stripe (credit card) for payments and crypto for payouts.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1358953132406739005)** (10 messages🔥): 

> `GPUMODE triton dataset, PyTorch version for triton kernels, GPUMODE website improvements, GPUMODE Job Portal` 


- ****GPUMODE Triton Dataset:** Genesis on PyTorch 2.5**: The **GPUMODE** "triton" dataset, used for [Inductor Created Data](https://huggingface.co/datasets/GPUMODE/Inductor_Created_Data_Permissive), was created using **PyTorch 2.5**.
   - The creator promised to update the readme to reflect this crucial detail, since users may have issues running it on **PyTorch 2.6+**.
- ****GPUMODE Website**: New Tab Navigations Suggested**: A user suggested that the "Lectures" and "Resources" tabs on the **GPUMODE** website should open in a new tab, since they are hyperlinks to YouTube/GitHub.
   - This would prevent users from navigating away from the **GPUMODE** website in the same tab, thus *improving user experience*.
- ****GPUMODE**: Job Portal Idea Scraped**: A member proposed adding a job portal to the **GPUMODE** website, which would scrape postings from a specific channel, to create new postings.
   - They also suggested a **static template** (JSON or YAML) for job posters to ensure consistent formatting and simplify entry creation, and the GPUMODE staff have acknowledged the suggestion.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1359114922361094144)** (14 messages🔥): 

> `block_ptr usage, tl.load and boundary_check, Boundary checks and performance` 


- **Block Pointers to fill out-of-bounds**: A member suggests using `tl.make_block_ptr` to create pointers that can fill with zeros for out-of-bounds memory accesses, specifically highlighting usage with **`boundary_check`** and **`padding_option="zero"`**.
   - The usage example provided utilizes **`tl.make_block_ptr`** with parameters like **`shape`**, **`strides`**, **`offsets`**, **`block_shape`**, and **`order`** to create the pointer, and then loads data using **`tl.load`** with boundary checks.
- **`tl.make_block_ptr` deep dive**: A member inquired about **`tl.make_block_ptr`**, asking if it could be spammed in a loop, how to use the offset parameter, and the meaning of the order parameter.
   - Another member clarified that **`tl.advance`** should be called to increment the pointer for loading data in a loop, the offset represents the start element index, and the order parameter defines the memory layout (e.g., col-major matrix).
- **`boundary_check` order is irrelevant, but required for correctness**: A member asked about the meaning and behavior of `boundary_check` in **`tl.load`**, specifically its order and the consequences of omitting it.
   - It was explained that the order of `boundary_check` doesn't matter, and omitting it increases speed, but risks errors like *"device-side assert triggered"* due to potential buffer overruns, especially when array dimension is not a multiple of the block size.
- **Can you fill with another value than zero or nan?**: A member asked if it was possible to **fill with a value other than zero or NaN** when using block pointers.
   - Another member answered that it is difficult to do, but you can replace NaN with another value by using **`tl.where(x == x, x, another)`** because `nan != nan`.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1358978121948463234)** (4 messages): 

> `Deepseek communication library, NVSHMEM and Unified Virtual Addressing (UVA), LDSM (Local Data Share Memory), Optimized smem load` 


- **Deepseek Lib Built off NVDA's NVSHMEM**: The deepseek communication library is built off **NVSHMEM** library from NVDA.
- **NVSHMEM Explored for UVA Intra-Node Comm**: A member inquired whether **NVSHMEM** uses **Unified Virtual Addressing (UVA)** for intra-node communication, enabling peer-to-peer loads/stores to remote GPUs via NVlink.
- **LDSM Copying Discussed**: A user asked for the code for defining `make_tilded_copy`, and stated that the current image does not look like one from a `tiled_mma`, instead it looks like an **LDSM** copy.
   - One explained that with **LDSM**, 32 threads in a warp coordinate to copy data from smem to rmem, and if the smem is row-major, T0 loads from 0-7 from smem, and stores 0,1,8,9,128,129,136,137 into its own registers memory.
- **Optimized smem load**: A member shared a code snippet, `tCsA = thr_mma.partition_A(sA); tCrA = thr_mma.make_fragment_A(tCsA); copy(tCsA, tCrA);`, for partitioning `sA` according to the `tiled_mma`.
   - They added that for optimized **smem load** we should use **LDSM** though.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1358940322679361676)** (9 messages🔥): 

> `TorchTitan's Compile Strategy, FSDP Numerical Issues, FSDP2 Model Extraction` 


- **TorchTitan's Pre-Compile Strategy**: The standard practice is usually to compile after operations, but **TorchTitan** does a unique per-block compile before, potentially to circumvent some **torch compile bugs**; see [torchtitan/parallelize_llama.py#L313](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313).
   - The block-wrapping approach aims to leverage Dynamo's caching to skip **Triton's LLVM** compilation, which is slow, however, numerical issues may still exist when using `torch.compile` and FSDP together.
- **FSDP and torch compile cause Numerical Issues**: A research lab experienced numerical issues using **FSDP** with `torch.compile`, leading to training instability, where the reward would suddenly plummet.
   - They discovered that disabling `torch.compile` resolved the issues, and cautioned to *be careful with torch compile*, highlighting that these problems were observed with **HF qwen2.5** and a custom **GRPO+entropy loss**.
- **The wrapped model extraction from FSDP2 remains challenging**: A member asked how to get the original model from an **FSDP2** wrapped model because the modifications are done in place and `copy.deepcopy` isn't implemented in **FSDPModule**.
   - Another member suggested that **FSDP** modifies the model in place by wrapping modules, recommending keeping the original model around before applying **FSDP**.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1359155079055282206)** (5 messages): 

> `CUDA physics simulation kernels go open source, Triton-Distributed, SMERF 3D` 


- **PhysX Goes Public!**: NVIDIA's **CUDA physics simulation kernels** are now [open source](https://github.com/NVIDIA-Omniverse/PhysX/discussions/384); some are already working on a **ROCm** version.
- **Triton Gets Distributed Superpowers!**: A learning note details **Triton-Distributed**, fusing Triton with **NVSHMEM/ROC-SHMEM** to enable multi-GPU execution, add IR for distributed tasks, and support compute-communication overlap ([link](https://x.com/thatperfguy/status/1909360454465433831)).
- **SMERF's Berlin Demo is Still Cool**: The **SMERF** (**Scalable Modelling of Explicit Radiance Fields**) project's [Berlin demo](https://smerf-3d.github.io/select_quality/?scene=berlin) remains impressive for its **3D scene reconstruction** capabilities; the project page is [here](https://smerf-3d.github.io/).


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1359031721986883726)** (2 messages): 

> `Krea hiring, ML engineers, GPU cluster, diffusion models, interns` 


- ****Krea** Seeks ML Engineers for **GPU** Brilliance!**: **Krea** is [hiring ML engineers](https://jobs.ashbyhq.com/krea) to optimize the training/inference pipeline for their **GPU** cluster, seeking individuals passionate about accelerating image generation models.
- ****Krea** needs Researchers for diffusion models**: **Krea** is also seeking researchers interested in enhancing the controllability and aesthetics of diffusion models.
- **Inquiries Emerge for Internship Spots**: A member inquired about potential internship openings at **Krea**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1358916933272994056)** (15 messages🔥): 

> `Graph Neural Networks (GNNs), Graph Attention Networks (GATs), CUDA compilation of C code, NVIDIA Streaming Multiprocessors, Thread cooperation in CUDA` 


- **GNN Computations are Radically Parallel**: Members discussed the parallel nature of **Graph Neural Networks (GNNs)**, noting that updates for each node in a graph can often be computed in parallel.
   - One member mentioned that **Graph Attention Networks (GATs)** architecture is one such example that comes to mind.
- **C++ Compilers may fail on valid C code**: Members discussed the claim that **C++** compilers can compile all **C** code, referencing the server's FAQ.
   - One member pointed out that it's possible to write **C** code that does not compile with a **C++** compiler, citing [a Wikipedia article](https://en.m.wikipedia.org/wiki/Compatibility_of_C_and_C++).
- **CUDA Glossary Updated**: A member suggested including a screenshot from Hennesy & Patterson that breaks down terminology into a common ground (Ex: **NVIDIA Streaming Multiprocessors = Cores**).
   - Another member suggested adding it as a suggestion in the [glossary](https://docs.google.com/document/d/1xNRvBJS1CPurxGESSRljGCS0fetpIOafd22JV8D4Ufg/edit?tab=t.0), or post it in the channel.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1359317987814936727)** (1 messages): 

> `torchao 0.10.0 release, MXFP8 training, PARQ, Module Swap Quantization API, Low Bit Kernels` 


- **TorchAO Drops New Release: v0.10.0**: The **0.10.0 release** of **torchao** introduces end-to-end training support for **mxfp8** on **Nvidia B200**, along with **PARQ** for quantization-aware training.
   - This release also includes a **module swap quantization API** for research and updates for **low-bit kernels**, with details available in the [release notes](https://github.com/pytorch/ao/releases/tag/v0.10.0).
- **Nvidia B200 can use MXFP8 training**: **MXFP8** is now supported for end to end training on **Nvidia B200** due to the updates from the **torchao 0.10.0 release**.
   - These training capabilities will allow for better and faster quantization aware training and new research.
- **TorchAO releases Module Swap Quantization API for Research**: The new **Module Swap Quantization API** will enable researchers to effectively apply quantization to custom modules in models.
   - The **torchao 0.10.0 release** enables researchers to experiment with quantization strategies more flexibly, by swapping standard modules with quantized versions.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

twzy: met yann lecun today and he seemed pissed
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1358899608889327776)** (9 messages🔥): 

> `Tom and Jerry Diffusion Transformers, Nvidia Hopper Distributed Shared Memory, Verifying Untrusted Low-Cost Compute, LiveDocs Code Documentation` 


- **Toon Time: Team Triumphs with Tom & Jerry Transformer**: A team completed a project creating 1 minute long **Tom and Jerry** cartoons by finetuning a **diffusion transformer**, and their work was accepted to **CVPR 2025** and released their [finetuning code on Github](https://github.com/test-time-training/ttt-video-dit).
   - They also released a [sample video](https://cdn.discordapp.com/attachments/1358899608889327776/1358907723433115749/homeless.mp4?ex=67f78730&is=67f635b0&hm=3b64ef6ea758875651b8faacd7f4e0ad769cfbe8488c5ea07b1511958c608660&) of the unedited output, fully generated by the diffusion transformer.
- **Hopper's Hidden Hardware Helps High-Performance RNNs**: A member mentioned that on **Nvidia Hopper** architecture, there is a really interesting feature where you can have **distributed shared memory** to transfer data directly between SRAM of SMs.
   - They used this feature to run **tensor parallelism** across SMs of their RNN's hidden states on a single GPU to remove the need of writing back to HBM.
- **Panthalia Platform Provides Proof of Trustworthy Parallelism**: A member has been working on a platform that verifies untrusted low-cost compute to train models over the internet with **distributed data parallel** [as described here](https://x.com/panthaliaxyz/status/1909342585505669228).
   - They use a compression algorithm heavily inspired by the **DeMo paper** ([docs](https://docs.panthalia.com/gradient-compression-algorithm)).
- **LiveDocs Launches Legit Lookin' Logistics**: The creator of **LiveDocs** invites users to *document your code* with their upgraded service, now with more features available via signup at [www.asvatthi.com](http://www.asvatthi.com).
   - Included was an image of the interface, showing off various code documentation pages.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1359233590935949443)** (1 messages): 

> `AlphaGeometry, KernelBench, GPU kernel generation` 


- **KernelBench Boasts Bootstrapping GPU Kernels**: A member mentioned prior work on using verifiers to bootstrap **GPU kernel generation** capabilities through test-time compute scaling, referencing experiments in [KernelBench](https://arxiv.org/abs/2502.10517).
   - The approach isn't quite **AlphaGeometry-style** but involves a small set of actions to apply angle chasing solvers.
- **Geometry and Verifiers Discussed**: Discussion involved methods related to **alpha-geometry style** techniques and verifiers.
   - Mentioned a method inherently involves a pretty small set of possible actions to apply angle chasing solvers.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1358956210988257401)** (6 messages): 

> `Quasar Alpha, Reasoning Gym Levels, Curricula Tasks` 


- **Quasar Alpha: Open Router Test Model**: A user shared the performance of **Quasar Alpha**, the open router test model, with an [attached image](https://cdn.discordapp.com/attachments/1316377974672588850/1358956210753114293/Figure_1.png?ex=67f7b458&is=67f662d8&hm=47ec1cb8ad8ce7db302a1d483baf3aeca52da523c45675b7e468a14ac8e5b740&).
   - Another user asked for the raw outputs to potentially add in a PR to [reasoning-gym-eval](https://github.com/open-thought/reasoning-gym-eval).
- **Reasoning Gym Task Levels Need Definition**: A user mentioned they are defining the levels for **15 tasks** in reasoning-gym that currently lack a defined level, planning to finish by the evening.
   - The user inquired about submitting a PR to make these definitions available on the main branch and it was approved.
- **Reasoning Gym Curricula Task PRs**: A user asked if adding curricula to tasks without them was appropriate for a PR to [reasoning-gym-eval](https://github.com/open-thought/reasoning-gym-eval).
   - It was encouraged to create a PR for this purpose.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1358939166250893482)** (3 messages): 

> `DeepSeek Communication Library, NVSHMEM and UVA, Intra-node communication` 


- **DeepSeek Leverages NVSHMEM**: A member inquired whether the **DeepSeek communication library** is built off **NVSHMEM library** from NVDA.
- **NVSHMEM's use of UVA Questioned**: A member questioned whether **NVSHMEM** uses **Unified Virtual Addressing (UVA)** for intra-node communication.
- **Peer-to-Peer Loads/Stores via NVLink**: The member added that using **UVA**, one can perform peer-to-peer loads/stores to data stored in a remote GPU (connected by something like **NVLink**).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1359236223498977391)** (11 messages🔥): 

> `Submitting .py files with inline CUDA, CUDA Kernels, Grayscale CUDA Example, torch::extension` 


- **Inline CUDA Submission Trouble**: A user reported trouble submitting **.py files** with inline **CUDA**, questioning the validity of the reference script.
   - Admins acknowledged the issue and requested a link to the failing job to assist with debugging.  Another user suggested the sample submission might be incorrect and that other inline CUDA implementations may work.
- **CUDA Inline Script Solution**: A user requested a sample script for inline **CUDA** submissions, and another user provided a code template using **C++** and **CUDA**.
   - The code template included **CUDA sources** (a `grayscale_kernel` function), **C++ sources** (including `<torch/extension.h>`), and a **Python module** loaded via `load_inline`.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1358888800029249841)** (17 messages🔥): 

> `vectoradd benchmarks, grayscale benchmarks, Modal runners` 


- **VectorAdd Benchmarks Galore**: Multiple benchmark submissions for **vectoradd** on **L4 GPUs** using **Modal runners** have succeeded with IDs ranging from **3500** to **3532**.
- **Grayscale Leaderboard Gains Traction**: Leaderboard submissions for **grayscale** on **L4, T4, A100, and H100 GPUs** using **Modal runners** were successful, including IDs **3503, 3536, 3539, and 3540**.
- **Modal Runners Deliver Results**: Submissions to the **vectoradd** leaderboard on **T4** and **A100 GPUs**, IDs **3537** and **3538** respectively, succeeded using **Modal runners**.


  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1358880301543329923)** (5 messages): 

> `Leaderboard discrepancy, CUDA submission failure` 


- **Leaderboard Time Units Cause Confusion**: A user noticed a discrepancy in time units between the web ([https://gpu-mode.github.io/discord-cluster-manager/](https://gpu-mode.github.io/discord-cluster-manager/)) and Discord leaderboards, with the former displaying **nanos** and the latter **millis**.
   - A new leaderboard website is being prepared, with time units converted for clarity.
- **CUDA Submission Stumbles**: A user reported that the sample CUDA submission from ([https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py)) failed to run as a test.
   - This was deemed unexpected, and the user was asked to provide the specific error message.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1358932410410205254)** (3 messages): 

> `A100 vs L40, FP8 support, 4bit weights, Open source w4a8 kernels, GPU Fryer tool` 


- **A100 Dominates L40 in Bandwidth and Tensor Ops**: The **A100** is reportedly nearly twice as fast as the **L40** in both **DRAM bandwidth** and **tensor operations**.
   - Despite the **L40** having **FP8 support** and a larger **L2 cache**, *vLLM doesn't include optimized kernels for Lovelace in its normal distribution*.
- **Limited Benefits of 8-bit Floating Point with 4-bit Weights**: With **4-bit weights**, the benefits of **8-bit floating point support** in **Hopper/Lovelace** are limited.
   - There are currently no open-source **w4a8 kernels** available.
- **GPU Fryer Tool for Problem Hunting**: Running the [GPU Fryer tool](https://github.com/huggingface/gpu-fryer) can help in identifying problems.
   - This tool, maintained by Hugging Face, is useful for stress-testing and debugging **GPU configurations**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1358883324382416936)** (52 messages🔥): 

> `FP4 Fine-tuning, Parasail Inference Provider, Llama.cpp Llama 4 Support, Mobile SQL Generation Models, Multi-Agent AI Deployment` 


- **FP4 Fine-Tuning Frenzy Fuels Faster Finishes**: Users are exploring fine-tuning quantized models using **FP4** with tools like [Unsloth](https://github.com/unslothai/unsloth), which allows loading lower precision models for training and quantization.
   - While fine-tuning a quantized model is possible via **LoRA**, directly fine-tuning the quantized model itself is not.
- **Parasail aims to Provide Premier Performance**: **Parasail**, a new inference provider, is looking to partner with Hugging Face after recently coming out of stealth, already serving **3B tokens a day** on Open Router and **5B+ a day** for private companies, as reported by [The Next Platform](https://www.nextplatform.com/2025/04/03/parasail-brokers-between-ai-compute-demand-and-supply/).
- **Llama.cpp Leaps to Llama 4**: The backend **Llama.cpp** has been updated to support Llama 4, according to the [GitHub releases](https://github.com/ggml-org/llama.cpp/releases).
- **Tiny Transformers Touted for Telephones**: For generating **SQL queries** from data descriptions on mobile devices, **Qwen 2.5 0.5B** and models from the [SmollM2 Intermediate Checkpoints collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-intermediate-checkpoints-67c079ca030f714c30ce49a1) and the [TinyLlama collection](https://huggingface.co/collections/TinyLlama/tinyllama-11b-v11-660bb405bf46efd55c2094fc) are recommended.
   - Converting models to **TensorRT** format via **ONNX** is suggested for leveraging older architectures.
- **Orchestration Options Open Opportunities**: **Oblix** ([https://oblix.ai/](https://oblix.ai/)), a new tool for orchestrating AI between edge and cloud, integrates with Ollama on the edge and supports OpenAI and ClaudeAI in the cloud, aiming to create low-latency, privacy-conscious workflows.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1359043209065402530)** (3 messages): 

> `Ollama local deployment, NLP in HuggingFace` 


- **Newbie Uses Ollama for Local Deployment**: A member is starting to learn using **Ollama** for local deployment with **Python** and **OpenAI**.
   - They are using **Ollama** local deployment to avoid paying for **OpenAI's API keys**.
- **Newbie learning NLP in HuggingFace**: A member mentioned they are learning about **NLP** in the **HuggingFace** page.
   - They are hoping to finish the course by the deadline.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1359180244493271211)** (1 messages): 

> `Daily Papers Podcast, Takara TLDR` 


- ****Takara TLDR** inspires a **Daily Papers Podcast**!**: A user took the **Takara TLDR** concept and created a [daily papers podcast](https://huggingface.co/spaces/eswardivi/Daily_Papers_Podcast).
   - The podcast seems to be hosted on the HuggingFace platform.
- **Daily AI paper summaries, now in Podcast form**: A user has remixed the **Takara TLDR** concept into a [daily papers podcast](https://huggingface.co/spaces/eswardivi/Daily_Papers_Podcast).
   - This could be a valuable resource for staying up-to-date with the latest AI research.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1358965715436044460)** (3 messages): 

> `AI Runner, GAPRS` 


- ****AI Runner** desktop GUI takes Flight!**: A member released **AI Runner**, a desktop GUI for running AI models locally using HuggingFace libraries as described in [this YouTube video](https://youtu.be/IPn3TcQr7e0).
   - The tool enables users to create and manage chatbots with custom voices, personalities, and moods, and the bots are agents built with llama-index using ReAct tools to generate images with **Stable Diffusion** and real-time voice conversations (espeak, speecht5, or openvoice).
- **GAPRS 3.0 sets Sail!**: A member launched the 3rd iteration of their master's thesis project, a web application called **GAPRS** (graph-based academic recommender system) at [lqhvwseh.manus.space](https://lqhvwseh.manus.space).
   - The goal of **GAPRS** is to help students know where to start when writing their theses, streamline the academic research process, and revolutionize monetization of academic papers; more details are available in the member's master's thesis.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1359061388873437254)** (3 messages): 

> `Monocular Depth Models, Segmentation Problem, Tools Recognition Task` 


- **Monocular Depth Models Explored**: A member inquired if another member had found a solution to a problem, noting they had already tried **monocular depth models**.
- **Segmentation Solution Proposed**: A member suggested a solution to a **segmentation problem** involving vertical poles, proposing to check the overlap of the x-coordinates of the bounding boxes of different segments with the same label.
   - The user said to *take the min(x_lefts)->max(x_rights)*; they also suggested trimming thick boxes by using (x_mid +/- 0.5*width_pole).
- **Tools Recognition Task**: A member asked for suggestions on the best model/algorithm for a **tool recognition task**, specifying that the model should identify tools by providing a reference picture.
   - The member asked if the model should be enhanced for better feature extraction.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1358967599786102866)** (4 messages): 

> `Dataset forms, Unit 1 Quiz failing to load, Agents Build Errors, Chat templating exercises` 


- **Dataset forms cause confusion**: A member pointed out that someone was *doing the same thing to both datasets, but they start in different forms*.
   - Another member requested more details, asking *what do you mean by forms?*
- **Unit 1 Quiz fails to load, redirects too many times**: A member reported the **Unit 1 quiz** fails to load and gets stuck in a redirect loop: *agents-course-unit-1-quiz.hf.space redirected you too many times*.
   - They mentioned they're new to coding and unsure how to resolve the issue, seeking support.
- **Agents Get Build Error**: A member reported experiencing a *Build error* when trying to fetch the error logs, getting stuck in a loop.
   - Initially, they couldn't get any response when chatting with the agent, and they were having issues even with *copying and pasting the name of the tool*.
- **Someone Needs Chat Template Exercise Buddy**: A member is seeking someone to discuss the **chat templating exercises** with them.
   - No other details were provided, they were just looking for a study buddy.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1358906149986107702)** (26 messages🔥): 

> `Code Agents Ch. 2 Notebook Issues, Gemini Models as Alternatives, Course FAQ Request, any-agent library release, RAG with smart glasses challenge` 


- **Code Agents Ch. 2 Notebook Requires Payment**: A member reported needing to pay to run the Chapter 2 notebook for Code Agents, receiving errors about invalid credentials and payment requirements for the recommended model.
   - They sought advice on logging in correctly or using alternative tokens to run the supposedly free course examples.
- **Gemini Models Recommended to bypass paywalls**: A member suggested using **Gemini models** as a free alternative in many countries, linking to [course notes](https://gist.github.com/skymaiden/8b472bbb01ea9bdfca43f64c32e583a6#using-other-llm-providers-outside-hugging-face) with instructions.
   - Other members highlighted resources like **Ollama** and other providers (**OpenAI**, **Grok**) offering generous free tokens, in order to bypass the Hugging Face paywalls.
- **FAQ Section Needed in Agent Course**: Multiple members requested an **FAQ section** within the Agent Course itself, as many users face the same initial issues and find the Discord navigation difficult.
   - The discussion clarified that while there isn't an official FAQ page, the Discord channel contains numerous frequently asked questions that can be searched.
- **`any-agent` Library Simplifies Agent Framework Evaluation**: The Mozilla AI team released `any-agent`, a library designed to simplify trying different agent frameworks.
   - The library supports frameworks like **smolagents**, **OpenAI**, **Langchain**, and **Llama Index**, with a [GitHub repository](https://github.com/mozilla-ai/any-agent) available for users to try and contribute.
- **Meta CRAG Multi-Modal Challenge Release**: The community shared an interesting challenge from **Meta**: the [CRAG Multi-Modal Challenge 2025](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025) related to RAG with smart glasses.
   - This is recommended as a knowledge exercise to solidify what was learned in the course.


  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1359333165776244917)** (13 messages🔥): 

> `Deepseek R1, Active AI Discord Chats` 


- **Deepseek R1 Chatroom Gossip Starts**: A member asked which **Deepseek** versions another member had been working with.
   - That other member quipped back that they aren't working at all, but assume this room is **Deepseek R1** related, and so the **AI** is working for *them*.
- **Chatter Seeks Active AI Communities**: A member inquired about active chats on **AI** in **Discord**, or even better, active voice chats.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1358883312361279630)** (75 messages🔥🔥): 

> `Semgrep MCP server, MCP HTTP Streaming, MCP and CORS errors, MCP Github server issues, MCP for Graph API application` 


- **Semgrep's MCP Server Makes Waves**: A member has been running [mcp.semgrep.ai/sse](https://mcp.semgrep.ai/sse) for over a month, hosted via **Docker** and **AWS EC2**.
- **CORS Error Squashed in Semgrep MCP Server**: A member reported a **CORS error** when connecting with the [Cloudflare Playground](https://playground.ai.cloudflare.com/), which was quickly fixed.
   - The reporter noted the tool was testing with **Cursor** and will need to fix CORS there as well.
- **MCP HTTP Request-Response Support Arrives**: A discussion emerged regarding the need for **HTTP request-response** support in MCP for enterprise customers, as highlighted in [this pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206#issuecomment-2766559523).
   - Members pointed out that many enterprise organizations are using MCP, and this feature is expected to further increase its adoption.
- **MCP Powers Up RAG with Graph DB**: A member inquired about using MCP in a **RAG** use case with a **Neo4j graph database**, focusing on vector search and custom **CQL search**.
   - Another member confirmed this is a good usecase, linking to [mcpomni-connect](https://pypi.org/project/mcpomni-connect/) as a viable MCP client.
- **Cloudflare Provides Remote MCP Server Tutorial**: For those seeking an easier tutorial to get started with remote MCP servers, a member recommends the [Cloudflare Agents guide](https://developers.cloudflare.com/agents/guides/remote-mcp-server/).


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1358879016286949799)** (15 messages🔥): 

> `Semgrep rewrites MCP, C# MCP SDK, ASGI style in process fastmcp sessions` 


- **Semgrep Rewrites MCP Server**: A member rewrote [Semgrep's MCP server](https://github.com/semgrep/mcp) and shared demo videos in [Cursor](https://www.loom.com/share/8535d72e4cfc4e1eb1e03ea223a702df) and [Claude](https://www.loom.com/share/f4440cbbb5a24149ac17cc7ddcd95cfa?sid=f190a5d6-176f-4ceb-86a2-35e98e701411).
   - The hosted server uses **SSE**, not HTTP streaming, because the [Python SDK](https://github.com/modelcontextprotocol/python-sdk/pull/416) doesn't support it yet.
- **MCP SDK Leverages Sqlite for LLM Memories**: A member played with the [C# MCP SDK](https://github.com/mbcrawfo/KnowledgeBaseServer) to leverage **sqlite** for **LLM memories**.
   - A new version is available that gives memories an **importance ranking** to help with search results, and is intended to scale to larger memory graphs.
- **ASGI Style Fastmcp Sessions Finalized**: A member bumped versions on [easymcp to 0.4.0](https://github.com/promptmesh/easymcp), with notable changes including **ASGI** style in process **fastmcp sessions**.
   - Other updates included a finalized **native docker transport**, refactored protocol implementation, a new mkdocs, and a full proper pytest setup.
- **Terminal Chat with MCP Servers**: A member created a [terminal chat with MCP servers](https://github.com/GeLi2001/mcp-terminal).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1358895968614879354)** (62 messages🔥🔥): 

> `Shopify AI Mandate, Anthropic API Credits, API Latency Benchmarking, Cybercriminals and AI, LLM Automated Exploitation` 


- **Shopify's AI Quest Gets Noticed**: Shopify's AI mandate is gaining traction, as highlighted in [this tweet](https://fxtwitter.com/tobi/status/1909251946235437514).
- **Anthropic's API Credits have Expiration Dates**: Anthropic API credits expire after one year, potentially for accounting simplification and to account for the rapidly evolving AI landscape.
   - As one member suggested, this policy helps manage projections in a quickly changing field.
- **NVIDIA's Reasoning Model has On/Off Switch**: NVIDIA has released a new model with the ability to turn reasoning on or off, detailed in [this blog post](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/) and available on [Hugging Face](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1).
- **Cybercrime's AI Shock May Be Delayed**: Despite basic AI applications like FraudGPT, mass adoption of AI by cybercriminals is surprisingly slow, with speculation that a "cybercrime AI shock" may occur when they adopt it more broadly.
   - One member noted that LLMs may have only recently become good enough for use in cybercrime.
- **Gemini Plays Pokemon and Streams the Madness**: The Gemini AI is now playing Pokémon, garnering attention as shown in [this tweet](https://fxtwitter.com/kiranvodrahalli/status/1909699142265557208).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1358957032652148937)** (14 messages🔥): 

> `Llama 4 flops on benchmarks, Bayesian Structural EM, Procedural model representation DNA, Meta should have clarified, Disrupt Science Hackathon Details` 


- **Llama 4 flops on Nongamed Benchmarks**: A member claimed that **Llama 4 flops hard** on nongamed nonoverfitted benchmarks.
   - The Daily Paper Discussion room will be discussing [this paper](https://arxiv.org/abs/2408.04220), and a recent talk by the main author ([YouTube link](https://www.youtube.com/watch?v=klW65MWJ1PY)) discusses this paper.
- **Meta should clarify Llama 4**: It was stated that *Meta should have made it clearer that “Llama-4-Maverick-03-26-Experimental” was a customized model to optimize for human preference.*
   - This discussion was based on [this fxtwitter link](https://fxtwitter.com/lmarena_ai/status/1909397817434816562?t=Gdzbf-abkahHSxqhEeqAkw&s=19).
- **Bayesian Inference Insights**: A member pointed out that Bayesian inference has been combining weights and architecture for about 100 years, and cited [Bayesian Structural EM](https://arxiv.org/pdf/1301.7373) as an advanced example.
   - It was argued that while combining weights and architecture is standard practice (e.g., in [DARTS](https://arxiv.org/pdf/1806.09055) or [ES-ENAS](https://arxiv.org/pdf/2101.07415)), *you do not gain any expressivity from updating both the architecture and the weights that you couldn't get from just weights*.
- **Procedural Model Representation: DNA of a Model**: A member introduced the concept of **procedural model representation**, where a small seed can generate a large model (architecture + weights).
   - They envisioned downloading a 10MB model that generates a 100TB model, or swapping seeds to generate different models, akin to *downloading DNA to generate a human*.
- **Disrupt Science Hackathon Details Posted**: Details for the **Disrupt Science Hackathon** have been posted.
   - The details can be found in [this discord link](https://discord.com/channels/714501525455634453/796137754508656641/1359164013304615143).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1358900565114884207)** (12 messages🔥): 

> `Fast.ai Diffusion Methods, F_A_E_S_I_k=2 Discussion, Open Source beautiful.ai Alternatives` 


- **Fast.ai Explores Diffusion Methods**: A member shared a link to [Fast.ai's diffusion methods course](https://course.fast.ai/Lessons/part2.html).
   - Another member inquired about the timing of the second part of the course.
- **Decoding F_A_E_S_I_k=2 Revelations**: A member joked about having `F_A_E_S_I_k=2`, leading to getting 40 hours of video, in relation to a paper discussion on [arxiv.org/abs/2408.04220](https://arxiv.org/abs/2408.04220).
   - They speculated it *might by construction have built this in, but probably not good at needles in a haystack, especially when the needles have dependencies on one another*.
- **Quest for Beautiful.ai Open Source Twins**: A member asked about open-source alternatives to [Beautiful.ai](https://www.beautiful.ai/).


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1359159357559804036)** (1 messages): 

> `Efficient Tool Calling Templates, Cogito 14b` 


- **Cogito 14b's Efficient Tool Template**: The **14b model** suddenly started using a more efficient tool calling template than was initially provided in the instructions.
   - It's recommended to check out the [Cogito model](https://ollama.com/library/cogito) for examples and inspiration.
- **New Efficient Tool Calling Template Implementation**: A user reported that a **14b model** unexpectedly adopted a more efficient tool calling template.
   - This suggests the model may have autonomously optimized its tool use, offering a potential area for further investigation.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1358885648978936050)** (9 messages🔥): 

> `Adapting Pre-training Text, Diffusion Modeling to Control LLMs, Llama 4 Release Issues, Iterative Improvement Strategy` 


- **Pre-Training Adaptation Talk**: A member shared an [Awesome talk](https://www.youtube.com/watch?v=klW65MWJ1PY) about adapting **pre-training text** to include database lookups for relevant facts, to train the **LLM** to look things up during generation.
- **Diffusion Models Guide LLMs**: A member mentioned using **diffusion modeling** to control **LLMs** and pointed to [this paper](https://arxiv.org/pdf/2408.04220) as a relevant resource.
- **Llama 4's Flop Explained**: The poor release of **Llama 4** was attributed to bad implementations.
- **DeepCogito Iterative Improvement Strategy Preview**: A member shared a link from **Hacker News** about an iterative improvement strategy using test time compute for fine-tuning, from [DeepCogito](https://www.deepcogito.com/research/cogito-v1-preview).


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1358879938358411466)** (28 messages🔥): 

> `IBM Granite 8B, RAG references, docling OCR, semantic chunking server, ComfyUI image generation` 


- **Granite 8B Shines for RAG tasks**: A member reported that [IBM Granite 8B](https://www.ibm.com/blogs/research/granite-foundation-models/) works well with **RAG**, especially concerning providing references by **LLM**.
   - Another member concurred, having also found **Granite** to be effective.
- **Docling for Non-Text PDF OCR**: A member recommended **docling** for excellent **image OCR**, especially for non-text PDFs like scans.
   - They highlighted its continuous operation for embeddings and integration into a database with indexed documents, enabling **RAG** through intersections.
- **Semantic Chunking for Contextual Text**: A member shared a semantic chunking server, demonstrating its use with [clipboard examples](https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html).
   - They noted its compatibility with audio and image processing, suggesting **ComfyUI** for combining all modalities.
- **Llama 4 Gets Trashed for Being Terrible**: A member trashed the **Llama 4th gen model** for being *terrible compared to smaller models*.
   - Others agreed, noting [Reddit comments](https://www.reddit.com/r/LocalLLaMA/) speculated that it may have overfit on smaller "high quality" datasets, despite some benchmarks showing promise.
- **GPT4All: Keep it Local to Be Safe**: A member advised using **GPT4All** primarily for local operations to ensure privacy and avoid sending private information to remote APIs.
   - They detailed how to run embedding models locally and index files by chunking and embedding, referencing a [shell script example](https://gnu.support/files/tmp/clipboard-2025-04-09-01-48-48.html).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1358887711406297212)** (4 messages): 

> `MLX vs MAX, Apple Silicon GPU limitations, MAX capabilities` 


- **MLX vs MAX: A Comparative Dive**: A member compared **MLX** (an array programming framework akin to JAX) and **MAX**, noting that while **MLX** is tailored for Apple Silicon GPUs, **MAX** currently cannot target them, which poses challenges for direct comparison.
   - The member highlighted that **MAX** for AMD GPUs will eventually mirror **MLX's** shared memory benefits on MI300A and AMD's consumer CPUs, suggesting a future convergence in capabilities.
- **Apple Silicon's Deployment Drawbacks**: The member cautioned against relying solely on **MLX** for extensive projects, citing the difficulty of deploying Apple Silicon in server environments, necessitating potential rewrites to frameworks like **MAX**, **JAX**, or **PyTorch** for deployment.
   - They emphasized that while **MLX** might offer convenience for initial experimentation, the practical limitations of Apple's ecosystem in server settings should be a key consideration.
- **MAX's Manual Vectorization and Multi-Device Support**: The member detailed that **MAX**, despite its lower-level API, offers capabilities comparable to NumPy and leverages Mojo for both automatic and manual vectorization, making it programmer-friendly.
   - They admitted **MAX's** limitations in autodiff but highlighted its multi-device support, exemplified by the Llama pipeline, and its avoidance of tensor shape issues, positioning it as a robust alternative despite certain challenges.
- **Discord Self-Promotion Rules Violated**: A member pointed out that a specific post seemed inappropriate for the Discord channel, suggesting it might be a violation of self-promotion rules.
   - A moderator agreed, confirming that the post indeed violated the community's self-promotion guidelines.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1359089138992808017)** (16 messages🔥): 

> `Mojo vs Rust, __moveinit__ and __copyinit__ in Mojo, Returning values in Mojo, Span lifetime in Mojo` 


- **Mojo's Borrowing vs. Rust's: A Fresh Look**: A newcomer to Mojo shared a [blog post comparing Mojo and Rust](https://www.modular.com/blog/mojo-vs-rust), noting that Mojo's *"borrow by default"* felt more intuitive.
   - The member then wondered about how Mojo handles returning values from functions.
- **Moveinit vs Copyinit: Deep Dive into Mojo Object Returns**: A member clarified that when returning objects in Mojo, the presence of `__moveinit__` dictates whether the object is moved, otherwise `__copyinit__` is used, and provided [an example on Github](https://github.com/sstadick/mojo-demo/tree/main/examples).
   - The member also pointed to the [official Mojo documentation](https://docs.modular.com/) for a complete picture.
- **Unlock Span Lifetimes in Mojo with rebinding!**: A member inquired how to specify in Mojo that *"the lifetime of the return value is at least the lifetime of self"*, specifically for a `Span`.
   - Another member suggested using `rebind[Span[UInt8, __origin_of(self)]](Span(self.seq))` or making the trait generic over origin, but noted that trait parameters are not yet supported.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1359015949952618607)** (5 messages): 

> `Tensor Naming, GPU Programming, Compiler Development, Tinygrad Contribution Resources, PMPP 4th ed` 


- **Elegant Tensor Naming Tricks Sought**: A member inquired about a more elegant way to name a tensor for easier tracking when printing model parameters, noting they currently add a *name* attribute in the Tensor class manually.
- **GPU Programming and Compiler Dev Resources Requested**: A member expressed interest in getting into **GPU programming** and **compiler development** for projects like tinygrad and requested learning resources or blog posts.
   - They are planning to read [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/) and asked for book or blog post recommendations on compiler development for GPUs.
- **geohotarchive YouTube Channel recommended**: A member suggested the [geohotarchive YouTube channel](https://www.youtube.com/@geohotarchive/videos) as a resource for learning about tinygrad.
- **"PMPP" 4th Edition Recommended for GPU Programming**: A member recommended **PMPP (4th ed)** for GPU programming, suggesting to share any excellent compiler resources if found.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1358887432199602199)** (12 messages🔥): 

> `METAL sync issue, AMD performance with BEAM=2, ContextVar type, LLaMA sharding issue, Device info loss after sampling` 


- **METAL Sync Glitch Causes Sharding Shenanigans**: A member found unexpected behavior in sharding while reproducing a minimal example of a **METAL sync issue** from the bounty.
   - The user suspected that the **COPY** from **METAL:1** to **CPU** was executing before the **XFER** from **METAL** to **METAL:1** ended, causing the CPU to read zeros instead of the correct shard.
- **AMD BEAM=2 Turbocharges Tinygrad**: One user reported impressive speed improvements using **AMD** with **BEAM=2**, achieving **64 it/s**, outperforming their previous best with Torch at **55+ it/s**.
   - Members noted that *BEAM=2 often beats torch*.
- **LLaMA Sharding Snafu: Device Info Lost in Translation**: A user encountered an **AssertionError** while running **llama.py** with `--shard 4`, indicating that the device info was lost after sampling.
   - A potential fix was proposed to move the tensor, as seen on [GitHub](https://github.com/tinygrad/tinygrad/pull/9761/files), but it's not directly related to **METAL** or **sync** issues.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1359195100206338160)** (2 messages): 

> `RAG workflow tutorial, Auth0 Auth for GenAI with LlamaIndex` 


- **RAG workflows using Llama 4**: A quickstart tutorial demonstrates building a **RAG workflow** from scratch using **Llama 4**, showcasing how to set up core steps around ingestion, retrieval, and generation using LlamaIndex workflows, as shown in [this tweet](https://twitter.com/llama_index/status/1909635186079453494).
- **Auth0 Auth for GenAI ships with LlamaIndex support**: **Auth0's Auth for GenAI** now ships with native LlamaIndex support, making it easier to build auth into agent workflows, as announced in [this tweet](https://twitter.com/llama_index/status/1909697035365961954).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1358991517569519756)** (13 messages🔥): 

> `Gemini 2.5 Pro, Google's latest unified SDK, StructuredPlannerAgent Docs, Agent Planning Tool` 


- **Gemini 2.5 Pro Not Available**: A member inquired if **Gemini 2.5 Pro** was available, but discovered a deprecation message suggesting to use **Google's latest unified SDK** instead of Gemini 2.5 pro, as noted in the [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/examples/llm/google_genai/).
- **Google SDK model names aren't validated**: A member noted that the **Google SDK** doesn't validate model names, but assumes the provided name is valid, and also suggests manually setting the `context_window` value since **Gemini 2.5's** context window is quite large.
- **`StructuredPlannerAgent` Docs Removed**: The documentation for `StructuredPlannerAgent` was removed because it is no longer maintained due to a cleanup of the agent docs, due to duplicate implementations.
   - A backlink to the old documentation was provided: [StructuredPlannerAgent](https://docs.llamaindex.ai/en/v0.12.15/examples/agent/structured_planner/).
- **Agent Planning Tool recommended**: Instead of `StructuredPlannerAgent`, it was suggested to use an agent with a **planning tool** that does some **Chain of Thought (CoT)** reasoning, or using the **LLM** itself to create a plan before using agent(s).


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1358932250565283982)** (8 messages🔥): 

> `Events Recording Availability, Structured Output Examples, Pydantic Schema Integration, API Requests without Cohere Package, Model Recommendation for Company List Generation` 


- **Events Recordings: Are They Available?**: A member inquired whether recordings of events are available for those unable to attend in real-time, as some events sounded interesting.
   - No response was given.
- **Members Seek Structured Output Examples**: A new member asked for examples on how to get structured output (e.g., a list of books) using Cohere, expressing their lack of experience in the field.
   - The member was directed to the [Cohere documentation](https://docs.cohere.com) as a starting point.
- **Pydantic Schema with Cohere**: A member sought ways to use **Pydantic schemas** directly in `response_format` with Cohere and on how to send requests without the Cohere Python package, aiming to avoid introducing a dependency.
   - They were provided with a [link to the Cohere Chat API reference](https://docs.cohere.com/reference/chat) and shown how to use **cURL** for requests to `https://api.cohere.com/v2/chat`.
- **OpenAI SDK example with Response Format**: A member found the **cURL** example useful and noted its presence in the **OpenAI SDK** example with the `response_format` parameter.
   - The member then asked for a recommendation on the most suitable model for generating a list of companies on a specific topic.


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1358890160380575934)** (1 messages): 

> `Vector Databases, Model Compatibility, Explicit Recommendations` 


- **Vector DB Recommendations Historically Avoided**: Historically, explicit recommendations for **vector DBs** have been avoided, because our models work well with all of them.
   - This is because the models are designed to function effectively with *all* **vector DBs** without specific optimizations for any particular one.
- **Model Compatibility Across Vector DBs**: The models are designed for broad compatibility, ensuring they perform well with various **vector database** solutions.
   - This approach avoids favoring specific **vector DBs** and maintains a neutral stance towards the ecosystem.


  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

competent: Currently not working!
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1359316466289348729)** (2 messages): 

> `Introduction to Aditya, Machine vision and control, Innovation accelerator, Openchain.earth project, Tools used by Aditya` 


- **Aditya Joins Cohere's Community**: Aditya introduced themself as having a background in **machine vision and control** for manufacturing equipment (Semi/Electronics).
   - Currently, they are taking a sabbatical from an **innovation accelerator/matchmaking/assessment role** to explore web/AI, with a project on [openchain.earth](https://openchain.earth).
- **Aditya's Tech Stack Revealed**: Aditya uses **VS Code**, **Github Co-Pilot**, **Flutter**, **MongoDB**, **JS**, and **Python** (Evaluating) in their projects.
   - They are here to find out more on **Cohere's AI** and how it can be used in their project.


  

---


### **Cohere ▷ #[【🟢】status-updates](https://discord.com/channels/954421988141711382/1346652044181897307/)** (1 messages): 

competent: Should work!
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1359340577581432873)** (1 messages): 

> `Contributor Tag Request, Discord Roles` 


- **Contributor Tag Incoming**: A member requested a Contributor tag on Discord, sharing their [GitHub username](https://github.com/nathan-az).
   - They also made a lighthearted mention of their Discord profile picture featuring the character Gus from *Psych*.
- **Requesting Discord Roles**: A user is seeking a role elevation within the Discord server, specifically the Contributor tag.
   - They linked their GitHub profile for verification and joked about their profile picture.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1358908107149021255)** (6 messages): 

> `DeepSpeed Integration, FSDP vs DeepSpeed, FSDP Sharding, zero1-3 training` 


- **DeepSpeed Integration Debated for TorchTune**: A member inquired about integrating [DeepSpeed](https://www.deepspeed.ai/) as a backend into TorchTune and created [an issue](https://github.com/pytorch/torchtune/issues/2569) to discuss the possibility.
   - A maintainer asked for more context, noting that **FSDP supports all the sharding options from DeepSpeed**; potential reasons for DeepSpeed integration include *fallback in lieu of FSDP bugs, diverging hardware/accelerator support, and speed*.
- **FSDP Favored Over DeepSpeed in TorchTune**: TorchTune leans towards **FSDP** due to its better composition with other PyTorch distributed features, with the belief that *supporting both versions well is not feasible*.
   - Users who migrated to TorchTune to avoid the complexities of composing DeepSpeed, PyTorch, and Megatron prefer sticking to native PyTorch, so there is no need to over-index on integrating and supporting other frameworks.
- **Community Recipe Idea: DeepSpeed with TorchTune**: A maintainer suggested creating a community recipe that imports TorchTune and hosts a DeepSpeed recipe, offering to feature it if a repo is made.
   - This allows users interested in **DeepSpeed** to leverage it with TorchTune while keeping the core framework focused on native PyTorch.
- **Tweaking FSDPModule for zero1-2 Training**: Since TorchTune defaults to the equivalent of **zero3**, documentation or more recipes on how to tweak recipes using the **FSDPModule** methods for **zero1-2** training are appreciated.
   - It's believed that **zero 1-3** are all possible with very minor tweaks to the collectives.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1359213591215210626)** (1 messages): 

> `MIPRO, Automated Prompt Engineering, Task Complexity Scaling` 


- **MIPRO Algorithm Tested on Scaling Complex Tasks**: An [article](https://tensorzero.com/blog/from-ner-to-agents-does-automated-prompt-engineering-scale-to-complex-tasks) tested the **MIPRO automated prompt engineering algorithm** across tasks of varied complexity, from named entity recognition to text-based game navigation.
   - The study leveraged tasks like **CoNLL++, HoVer, BabyAI**, and **τ-bench** (customer support with agentic tool use).
- **Model Size Matters for MIPRO Optimization**: The study found that **larger models benefit more from MIPRO optimization** in complex settings, potentially because they handle longer multi-turn demonstrations more effectively.
   - The quality of feedback significantly impacts the MIPRO optimization process, with meaningful improvements seen even from **noisy AI-generated feedback**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1358886256062500924)** (1 messages): 

> `Kaiyu Yang, AI4Math, Theorem Proving, Autoformalization` 


- ****Kaiyu Yang** Lectures on Formal Math Reasoning**: Guest speaker **Kaiyu Yang** presented on *"Language models for autoformalization and theorem proving"* on a livestream today; [link here](https://www.youtube.com/live/cLhWEyMQ4mQ).
   - His lecture covered using LLMs for formal mathematical reasoning, including **theorem proving** and **autoformalization**.
- **AI4Math Crucial for AI-Driven Systems**: **AI for Mathematics (AI4Math)** is crucial for AI-driven system design and verification, and techniques mirror NLP, especially training LLMs on curated math datasets.
   - A complementary approach involves formal mathematical reasoning grounded in systems like **Lean**, which verify reasoning correctness and provide feedback.
- **Meta's Dr. Yang Enhances AI in Math**: **Dr. Kaiyu Yang**, a Research Scientist at Meta FAIR, focuses on enhancing AI's mathematical reasoning by integrating formal systems like **Lean**.
   - His work explores using LLMs for tasks like theorem proving (generating formal proofs) and autoformalization (translating informal to formal).


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1358918518338355270)** (1 messages): 

> `Manifold Research Group, Multimodal AI, self-assembling space robotics, robotic metacognition, Community Research Call #4` 


- **Manifold Research Group hosts Research Call #4**: Manifold Research Group is hosting [Community Research Call #4](https://lu.ma/wlne416w) this Saturday (4/12 @ 9 AM PST).
   - The call will cover their latest work in **Multimodal AI**, **self-assembling space robotics**, and **robotic metacognition**.
- **Space Robotics Research Taking Off**: A PhD student from Manifold Research Group, specializing in robotic swarms in space, extended an invitation to a research call.
   - The call aims to foster collaboration and explore frontier science in space robotics.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1358969916283162864)** (1 messages): 

> `Codeium rename, Windsurf Reddit, Windsurf Plugins` 


- **Codeium Rebrands as Windsurf**: Codeium has officially rebranded to **Windsurf**, following the launch and *incredible adoption* of the **Windsurf Editor** in November 2024.
   - The new name better reflects their vision of *combining human and machine to create effortlessly powerful experiences*, according to their [rebrand announcement](https://windsurf.com/blog/windsurf-rebrand-announcement).
- **Windsurf Launches New SubReddit**: The company has launched a new [SubReddit](https://www.reddit.com/r/windsurf) for the community.
   - The announcement was made alongside changes to the Discord server, including refreshed pages and renaming of channels.
- **Codeium Extensions are now Windsurf Plugins**: With the rebrand, **Codeium Extensions** are now officially **Windsurf Plugins**.
   - The company promised to continue improving the **Windsurf Editor**, wave by wave, with the same commitment to innovation.


  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
