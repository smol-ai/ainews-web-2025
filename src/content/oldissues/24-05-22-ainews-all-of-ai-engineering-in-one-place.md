---
id: cb95bcde-3d85-4898-9cc4-6b7272ca5b88
title: ALL of AI Engineering in One Place
date: '2024-05-23T01:22:53.232395Z'
original_slug: ainews-the-top-ai-engineer
description: >-
  Anthropic has made progress in interpretability research on Claude 3 Sonnet by
  extracting millions of multilingual and multimodal features that correspond to
  abstract concepts, enabling feature steering to reliably modify model
  behavior. They also identified safety-relevant features linked to unsafe or
  biased outputs, though further research is needed to confirm practical
  applications.
tags:
  - anthropic
  - claude-3
  - interpretability
  - feature-steering
  - safety
  - multimodal
  - multilingual
---


<!-- buttondown-editor-mode: plaintext -->**Deep IRL networks are all you need! Jun 25-27 in SF.**

> AI News for 5/21/2024-5/22/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**380** channels, and **7699** messages) for you. 
Estimated reading time saved (at 200wpm): **805 minutes**.

Lots of nontechnical news - [the California Senate passed SB 1047](https://www.reddit.com/r/LocalLLaMA/comments/1cxqtrv/comment/l54rdfh/), more explosive news on [OpenAI employee contracts from Vox](https://x.com/KelseyTuoc/status/1793402040439476554) and [safetyist resignations](https://x.com/GretchenMarina/status/1793403475260551517), and though [Mistral v0.3 was released](https://x.com/reach_vb/status/1793337655595340267) there's no evals or blogpost to discuss yet.

Given its a technically quiet day, we take the opportunity to share our announcements of [the initial wave of **AI Engineer World's Fair** speakers](https://x.com/aiDotEngineer/status/1791506805065216017)! 

> TLDR we're giving a onetime discount to AI News readers: [**CLICK HERE**](https://ti.to/software-3/ai-engineer-worlds-fair/discount/AINEWS) and enter `AINEWS` before EOD Friday :) 

 ![image.png](https://assets.buttondown.email/images/a40e40f8-c9f2-4721-a6f8-3bbcfb777204.png?w=960&fit=max) 

## The AI Engineer World's Fair (Jun 25-27 in SF)

The [first Summit was well reviewed](https://eugeneyan.com/writing/aieng-reflections/) and now the **new format is 4x bigger**, with booths and talks and workshops from:

- **Top model labs** (OpenAI, DeepMind, Anthropic, Mistral, Cohere, HuggingFace, Adept, Midjourney, Character.ai etc)
- **All 3 Big Clouds** (Microsoft Azure, Amazon AWS, Google Vertex)
- **BigCos putting AI in production** (Nvidia, Salesforce, Mastercard, Palo Alto Networks, AXA, Novartis, Discord, Twilio, Tinder, Khan Academy, Sourcegraph, MongoDB, Neo4j, Hasura etc)
- **Disruptive startups setting the agenda** (Modular aka Chris Lattner, Cognition aka Devin, Anysphere aka Cursor, Perplexity, Groq, Mozilla, Nous Research, Galileo, Unsloth etc)
- **The top tools in the AI Engineer landscape** (LangChain, LlamaIndex, Instructor, Weights & Biases, Lambda Labs, Neptune,  Datastax, Crusoe, Covalent, Qdrant, Baseten, E2B, Octo AI, Gradient AI, LanceDB, Log10, Deepgram, Outlines, Unsloth, Crew AI, Factory AI and many many more)

across **9 tracks** of talks: **RAG, Multimodality, Evals/Ops (new!), Open Models (new!), CodeGen, GPUs (new!), Agents, AI in the Fortune 500 (new!)**, and for the first time a dedicated **AI Leadership** track for VPs of AI, and **50+ workshops and expo sessions** covering every AI engineering topic under the sun. Of course, the most important track is the unlisted one: **the hallway track**, which we are giving lots of love to but can't describe before it happens.

To celebrate the launch of the World's Fair, **we're giving a onetime discount to AI News readers**: [**CLICK HERE**](https://ti.to/software-3/ai-engineer-worlds-fair/discount/AINEWS) and enter `AINEWS` before EOD Friday :) 

If the curation here/on Latent Space has the most cosine similarity with your interests, **this conference was made for you**. See you in SF **June 25-27**!

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and Discord Summaries have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}



---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Anthropic's Interpretability Research on Claude 3 Sonnet**

- **Extracting Interpretable Features**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935511582986466) used dictionary learning to extract millions of interpretable "features" from Claude 3 Sonnet's activations, corresponding to abstract concepts the model has learned. **Many features are multilingual and multimodal**.
- **Feature Steering to Modify Behavior**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935517991895061) found that intervening on these features during a forward pass ("feature steering") could **reliably modify the model's behavior and outputs** in interpretable ways related to the meaning of the feature.
- **Safety-Relevant Features**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935524220481777) identified many "safety-relevant" features corresponding to concerning capabilities or behaviors, like **unsafe code, bias, dishonesty, power-seeking, and dangerous/criminal content**. Activating these features could induce the model to exhibit those behaviors.
- **Preliminary Work, More Research Needed**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935531925430407) notes this work is preliminary, and while the features seem plausibly relevant to safety applications, **much more work is needed to establish practical utility**.
- **Hiring for Interpretability Team**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935540536279368) is hiring managers, research scientists, and research engineers for their interpretability team to further this work.

**Microsoft's Phi-3 Models**

- **Phi-3 Small and Medium Released**: [@_philschmid](https://twitter.com/_philschmid/status/1792934321407369532) announced Microsoft has released Phi-3 small (7B) and medium (14B) models under the MIT license, **with instruct versions up to 128k context**.
- **Outperforming Mistral, Llama, GPT-3.5**: [@_philschmid](https://twitter.com/_philschmid/status/1792934321407369532) claims Phi-3 small outperforms Mistral 7B and Llama 3 8B on benchmarks, while **Phi-3 medium outperforms GPT-3.5 and Cohere Command R+**.
- **Training Details**: [@_philschmid](https://twitter.com/_philschmid/status/1792934321407369532) notes the models were trained on 4.8 trillion tokens including synthetic and filtered public datasets with multilingual support, **fine-tuned with SFT and DPO**. No base models were released.
- **Phi-3-Vision Model**: Microsoft also released Phi-3-vision with 4.2B parameters, which [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793031695702036869) notes **outperforms larger models like Claude-3 Haiku and Gemini 1.0 Pro V on visual reasoning tasks**.
- **Benchmarks and Fine-Tuning**: Many are eager to benchmark the Phi-3 models and potentially fine-tune them for applications, though [@abacaj](https://twitter.com/abacaj/status/1792991309751284123) notes **fine-tuning over a chat model can sometimes result in worse performance than the base model**.

**Perplexity AI Partners with TakoViz for Knowledge Search**

- **Advanced Knowledge Search with TakoViz**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1792948540542517458) announced a partnership with TakoViz to bring **advanced knowledge search and visualization** to Perplexity users, allowing them to search, juxtapose and share authoritative knowledge cards.
- **Authoritative Data Providers**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1792948544669667554) notes TakoViz sources knowledge from **authoritative data providers with a growing index spanning financial, economic and geopolitical data**.
- **Interactive Knowledge Cards**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1792963691669008390) explains users can now prompt Perplexity to **compare data like stock prices or lending over specific time periods**, returning interactive knowledge cards.
- **Expanding Beyond Summaries**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1792964343874892202) says this allows Perplexity to go beyond just summaries and **enable granular data queries across timelines**, which is now possible from a single search bar.
- **Passion for the Partnership**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1792966258822095347) expresses his love for working with the TakoViz team and participating in their pre-seed round, noting their **customer obsession and the value this integration will bring to Perplexity users**.

**Miscellaneous**

- **Karina Nguyen Joins OpenAI**: [@karinanguyen_](https://twitter.com/karinanguyen_/status/1792996299069071760) announced she has left Anthropic after 2 years to join OpenAI as a researcher, sharing **lessons learned about AI progress, culture, and personal growth**.
- **Suno Raises $125M for AI Music**: [@suno_ai_](https://twitter.com/suno_ai_/status/1792922276683297162) announced raising $125M to **build AI that amplifies human creativity in music production**, and is hiring music makers, music lovers and technologists.
- **Yann LeCun on LLMs vs Next-Gen AI**: [@ylecun](https://twitter.com/ylecun/status/1793326904692428907) advises students interested in building next-gen AI systems to **not work on LLMs**, implying he is working on alternative approaches himself.
- **Mistral AI Releases New Base and Instruct Models**: [@_philschmid](https://twitter.com/_philschmid/status/1793337888110694683) shared that Mistral AI released new 7B base and instruct models with **extended 32k vocab, function calling support, and Apache 2.0 license**.
- **Cerebras and Neural Magic Enable Sparse LLMs**: [@slashML](https://twitter.com/slashML/status/1793030889233936695) shared a paper from Cerebras and Neural Magic on **enabling sparse, foundational LLMs for faster and more efficient pretraining and inference**.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Model Releases and Benchmarks**

- **Microsoft releases Phi-3 models under MIT license**: In /r/LocalLLaMA, Microsoft has released their Phi-3 small (7B) and medium (14B) models under the MIT license on Huggingface, including [**128k and 4-8k context versions along with a vision model**](https://www.reddit.com/r/LocalLLaMA/comments/1cxa6w5/phi3_small_medium_are_now_available_under_the_mit/). 
- **Phi-3 models integrated into llama.cpp and Ollama**: The Phi-3 models have been [added to the llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1cxi14h/phi3_128k_model_support_merged_into_llamacpp/) and [Ollama](https://www.reddit.com/r/LocalLLaMA/comments/1cxofw0/has_anyone_gotten_phi3vision128kinstruct/) frameworks, with **benchmarks showing they outperform other models in the 7-14B parameter range**.
- **Meta may not open source 400B model**: According to a [leaker on /r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1cxnrov/disappointing_if_true_meta_plans_to_not_open_the/), Meta may go back on previous indications and **not open source their 400B model, which would disappoint many**.
- **Benchmark compares 17 LLMs on NL to SQL**: A [comprehensive benchmark posted on /r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1cx1wdy/findings_from_latest_comprehensive_benchmark/) compared **17 LLMs including GPT-4 on natural language to SQL tasks, with GPT-4 leading in accuracy and cost but significant performance variation by hosting platform**.

**AI Hardware and Compute**

- **Microsoft introduces NPUs for mainstream PCs**: Microsoft announced that [neural processing units (NPUs) will become mainstream in PCs for AI workloads](https://www.reddit.com/r/singularity/comments/1cxcdie/microsoft_event_live/), with new Surface laptops having an **exclusive 64GB RAM option to support large models**.
- **Overview of M.2 and PCIe NPU accelerators**: An [overview on /r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1cx5jvc/overview_of_m2_pcie_npus/) looked at the current landscape of **M.2 and PCIe NPU accelerator cards, noting most are still limited in memory bandwidth compared to GPUs but the space is evolving rapidly**.

**AI Concerns and Regulation**

- **Europe passes AI Act regulating AI development and use**: The [EU has passed the comprehensive AI Act](https://www.reddit.com/r/singularity/comments/1cxfqau/corpos_should_drop_the_whole_ai_safety_facade/) which will **regulate the development and use of AI systems starting in 2026 and likely influence regulation globally**.
- **California Senate passes AI safety and innovation bill**: The California Senate has passed [SB1047 to promote AI safety and innovation](https://www.reddit.com/r/LocalLLaMA/comments/1cxnrov/disappointing_if_true_meta_plans_to_not_open_the/), with **mixed reactions and some concerns it will limit AI progress in the state**.
- **TED head calls Meta "reckless" for open sourcing AI**: [Chris Anderson, head of TED, called Meta "reckless"](https://www.reddit.com/r/singularity/comments/1cx4onf/i_am_tired_of_scarjo_conflict/) for open sourcing AI models, **a concerning stance to AI progress advocates from an influential figure**.

**AI Assistants and Agents**

- **Microsoft introduces Copilot AI agent capabilities**: Microsoft announced [new agent capabilities for Copilot](https://www.reddit.com/r/OpenAI/comments/1cxhjov/recall_is_actually_a_massive_game_changer/) that can **act as virtual employees, with early previews showing ability to automate complex workflows**.
- **Demo showcases real-time multimodal AI game agents**: A [demo posted on /r/singularity](https://www.reddit.com/r/singularity/comments/1cxcdie/microsoft_event_live/) showcased **real-time multimodal AI agents assisting in video games by perceiving game state visually and providing strategic guidance**.
- **Questions raised about Amazon's lack of AI assistant progress**: /r/singularity [discussed Amazon's apparent lack of progress](https://www.reddit.com/r/singularity/comments/1cxbib0/how_is_amazon_missing_the_ai_boat_so_badly_here/) in AI assistants compared to other tech giants, **given their broad consumer reach with Alexa**.

**Memes and Humor**

- **Memes highlight rapid AI progress**: Memes and jokes circulated about the [rapid pace of AI progress](https://www.reddit.com/r/LocalLLaMA/comments/1cxa6w5/phi3_small_medium_are_now_available_under_the_mit/), **companies making dramatic claims, and concerns about advanced AI systems**.

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **LLM Benchmarking and Performance Optimization**:
   - **Microsoft's [Phi-3 Models](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)** offer high context lengths and robust performance, stirring discussions on benchmarks and memory usage but uncovering compatibility issues in tools like **llama.cpp**.
   - Various techniques like **torch.compile** and specific GPU setups were debated for optimizing computation efficiency, shared via insights like those in [tensor reshaping examples](https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e46c6e1594c2dcc/tinygrad/tensor.py#L83).

2. **Open-Source AI Tools and Frameworks**:
   - The **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** framework emerged as a go-to for fine-tuning models like Llama and Mistral, with Docker setups facilitating ease of use ([quickstart guide](https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl)).
   - **[LlamaIndex](https://docs.llamaindex.ai/en/stable/api_reference/packs/code_hierarchy/?h=code)** introduced techniques for document parsing and batch inference, integrating GPT-4o's capabilities to enhance complex document manipulation and query accuracy.

3. **AI Legislation and Community Responses**:
   - California's SB 1047 [bill](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047) prompted heated debates on the impact of new regulations on open-source models, with concerns over stifling innovation and favoritism towards major incumbents.
   - Discussions on ethical and legal questions arose around AI voice replication, highlighted by OpenAI's controversial mimicking of Scarlett Johansson's voice, leading to its subsequent removal after public backlash.

4. **Novel AI Model Releases and Analysis**:
   - Community excitement surrounded new releases such as **Mistral-7B v0.3** with extended vocabularies and function calling ([details](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)), while **Moondream2** updates improved resolution and accuracy in visual question-answering.
   - Anthropic's work on **interpretable machine learning** and the release of **Phi-3 Vision** spurred deep dives into scaling monosemanticity ([research](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)) and practical AI applications.

5. **Practical AI Implementations and Challenges**:
   - Members shared practical AI implementations, from **PDF extraction with Surya OCR** transforming documents into markdown ([GitHub repo](https://github.com/satish860/PDF-Extraction-API)), to building **secure code execution environments on Azure** ([dynamic sessions](https://t.co/lTrUPoTMcF)).
   - The **LangChain** community highlighted issues with deployment and endpoint consistency, with detailed troubleshooting on the [GitHub repo](https://github.com/langchain-ai/langserve/issues/301) helping streamline deployment processes and enhance chatbot functionalities.

---

{% if medium == 'web' %}



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Phi-3 Comes into Play, Skepticism and Excitement Ensue**: The introduction of **Phi-3 models** by Microsoft, such as [Phi-3-Medium-128K-Instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct), sparked discussions, with excitement tinged by skepticism due to potential benchmarking issues, highlighted by a user's single-word remark: "*literally*."

**New Legal Frontiers in AI**: California's SB 1047 sparked discussions concerning AI laws and open-source model implications, accentuated by Meta's decision to not open the weights for its 400B model, provoking a community debate on the wide-reaching effects of such restricted access.

**Unsloth Woes with Model Saving and Flash Attention**: Trouble reported with Unsloth's `model.save_pretrained_gguf()` function and Flash Attention compatibility, with suggestions from the community advising an Unsloth reinstall or [removing Flash Attention](https://github.com/unslothai/unsloth) and specific workarounds for T4 GPU issues on PyTorch version 2.3.

**Guided Decoding and YAML Finesse**: A spirited discussion on using *guided decoding* for generating structured YAML outputs revealed potential vLLM support with advanced syntaxes, emphasizing the integration of grammars into the prompting process.

**Cutting-Edge Model Discussions Mix with Sci-Fi**: Users shared advancements and tested methods like [MoRA](https://arxiv.org/abs/2405.12130), alongside spirited talks about the Dune series' philosophical undertones and defenses of novel reading over movie watching, underscoring a preference for depth in sci-fi storytelling.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **PDF Extraction Wins with Surya OCR**: Marker PDF effectively converts PDFs into markdown, surpassing other models with Surya OCR, and the solution has been [open-sourced on GitHub](https://github.com/satish860/PDF-Extraction-API).

- **Self-Translation Outshines Native Prompts**: Native language prompts were compared to translated English instructions; a member shared research on self-translation, recommending task-specific prompt strategies, and provided a [relevant paper](https://arxiv.org/pdf/2308.01223).

- **Singapore Member Shares LLM Workshop Notes**: Cedric from Singapore summarized key points on LLM mastery in his [workshop notes](https://gist.github.com/cedrickchee/c3d9f8fed88f1c486b883153a64ee7dc), which were well-received by the community.

- **Convergence on `axolotl` for Model Training and Tuning**: Multiple channels discussed using Axolotl for fine-tuning models with a reference to [Axolotl's main branch](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main?tab=readme-ov-file#quickstart-). Users are directed to the [Axolotl Docker image](https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018?context=explore) and shared a [setup guide](https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl).

- **Gradio Maintainer Jumps In**: Freddy, a Gradio maintainer, supports the community with Gradio resources [for quickstarts](https://www.gradio.app/guides/quickstart) and [developing chatbots](https://www.gradio.app/guides/creating-a-chatbot-fast), while another member indicates they'll have questions about a Gradio extension they've written.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Microsoft's Bold Move with Copilot+**: Discussions erupted over [Microsoft's announcement](https://blogs.microsoft.com/blog/2024/05/20/introducing-copilot-pcs/) of their "Copilot+ PCs" which have been observed to incorporate features remarkably similar to those of OpenAI. These PCs boast of 40+ TOPS performance, all-day battery life, AI image generation capabilities, and live captions in over 40 languages.

- **Dissecting the GPT-4o Context Window**: Amid debates regarding the context window of GPT-4o, the guild anchored on the understanding that a **32k** default size is the status quo, though the boundary of its capabilities remained a subject of intrigue.

- **Perplexity's Haiku Hurdles** *(Avoiding "Unleash")*: The guild uncovered a significant shift in Perplexity's default model usage, pivoting from GPT-3.5 to **Haiku** for regular users while **Sonar** remains exclusive for pro users, sparking discussions on model availability and strategy.

- **API Anomalies Afoot**: Concerns surfaced as Perplexity's API was found to lag behind its web counterpart, generating outdated headlines and unsatisfactory search outputs; further compounded by its beta status and limited endpoint support.

- **Community Collaboration Callout**: Members of the guild were nudging each other to make shared threads properly shareable and provided [visual aids](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) to help understand the process, while also sharing specific Perplexity AI search links for topics of collective interest.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Mixing Models - Not a Perfect Blend**: Discussions about integrating *Lightning* and *Hyper models* with base stable models revealed that while this approach could reduce image generation steps, incompatible architectures often lead to low-quality results.

- **EU AI Act Concerns Rise**: Users criticized the newly approved [EU AI Act](https://www.consilium.europa.eu/en/press/press-releases/2024/05/21/artificial-intelligence-ai-act-council-gives-final-green-light-to-the-first-worldwide-rules-on-ai/), particularly the watermarking requirements, which could pose difficulties for AI-generated content creators.

- **Local AI Setup Woes**: The community shared struggles with implementing *Stable Diffusion* locally, especially on AMD GPUs. The consensus hinted at a preference for Nvidia GPUs due to setup simplicity and performance advantages.

- **Quality Control for AI-Generated Content**: There was palpable discontent with the flood of low-effort, generic, and often sexualized AI-generated images in various online spaces, suggesting a need for better content curation and value assessment in the AI art space.

- **GPUs Debate - Nvidia Wins Favor**: A lively debate confirmed Nvidia GPUs as the preferred choice for running *Stable Diffusion*, with recommendations favoring versions with at least 12GB of VRAM for optimal AI performance.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX Implementations Face TFLOPS Discrepancies**: Engineers shared difficulties in benchmarking **JAX** implementations of pallas, naive, and flash v2. Shared memory errors and TFLOPS discrepancies on GPUs were reported, highlighting the necessity for precise performance measurements.

- **Mixed Opinions on Preprints and Academic Publishing**: The guild debated the usage of preprints on platforms like ArXiv. The consensus appears to be shifting, with major journals increasingly accepting preprints, signaling a change in how academic dissemination is being approached.

- **GPT-3's Randomness at Zero Temperature**: Conversations revolved around GPT-3's non-deterministic outputs at temperature 0, with insights into potential hardware level factors such as CUDA kernel non-determinism. Mentioned resources include an [arxiv paper](https://arxiv.org/abs/2210.14986) and an OpenAI [forum discussion](https://community.openai.com/t/run-same-query-many-times-different-results/140588).

- **Small Data Set Dilemmas**: In a brief interjection, members talked about the challenges of training AI on small datasets, pointing out that the performance often lags behind models trained on much larger corpuses, like the entirety of the internet.

- **Interpretable Machine Learning Gains Traction**: Excitement brewed over Anthropic's work on interpretable machine learning features, which can be further explored [here](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).

- **MCQ Randomization Query in lm-evaluation Harness**: Guild members raised concerns regarding the lack of answer choice randomization for MCQs within **lm-eval-harness**, especially for datasets like **SciQ** and **MMLU**, suggesting the potential for benchmark biases.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Microsoft's Phi-3 Integration into Transformers**: Microsoft announced the release of **Phi-3** models with up to 128k context and a vision-language (VLM) version, accessible on [HuggingFace](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct). These releases offer new possibilities for instruction-based and vision-language AI tasks.

- **ZeroGPU Bolstering Open-Source AI**: With a **\$10M ZeroGPU** initiative, Hugging Face is supporting independent and academic creators by providing free GPU resources, reaching over 1,300 spaces since May 1, 2024. See the [official tweet](https://x.com/ClementDelangue/status/1791115403734778185) for more details.

- **Struggles in Fine-Tuning Large Models**: The community engaged in discussions concerning the challenges of fine-tuning models like **Falcon-180B**, noting the need for hardware beyond an **8xH100** configuration. There are ongoing efforts to adapt embedding quantization in models like **Llama-8B** for more efficient memory usage.

- **Legislative Watch on AI**: Conversations indicate apprehension towards California's AI regulatory law and its implications for startups versus large companies. [NousResearch's Discord server](https://discord.gg/jqVphNsB4H) was suggested for deeper discourse on the topic.

- **Tooling and Contributions in AI**: Developers contributed several tools and datasets, such as a markdown note-taking app named [Notie](https://github.com/branyang02/notie), a Docker-friendly static wiki with Hexo.js, and various new models like the multilingual **NorskGPT-Llama3-70b**. There's also mention of a tool called [SDXL Flash](https://huggingface.co/spaces/KingNish/SDXL-Flash) purportedly generating high-quality images in seconds, showcasing the dynamism in AI tool development.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Dual GPU Dynamics in LM Studio**: LM Studio can handle dual GPUs, but they must be of the same type, and users should align VRAM capacities for optimal performance. Configuration for multiple GPUs involves creating and modifying a preset file in the system.

**Prompt Precision and Levity**: Users suggest quoting text directly in prompts for clarity, while the light-hearted term "prompt engineering" was used to describe meticulous prompt crafting strategies.

**Phi-3 Models in the Spotlight**: Integrating the **Phi-3** models into llama.cpp is a work in progress, with users eagerly waiting for a beta release and an LM Studio update to support the new models. Meanwhile, quantization advice for running **Phi-3 Medium** suggests staying at **Q4** or below.

**ROCM Realm for Linux**: Linux users expressed their interest in ROCm test builds, with the acknowledgment of challenges running **Phi-3-medium-128k** models due to tensor mismatch errors on ROCm platforms.

**Intriguing New Model Releases**: **Mistral v0.3 Instruct**, featuring an improved tokenizer and function calling support, is now available for use, offering advancements in language model functionality. Access it on the [lmstudio community Huggingface page](https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Apple ID Unlock with a Twist**: Engineers revealed a [new website](https://x.com/crinquand_paul/status/1793037790864687448) to bypass Vision Pro app restrictions for non-US Apple IDs, possibly interesting for those looking to access geo-restricted AI tools.

- **Enhanced Moondream Release Pushes Limits**: The latest update to Moondream has increased image resolution up to **756x756**, and improved TextVQA scores from **53.1 to 57.2**, marking a ~0.5% improvement on various benchmarks, as detailed in [this tweet](https://fxtwitter.com/vikhyatk/status/1792512588431159480?s=19).

- **Phi-3 Small on the Horizon?**: Speculation is rife on Microsoft's release strategy for Phi models as engineers shared insights into the availability of Phi 3, 7, and 14. Yann LeCun debunked rumors about the upcoming **LLaMa 3 400B+** model being closed-weight, pointing to its continued open status on [Twitter](https://x.com/q_brabus/status/1793227643556372596?s=46).

- **SB 1047 Stirs the Pot**: California's SB 1047 has engineers worried over its implications for open-source software (OSS), highlighted by shared [bill text](https://legiscan.com/CA/text/SB1047/id/2919384) and Meta being criticized for alleged regulatory manipulation.

- **Anthropic's Cognitive Cartography**: Anthropic's efforts to trace the cognitive map of language models captured engineers' attention, providing a potentially valuable resource to those focused on AI interpretation. Conversations around home setups for LLM inference, with personal infrastructure using 2x **4090s**, and platforms like **Runpod** and **Replicate** were up for discussion due to convenience, despite some platforms being harder to navigate.

- **Phi-3 Vision Drops with Depth and Access**: Launched with a comprehensive educational package, engineers discussed the **128K context** multi-modal model, Phi-3 Vision, providing links to Microsoft resources like the [Tech Report](https://aka.ms/phi3-tech-report) and the model on [Hugging Face](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct).

- **Grand Designs of Digital Knowledge**: A conversation emerged around Obsidian's knowledge graph visualization, likened to "synthetic brains", and expanded to cover its plugin integrations and data philosophy, complemented with a [knowledge graph time-lapse video](https://youtube.com/shorts/4YQhH61tvOc?si=0Dx1KyJP8VMz-pXY) and explanatory videos for users new to Obsidian.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **SASS Crash Course Wanted**: Engineering guild members are seeking guidance on how to learn **Syntactically Awesome Style Sheets (SASS)**, an extension of CSS with a focus on maintaining style sheets efficiently.
- **CUDA Curiosity on Function Modifiers**: There's an ongoing discussion about function qualifiers in **CUDA**, including why a function can be both `__device__` and `__host__` but not `__global__` and `__host__`.
- **Optimizations and Pitfalls in Torch & Numpy**: Members are comparing the performance of `torch.empty_like` with `torch.empty` and discussing memory leaks caused by `numpy's np.zeros_like`. There are also shared insights on compiling_issues with **ResNet blocks**, leveraging user-defined **Triton kernels** for optimization, and an informative [PyTorch tutorial](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html).

- **Legislative Buzz for AI Safety**: There's a vibrant conversation about the passing of [SB 1047](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047), a safety and innovation bill that sets the stage for more regulated AI development, alongside the mention of an ultra-compact ray-casting engine described [here](https://frankforce.com/city-in-a-bottle-a-256-byte-raycasting-system/).

- **Technical Dive into GitHub Pull Requests**: There are deep dives into GitHub pull requests focusing on determinism in encoder backward passes, DataLoader refactoring for large datasets, HellaSwag evaluation in C, and determinism in kernel operations, reflecting the community’s emphasis on efficiency and precision. Links such as [this PR for deterministic encoder backward kernels](https://github.com/karpathy/llm.c/pull/442) and [this one for a DataLoader refactor](https://github.com/karpathy/llm.c/pull/440) are part of this roundup.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Artificial Voicing Controversy**: An AI-generated voice similar to that of **Scarlett Johansson** led to concerns over ethical practices in AI after OpenAI's model was noted to create a voice "eerily similar" to hers. OpenAI's later decision to remove the voice came after a request for transparency by Johansson's legal team.
  
- **Chatbots Galore**: For **coding assistance**, users recommended alternatives to **GPT-3.5**, singling out **Meta AI’s Llama 3** and **Mistral Large** as effective, free options. In contrast, there was dissatisfaction with Microsoft's **Copilot** owing to its perceived **intrusiveness** and telemetry issues.

- **Tools and Tricks for Tighter Tokens**: In managing token usage and response verbosity, AI Engineers advised setting **max tokens** and using **output templates** to create succinct responses. Regarding custom tools, some developers cited stronger results with their own prompts as compared to using aids like **CodePilot**.

- **Platform and Model Tweaks Needed**: Participants pointed out formatting issues, such as unwanted line breaks in **OpenAI Playground's** output and inconsistent newline handling. Additionally, service outages prompted the sharing of the [OpenAI Status Page](https://status.openai.com) for service monitoring.

- **Microsoft Expanding Multimodal AI**: Microsoft introduced the **Phi-3-vision**, which combines language and vision, remarking on its potential for various applications. For further reading, members referred to a [blog post](https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/) detailing new models added to the **Phi-3 family on Azure**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Community Meeting Recap**: Mojo enthusiasts can catch up on the latest community meeting by watching the [recording](https://www.youtube.com/playlist?list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D) which covered topics on Basalt and Compact Dict. The meeting signaled the deprecation of Tensors in Mojo, opening a dialogue on developing new libraries for numerical and AI applications.

- **Python IPC vs. Threading**: For long-running queries in a Tkinter app, solutions ranged from threading, message queues, to IPC modules to prevent UI lag. A [link to RabbitMQ's Pika Python client tutorial](https://www.rabbitmq.com/tutorials/tutorial-six-python), although promising, led to implementation difficulties.

- **Mojo's Technical Evolution and Practices**: Discussion on Mojo revealed no official package manager but `.mojopkg` files are in play, particularly with [lightbug_http](https://github.com/saviorand/lightbug_http/releases/tag/latest-build). Optimizations in Mojo are MLIR-backed, with ongoing curiosity about their impact on custom data types. `math.bit` has now been aptly renamed to `bit`, with adjustments to several function names like `bswap` to `byte_reverse`.

- **Nightly Build and Dev Challenges**: Nightly build discussions included a PR issue with a commit by the wrong author, leading to a DCO test suite failure, addressed on [GitHub](https://github.com/modularml/mojo/pull/2739). Delays in the nightly release were traced to GitHub Actions, confirmed via [GitHub Status](https://www.githubstatus.com/). The `math.bit` module was also renamed to `bit`, amending function names for clarity.

- **Performance Optimization Suggestions**: When sorting small data sets, sorting an array of pointers can be more efficient. Regarding **DTypePointer memset**, a vectorized version performed 20% faster for 100,000 bytes but didn't scale up as effectively with larger data, due to potential issues with "using clobber memory".



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Voice AI's Legal Labyrinth**: Utilizing a voice actor mimicking Scarlett Johansson raised legal and ethical debates about 'passing off' rights, with members reflecting on the [Midler v. Ford Motor Co.](https://en.wikipedia.org/wiki/Midler_v._Ford_Motor_Co.) case as a potential precedent.
  
- **Investigating Dataset Disappearances**: The sudden removal of the Sakuga-42M dataset, involved in cartoon animation frame research, has left members puzzled about potential legal triggers, stirring up discussions about the broader implications of sharing datasets within legal confines.

- **Microsoft's Multimodal Model Causes a Stir**: Discussion on Microsoft's **Phi-3 Vision** model delved deep into its mechanics, showcased by [Hugging Face](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct), sparking conversations about its functionality, particularly when compared with **GPT-4**'s color-sorted chart outputs.

- **Anthropic paper perplexes engineers**: The recent Anthropic scaling paper has been marked as heavy yet unread, suggesting that despite its potential significance in the field, it may need clearer distillation to be fully appreciated by practitioners.

- **Old School Synthetic Voices Charm the Community**: Members took a stroll down memory lane, reminiscing about the DECtalk voice synthesis technology and shared nostalgia through a [Speak & Spell video](https://youtu.be/RpeegJ0J5mE?t=121), which was one of the earliest introductions to personal computing for many.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GPT-4o Paves the Path for Document Parsing**: GPT-4o has been leveraged to parse complex documents like PDFs and slide decks into structured markdown, despite challenges with background images and irregular layouts, using [LlamaParse](https://t.co/g5TG7brSwt). Details are available [here](https://t.co/vhtYzsleh2).

- **Secure Containerized Code Execution on Azure**: Azure Container Apps are enabling the secure execution of LLM-generated code in dynamic sessions. Further insights are provided in these Azure-related links: [Container Apps](https://t.co/2cnsBH411k) and [Code Security](https://t.co/lTrUPoTMcF).

- **Introduction to OpenDev AI Engineers**: The release of a webinar discussing OpenDevin, a platform designed for creating autonomous AI engineers, offers a tutorial by Robert Brennan. Interested viewers can find it [here](https://t.co/a22k0zsV3n).

- **Batch Inference Bolsters GenAI Capabilities**: The latest on batch inference processing for GenAI applications suggests major benefits for data analysis and querying capabilities. Delve into the details via these links: [Batch Inference Integration](https://t.co/vnuvvypZCz) and [GenAI Techniques](https://t.co/M0vQQ1uAki).

- **Navigating LlamaIndex Challenges and Solutions**: AI engineers have wrestled with LlamaIndex challenges, from setting up document previews in chat frontends to errors like `"ModuleNotFoundError"` and `"pydantic.v1.error_wrappers.ValidationError"`. Solutions to these issues involve import path corrections and prompt removal, while indexing strategies, such as retrievers using cosine similarity and HNSW, are under discussion for scaling efficiency.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Typing Quirks Spark Role-playing Debate**: Members humorously identified two main types of **OpenRouter** users: those seeking AI companionship and those delving into fantasy narratives. The conversation took a light-hearted dive into the role-playing tendencies of some users.

**Eyes on Phi-3**: The **Phi-3 Vision Model**, praised for high-quality reasoning, was introduced on the server. The model's attributes can be explored through [HuggingFace](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct).

**Verbose Wizard Needs a Trim**: **Wizard8x22** model's verbosity issues are recognized, with an adjustment to the repetition penalty proposed as a solution. The dialogue extended to compare other models' performance, highlighting that model behavior is not consistent across the board.

**Billing Blues and Nonprofit Woes**: A user's billing error on a student platform spurred discussion, leading to a temporary fix involving re-entering billing info. Hopes for nonprofit discounts in the future were also expressed.

**Experimenting with LLM Action Commands**: Innovative use of **LLMs** was shared through a [Twitter thread](https://x.com/leonjcoe/status/1792946945528320382), exploring action commands as a fresh way to enhance interactions with language models. Feedback from fellow engineers was solicited to push the boundaries of current LLM interaction paradigms.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Phi Models Join the Fray**: The launch of **Phi-small** and **Phi-medium** prompted discussions about the characteristics of **Phi-3 Vision**, with confirmations that it represents a new and slightly larger variant.

**Meta's Model Decisions Cause Stir**: A tweet suggested **Meta** might keep its 400B model closed due to legislative fears, but this was refuted by another source stating the **model will remain open-weight**. The confusion underscores the delicacy of sharing large-scale model weights in the current regulatory landscape. 

**OpenAI Under Fire for Unkept Promises**: OpenAI has disbanded its **Superalignment team** due to the unfulfilled promise of 20% compute resource allocation, sparking resignations. This, coupled with a scandal involving NDAs and vested equity issues for ex-employees, casts a cloud over OpenAI's leadership and transparency.

**AI Performance Takes a Drawback**: Microsoft's Surface drawing AI faces criticism due to latency issues resulting from cloud-based safety checks — reflecting the compromises between local processing power and safety protocols in AI applications. 

**The Trope of Researcher Titles**: Amazement was expressed at **Anthropic** now boasting over 500 'researchers', igniting a conversation about the dilution of the 'researcher' title and its implications for perception in the tech industry.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Cohere Integration and Tokenizer Troubles**: Engineers are working on integrating [Cohere (commandr)](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files) into the **Axolotl system**, while resolving tokenization issues with references to the `CohereTokenizerFast` [in the documentation](https://github.com/huggingface/transformers/blob/d24097e0229485287ff4959258c552168bd898c6/src/transformers/models/cohere/tokenization_cohere_fast.py#L51C7-L51C26).

- **Discovering Tiny Mistral and Distillation Pipeline Updates**: A **Tiny Mistral model** for testing custom functions was shared, as the community discussed ongoing work on a distillation pipeline for **Mistral models**, reported to be functioning decently.

- **Full Finetuning Versus LoRA Discussion**: There was a constructive back-and-forth over full finetuning versus **LoRA** with insights on performance differences, particularly around style retention for model adjustments, also suggesting direct referencing of the [Axolotl GitHub README](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenization-mismatch-bw-inference--training) for tokenization issues.

- **Axolotl's Next Major Release and GPU Finetuning Woes**: Users expressed curiosity about the next stable major release of **Axolotl** and discussed challenges with GPU memory requirements when finetuning using `examples/mistral/lora.yml`, seeking advice on managing `CUDA out of memory errors`.

- **Guidance on LoRA merges and State Dictionary Offloading**: Clarification was given on setting `offload_dir` for **LoRA merges**, pointing out the importance of using the `offload_state_dict` function post-merge to handle large model state dictionaries, referring to the [code search in Phorm AI](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=dce0e2d6-3e84-461f-a383-70860ed4ddfb)).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Langchain JS Awaits Refinements**: Engineers discussed the utility of **Langchain JS** for quick prototyping, despite lagging behind its Python counterpart in refinement. Plans for rearchitecture promise enhancements in future versions.

- **Scale AI Hits the Billion-Dollar Jackpot**: [Scale AI](https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/) has raised a staggering $1 billion in a funding round, skyrocketing its valuation to $13.8 billion, with the phasing forecast of profitability by the end of 2024.

- **Phi Packs a Punch**: Microsoft's **Phi 3 models** with links to 4K and 128K context lengths have debuted and are being praised for their capacity to run on platforms as light as a MacBook Pro M1 Pro. The community is scrutinizing them for competitive performance against leading models like **Mixtral**, **Llama**, and **GPT**.

- **Anthropic Defines Features with Dictionary Learning**: Anthropic has made significant strides with dictionary learning in their frontier model, allowing millions of features to be extracted. This is viewed as a leap forward in AI safety and effectiveness, transforming the handling of model activations.

- **Humane Eyes a Ripe Acquisition after AI Pin Stumbles**: Humane is seeking acquisition after their AI Pin device's market obstacles, with talks indicating a valuation aspiration between $750 million and $1 billion. Conversations revolve around the difficulties of hardware innovation in a market dominated by giants like Apple.

- **Survey Paper Club: Condensing AI Research**: Members are invited to join the **Survey Paper Club** for efficient exposure to multiple research papers within an hour, with email notifications facilitated upon [signing up](https://lu.ma/e5nk2ebp).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Community Specs vs LangChain**: Discussions articulated distinctions between **LangChain** and **LangChain Community** versions; the former's architecture is elaborated in the [official documentation](https://python.langchain.com/v0.2/docs/concepts/#architecture).

- **LangServe 'invoke' Woes**: Technical issues in **LangServe** concerning the 'invoke' endpoint which fails to provide comprehensive outputs were reported, spurring debate across several channels, with users flagging inconsistencies in output delivery. Specific problems included the absence of document retrieval and empty outputs, as documented in [LangServe discussion #461](https://github.com/langchain-ai/langserve/discussions/461) and related GitHub issues.

- **Operational Issues with RemoteRunnable**: Inconsistency was noted when **RemoteRunnable** did not perform as expected, unlike the **RunnableWithMessageHistory**, leading to missing document sources and affecting the operational reliability ([see GitHub issue](https://github.com/langchain-ai/langserve/issues/618)).

- **PDF Powered by Upstage AI Solar and LangChain**: A [blog post](https://medium.com/@sonam.gupta1105/creating-a-pdf-query-assistant-with-upstage-ai-solar-and-langchain-integration-6631280093b5) was shared guiding on harnessing the new **Upstage AI Solar LLM** with **LangChain** to build a PDF query assistant.

- **LangServe AWS Deployment Made Easier**: Members were directed to a [Medium article](https://medium.com/aimonks/deploy-langserve-application-to-aws-2d34b6ee5c1a) that simplifies deploying LangServe on AWS, eschewing the complexities of cloud technologies such as Terraform, Pulumi, or AWS CDK.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Tech Talk: OpenInterpreter's Device Dialogues**: Engineers are exploring how Open Interpreter can create links between apps and devices, utilizing tools like Boox E Ink tablets, OneNote, and VSCode. There's particular interest in using Open Interpreter for querying code or papers without browser intervention.

**Speedy GPT-4o Troubleshot**: While integrating GPT-4o with Open Interpreter, users note a minimum 5x speed increase but face challenges with error messages pertaining to API keys.

**Newline Nuisance in Gemini**: Code execution is being hindered in models such as Gemini 1.5 and Gemini Flash due to unnecessary newline characters; the absence of "python" declarations in code blocks also came under scrutiny.

**Legislative Lore and AI**: California’s controversial AI bill and subsequent discussions have ignited the community, with an [open letter](https://x.com/Scott_Wiener/status/1792572175116816853) from Senator Scott Wiener being circulated and debated for its emphasis on responsible AI development.

**Bill Gates Foresees Friendlier AI**: Gates recently penned thoughts on the future of AI in software, anticipating interfaces that can handle tasks through simple language directives, akin to a friend's assistance; his article is gaining traction among tech enthusiasts. An unofficial ChatGPT macOS app waitlist workaround made rounds on [Twitter](https://x.com/testingcatalog/status/1793347117458636981), demonstrating interest in quicker access to AI software tools.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Trigonometric Redefinition a No-Go**: Community members debated the efficacy of attempting to redefine trigonometric functions such as **sine** using Taylor series, with the consensus being that it's an unnecessary reinvention. IBM's practical approach to partitioning intervals for functions like sine was cited, showing that achieving perfect accuracy in functions is possible with established methods.

- **IBM's Code Holds the Answers**: Participants shared **IBM’s implementation** of the sine function, highlighting the intricacies of achieving perfect accuracy. Further, they referenced IBM’s range reduction solution for large numbers which is complex but doesn’t generally impact performance.

- **Training Mode Tips and Tricks**: In **tinygrad**, the use of `Tensor.train()` and `Tensor.no_grad` was explained for toggling gradient tracking. Helpful code examples, such as this [cartpole example](https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e46c6e1594c2dcc/examples/beautiful_cartpole.py#L58), illustrate the usage and benefits of these mechanisms.

- **Under the Hood of `Tensor.train`**: It was made clear that `Tensor.train` is effectively managing the `Tensor.training` status. For those preferring direct control, manually setting `Tensor.training` is an option, supported by tinygrad’s [backend implementation](https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e46c6e1594c2dcc/tinygrad/tensor.py#L83).

- **Nailing Views with Movement Ops**: A discussion unfolded around the behavior of chained movement operations and their potential to create multiple views. An example using `ShapeTracker` demonstrated how specific op combinations could produce such scenarios.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**SFT vs Preference Optimization Debate**: In a discussion on model training strategies, a member distinguished **Supervised Fine-Tuning (SFT)** as enhancing the model's probability distribution for target data points, whereas **Preference Optimization** adjusts both desired and undesired outcomes. They questioned the prevalent use of SFT over Preference Optimization, which may offer a more rounded approach to model behavior.

**Excitement Over Phi3 Vision's Low-Parameter Efficiency**: One engineer highlighted the development of **Phi3 Vision** with only 4.2 billion parameters as a significant advancement for low-latency inference in image processing tasks. Asserting that this could have groundbreaking implications for robotics, the model was praised for potential throughput improvements, as links to the announcement were shared ([source](https://x.com/jphme/status/1792950682695479734)).

**Comparing Image Models Between Moondream2 and Phi3 Vision**: The community weighed in on the performance of **Moondream2** comparative to **Phi3 Vision** for image-related tasks. While **Moondream2** has had issues with hallucinations, a member mentioned efforts to mitigate this, showcasing the ongoing pursuit of fidelity in image models ([Moondream2](https://huggingface.co/spaces/vikhyatk/moondream2)).

**Mixed Reactions to Microsoft's Model Drops**: The release of **Microsoft's 7b and 14b Instruct models** sparked diverse opinions, from concerns about their limitations in certain languages to optimism about their utility in complex reasoning and extraction tasks. The discussion reflects the community's critical analysis of newly released models and their capabilities.

**Skepticism Towards Meta's 400b Model**: With concerns circulating in the community about **Meta potentially not releasing a 400b model** as open source, one member highlighted skepticism by pointing to the uncertain credibility of the source, nicknamed Jimmy. This indicates a critical attitude toward rumor validation within the community.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere is hiring**: An enthusiastic member shared a [career opportunity at Cohere](https://cohere.com/careers), highlighting the chance to tackle real-world problems with advanced ML/AI.
  
- **VRAM Calculator Intrigues**: Engineers are discussing the findings of the [LLM Model VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator), questioning the higher VRAM use of the Phi 3 Mini compared to Phi 3 Medium for identical context lengths.

- **Bilingual Bot Integration Quest**: Multiple posts indicate a member searching for a guide to incorporate **Command-R** into **BotPress**, requesting help in both English and Spanish.

- **Link Confusion Alert**: There is confusion over accessing the Cohere careers page, with at least one member unable to find the correct page through the provided link.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Banter About AI Companions**: Discussion sparked by the phrase *"AI waifus save lives!"* led to a conversation about potentially emotional AI, alluding to the relevance of *sentiment analysis* for chatbots.
- **Emotional Intelligence in Chatbots on the Rise**: Shared **VentureBeat article** prompts engineers to consider the implications for business bots when AI begins to 'understand' emotions, which could be significant for **user experience** and **interface design**. [VentureBeat article on Emotional AI](https://venturebeat.com/ai/exclusive-inflection-ai-reveals-new-team-and-plan-to-embed-emotional-ai-in-business-bots).
- **3D Chatbots Gaining Traction**: A member from **4Wall AI** highlighted their ongoing work on **3D character chatbots**, suggesting new opportunities for **human-computer interaction** within the field.
- **Pop Culture Meets AI**: A reference to *"Just Monika"* prompted sharing of a *Doki Doki Literature Club* GIF, showcasing how pop culture can influence dialogues around AI personas. [Ddlc Doki Doki Literature Club GIF](https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242).



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**Snapdragon Dev Kit Sparks Debate**: Qualcomm's new Snapdragon Dev Kit priced at $899.99, featuring Snapdragon X Elite and boasting 32GB of LPDDR5x RAM and 512GB NVMe storage, has sparked discussions on cost-effectiveness compared to the previous $600 model, as detailed on [The Verge](https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite) and [Microsoft Store](https://www.microsoft.com/en-us/d/windows-dev-kit-2023/94k0p67w7581?activetab=pivot:overviewtab).

**Mac Mini Server Gets Thumbs Up**: An AI engineer shared their success in using a Mac Mini as a reliable Llamafile server with Tailscale, praising its zero-cold start feature and seamless 'llm' CLI integration, suggesting a practical use case for developers needing stable server solutions.

**Affordable Dev Kits in Demand**: Discussion among users indicates a strong desire for more affordable development kits, with aesthetic preferences also being voiced, such as a wish for a translucent case design, yet no specific products were mentioned.

**Smalltalk AI Shows Promise**: A member introduced Claude's ability to engage in Smalltalk, using "What are frogs?" as an example question overcome by the AI with a basic reply about amphibians, indicating advances in AI's conversational capabilities.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**Brevity Blunder in Llama3/Phi3**: An inquiry was made regarding how to stop **llama3/phi3** from truncating responses with "*additional items omitted for brevity*," but no solutions or further discussion ensued.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Community Events for Engineering Minds**: Mozilla AI announced the initiation of **member-organized events** to inspire idea-sharing and community interaction, featuring talks, AMAs, and demos, starting with an [AMA hosted by LLM360](https://discord.com/events/1089876418936180786/1240722407594004561).
  
- **AMA on Open-Source LLMs**: [LLM360](https://www.llm360.ai/) hosted an [AMA session](https://discord.com/events/1089876418936180786/1240722407594004561), diving into the specifics of their work with open-source LLMs and attracting a tech-savvy crowd.

- **Embeddings with Llamafiles**: Kate Silverstein, a Staff Machine Learning Engineer, will demonstrate the use of llamafiles for generating embeddings and elaborate on her [latest blog post](https://discord.com/channels/1089876418936180786/1242235316170129439).

- **Events Calendar a Click Away**: Mozilla AI encourages members to frequent the events calendar for a robust lineup of community-led discussions and technical activities.

- **Query on Model Spec in LLaMA CPP**: A member sought clarity on using a **tinyllama** model via terminal, questioning whether the `model="LLaMA_CPP"` specification is necessary and which model is actually in play when the code snippet runs successfully.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1242398812887056434)** (1309 messages🔥🔥🔥): 

- **OpenAI and Dataset Challenges**: Members discussed various **dataset challenges** including converting formats, using **ShareGPT**, and optimizing training parameters such as batch sizes. One user shared that they "spent 5 hours scraping site into alpaca format" only to find it unhelpful, indicating how persnickety these processes can be.
- **Phi-3 out!; Users skeptical but excited**: **Phi-3 models** from Microsoft generated excitement with members mentioning [Phi-3-Medium-128K-Instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct), yet some noted skepticism about the validity of its benchmarks. One user said, "*literally*".
- **Latest Legal Constraints**: Conversations about **AI regulations** like California's SB 1047 law sparked discussions on the implications for open-source models. "Meta plans to not open the weights for its 400B model," catalyzed a debate, with users expressing concerns about its global effects.
- **Technical Issues and Workarounds for Colab/Kaggle**: **Common technical glitches** were noted, especially around updates breaking compatibility. User `theyruinedelise` pointed out necessary workarounds like restarting Colab sessions due to the *"Pytorch not detecting T4's properly"* issue.
- **Unsloth Platform Developments**: Users discussed **new model support** on the Unsloth platform such as [Mistral v3](https://twitter.com/danielhanchen/status/1793356226006511902), expressing excitement over improved fine-tuning features. "Unsloth now supports Mistral v3", facilitating easier adoption of cutting-edge models in the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Sao10K/Fimbulvetr-11B-v2">Sao10K/Fimbulvetr-11B-v2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/mistral-7b-v0.3-bnb-4bit">unsloth/mistral-7b-v0.3-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/fai">fai (fai)</a>: no description found</li><li><a href="https://x.com/Scott_Wiener/status/1792572175116816853">Tweet from Senator Scott Wiener (@Scott_Wiener)</a>: In recent weeks, there&#39;s been a flurry of discussion online about SB 1047, my bill on responsible development of the largest & most powerful AI frontier models. We’ve heard some incredibly thought...</li><li><a href="https://github.com/oKatanaaa/kolibrify/tree/master/examples/training_mini_dolphin">kolibrify/examples/training_mini_dolphin at master · oKatanaaa/kolibrify</a>: Curriculum training of instruction-following LLMs with Unsloth - oKatanaaa/kolibrify</li><li><a href="https://x.com/q_brabus/status/1793227643556372596">Tweet from QBrabus eu/acc (@q_brabus)</a>: @apples_jimmy @ylecun @iamgingertrash Question: Regarding the upcoming LLaMa 3 400B+ model, will it be open-weight? There are several rumors about this...  Answer: No, it is still planned to be open a...</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-128k-instruct">microsoft/Phi-3-medium-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://x.com/erhartford/status/1791573520176025716">Tweet from Eric Hartford (@erhartford)</a>: In response to California&#39;s SB 1047 and OpenAI&#39;s closed-source stance, Cognitive Computations introduces Patchy-2.0. This license mirrors Apache-2.0 but expressly forbids OpenAI and the State ...</li><li><a href="https://huggingface.co/docs/datasets/en/loading#csv">Load</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/hsiehjackson/RULER?tab=readme-ov-file>">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models? - hsiehjackson/RULER</li><li><a href="https://github.com/huggingface/transformers/issues/11693">Flag to disable shuffling for data loader · Issue #11693 · huggingface/transformers</a>: 🚀 Feature request Currently, Trainer is shuffling the train_dataset by default and there is no flag to enable/disable it. @sgugger Motivation Even if shuffling the dataset brings a lot of benefits .....</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cxnrov/disappointing_if_true_meta_plans_to_not_open_the/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/e3Gvq4NDqvw?si=3b2lILNAiR5CZJMW">Scarlett Johansson demands answers after OpenAI releases voice &quot;eerily similar&quot; to hers</a>: Scarlett Johansson is demanding answers from OpenAI and its CEO, Sam Altman, after it released a ChatGPT voice that she says sounds &quot;eerily similar&quot; to her o...</li><li><a href="https://imgur.com/FhBnfFP">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cxw3u5/it_did_finally_happen_a_law_just_passed_for_the/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047">Bill Text - SB-1047 Safe and Secure Innovation for Frontier Artificial Intelligence Models Act.</a>: no description found</li><li><a href="https://x.com/Scott_Wiene">Tweet from FxTwitter / FixupX</a>: Sorry, that user doesn't exist :(</li><li><a href="https://github.com/unslothai/unsloth/issues/504">[URGENT] Colab is broken · Issue #504 · unslothai/unsloth</a>: Colab is broken currently - working on a fix</li><li><a href="https://x.com/danielhanchen/status/1792985678030221464">Tweet from Daniel Han (@danielhanchen)</a>: @GoogleColab @PyTorch @thechrisperry Update: An @UnslothAI community member (Edd) found Pytorch 2.3 is not detecting Tesla T4s correctly - Pytorch thinks Tesla T4 can support bfloat16, but it cannot. ...</li><li><a href="https://huggingface.co/datasets/Skorcht/schizoroleplaydataset">Skorcht/schizoroleplaydataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Skorcht/ariannarp">Skorcht/ariannarp · Datasets at Hugging Face</a>: no description found</li><li><a href="https://lu.ma/1wu5ppl5">GPU Optimization Workshop · Luma</a>: We’re hosting a workshop on GPU optimization with stellar speakers from OpenAI, NVIDIA, Meta, and Voltron Data. The event will be livestreamed on YouTube, and…</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_optim_utils.py#L1369>">pytorch/torch/distributed/fsdp/_optim_utils.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1242432957030076466)** (233 messages🔥🔥): 

- **MoRA sparks curiosity**: Members inquired about a new method called [MoRA](https://arxiv.org/abs/2405.12130) and shared plans to test its vanilla implementation. One noted that it appears to be a "scaled up" version of LoRA, optimized for measuring the intrinsic dimension of objective landscapes.

- **Dune series and philosophy discussions dominate**: Users engaged in a detailed discussion about the philosophical depth of the Dune series beyond its initial hero's journey. They noted that subsequent books become progressively more philosophical, moving away from simple narratives.

- **Sci-fi novels and recommendations flood the chat**: The conversation shifted to various sci-fi novels and recommendations, including Peter Watts' "Blindsight," which features unique takes on alien intelligence and vampires, described as "the hardest sci-fi that ever sci-fied."

- **Fondness for intricate sci-fi plots**: Users expressed enthusiasm for complex and intriguing sci-fi plots, comparing elements of hard sci-fi novels to modern AI's behavior. They discussed the appeal of realistic and imaginative alien life forms in literature over cliché humanoid representations.

- **Debate on movies versus reading novels**: Members compared the experience of watching sci-fi movies to reading novels, with some expressing a preference for the latter due to the more profound and imaginative storytelling. The conversation highlighted dissatisfaction with recent movie adaptations of popular sci-fi stories, noting a decline in quality compared to the depth found in books.

**Link mentioned**: <a href="https://arxiv.org/abs/2405.12130">MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning</a>: Low-rank adaptation is a popular parameter-efficient fine-tuning method for large language models. In this paper, we analyze the impact of low-rank updating, as implemented in LoRA. Our findings sugge...

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1242371434945187861)** (192 messages🔥🔥): 

- **Unsloth models face saving issues**: Users report problems with the `model.save_pretrained_gguf()` function documented in [GitHub Issue #485](https://github.com/unslothai/unsloth/issues/485), which breaks due to a `UnboundLocalError`.
- **Flash Attention causes CUDA issues**: Several users experienced errors with Flash Attention versions misconfigured for their setups and discussed switching to `xformers` instead. Consequently, starsupernova recommended [uninstalling and reinstalling](https://github.com/unslothai/unsloth) Unsloth without Flash Attention.
- **Pytorch ≥ 2.3 breaks T4 GPUs**: Multiple users reported compatibility issues with Pytorch version 2.3 on Tesla T4 GPUs, leading to recommendations to downgrade and disable bf16 support. A community workaround involved specifying `dtype` explicitly.
- **Guided decoding for YAML**: There was an in-depth discussion on leveraging guided decoding for structured output in YAML, with insights on using [grammars and constraining output](https://www.grammar-lib.com) effectively while prompting models. This includes potential support in vLLM using different syntaxes like JSON Schema or BNF.
- **Installation and training discrepancies**: Discrepancies in installation instructions and training behavior were analyzed, particularly focusing on the `trl` library and its versions affecting model training. Adjustments to ensure consistent setups and installations were advised, especially considering instability in recent library versions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Cocktail_party_effect">Cocktail party effect - Wikipedia</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py">unsloth/unsloth/kernels/cross_entropy_loss.py at main · unslothai/unsloth</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://x.com/danielhanchen/status/1792982364894929083">Tweet from Daniel Han (@danielhanchen)</a>: Oh no @GoogleColab upgraded to @PyTorch 2.3, and T4 GPUs don&#39;t work with Triton 2.3!  I tried downgrading Triton to 2.2, but it still fails. It seems like this is a Torch 2.3 issue.  @thechrisperr...</li><li><a href="https://huggingface.co/google/gemma-2b/discussions/60#664de2208ab2524c032b00b4">google/gemma-2b · Following blog for fine tuning gemma-2b doesn&#39;t yield same results</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/485">Using llama-cpp-python · Issue #485 · unslothai/unsloth</a>: Hi, Thanks for creating this wonderful package! The save_to_gguf currently fails because llama.ccp installation seems to be broken. Could something like llama-cpp-python be used instead?</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...</li><li><a href="https://github.com/unslothai/unsloth/pull/506/commits/2b23b9357aba25ab2f3a49d899045547d7dde1d7">Nightly by danielhanchen · Pull Request #506 · unslothai/unsloth</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#training">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/commit/5134a42f0689c0bb69aba12dc668755bdd4b4693">Nightly (#506) · unslothai/unsloth@5134a42</a>: * Update llama.py
 
 * offload
 
 * Update llama.py
 
 * Update llama.py
 
 * Update llama.py
 
 * Update llama.py
 
 * Update llama.py
 
 * Update llama.py
 
 * Update llama.py
 
 * continued pret...</li><li><a href="https://github.com/pytorch/pytorch/blob/b40fb2de5934afea63231eb6d18cc999e228100f/torch/cuda/__init__.py#L130C1-L151C1">pytorch/torch/cuda/__init__.py at b40fb2de5934afea63231eb6d18cc999e228100f · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1242371648557158461)** (7 messages): 

- **Superfantastic results ignite excitement**: One member expressed amazement with their results, describing them as "super fantastic." Another commented on their own struggles, stating they couldn’t get results below 52k and expressed anticipation for an upcoming article.
- **Recipe for success forthcoming**: A member mentioned they will "release the recipe this/next week," despite noting that it won't fully reproduce earlier results due to the use of proprietary data. They added that it might perform a bit better for English datasets.
- **Knowledge Graph Embeddings**: A member shared their past experience with Knowledge Graph Embeddings, mentioning difficulty in transitioning from a Neo4j graph to a PyTorch Geometric Dataset due to complex `cypher` queries. Another member implied that such a task should be easier with current tools.
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1242377577696329828)** (242 messages🔥🔥): 

- **Modal Learning Opens New Doors**: One member shared that their company uses Marker PDF, leveraging Surya OCR to convert PDFs into markdown format. They noted that the tool's results surpass other open models, and they have open-sourced the solution on [GitHub](https://github.com/satish860/PDF-Extraction-API).

- **Native Prompts vs. Translations?**: Members discussed the efficacy of native language prompts versus English prompts with instructions to translate. One member shared a [paper](https://arxiv.org/pdf/2308.01223) focusing on self-translation models, adding various experiences and suggesting task-specific strategies.

- **PDF Parsing and Multimodal LLMs**: Challenges in PDF parsing were highlighted with multiple tools such as LlamaParse, Unstructured, and table transformers mentioned, but none provided perfect results. There was interest in strategies involving multimodal LLMs and fine-tuning on target data.

- **Anthropic’s Sonnet Paper Sparks Interest**: A member shared a [link](https://www.anthropic.com/research/mapping-mind-language-model) to a paper on interpretability by Anthropic, sparking discussions about safety and steering model behavior. Another member added further insights with a related [Twitter thread](https://x.com/mlpowered/status/1792948212728524917).

- **Community Engagement with Modal and Tools**: Discussions included preference for tools like pyenv, mamba (through miniforge), and the ease of using GUI for language model fine-tuning. Members shared [installation guides](https://github.com/pyenv/pyenv?tab=readme-ov-file#automatic-installer) and discussed various workflows and their experiences with different packages and environments.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.00732">LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report</a>: Low Rank Adaptation (LoRA) has emerged as one of the most widely adopted methods for Parameter Efficient Fine-Tuning (PEFT) of Large Language Models (LLMs). LoRA reduces the number of trainable parame...</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: no description found</li><li><a href="https://www.loom.com/share/30d3b2e054f142fda5d905f95fedc29f">Exploring Fine-tuning with Honeycomb Example</a>: In this video, I walk you through the process of fine-tuning a model using the honeycomb example. I provide step-by-step instructions on cloning the repository, installing dependencies, and running th...</li><li><a href="https://huggingface.co/blog/sc2-instruct">StarCoder2-Instruct: Fully Transparent and Permissive Self-Alignment for Code Generation</a>: no description found</li><li><a href="https://youtu.be/C9p7suS-NGk?si=AM4sr3OXeFRKZo7c">Vincent Warmerdam - Keynote &quot;Natural Intelligence is All You Need [tm]&quot;</a>: In this talk I will try to show you what might happen if you allow yourself the creative freedom to rethink and reinvent common practices once in a while. As...</li><li><a href="https://github.com/imaurer/awesome-llm-json/?tab=readme-ov-file#local-models">GitHub - imaurer/awesome-llm-json: Resource list for generating JSON using LLMs via function calling, tools, CFG. Libraries, Models, Notebooks, etc.</a>: Resource list for generating JSON using LLMs via function calling, tools, CFG. Libraries, Models, Notebooks, etc. - imaurer/awesome-llm-json</li><li><a href="https://arxiv.org/abs/2212.09741">One Embedder, Any Task: Instruction-Finetuned Text Embeddings</a>: We introduce INSTRUCTOR, a new method for computing text embeddings given task instructions: every text input is embedded together with instructions explaining the use case (e.g., task and domain desc...</li><li><a href="https://arxiv.org/abs/2211.09260">Task-aware Retrieval with Instructions</a>: We study the problem of retrieval with instructions, where users of a retrieval system explicitly describe their intent along with their queries. We aim to develop a general-purpose task-aware retriev...</li><li><a href="https://x.com/mlpowered/status/1792948212728524917">Tweet from Emmanuel Ameisen (@mlpowered)</a>: Today, we announced that we’ve gotten dictionary learning working on Sonnet, extracting millions of features from one of the best models in the world.  This is the first time this has been successfull...</li><li><a href="https://youtu.be/Y9464wasHuE">How to run axolotl on JarvisLabs | Tutorial</a>: Check out axolotl on JarvisLabs : jarvislabs.ai/templates/axolotlCheck out axolotl github : https://github.com/OpenAccess-AI-Collective/axolotl</li><li><a href="https://github.com/conda-forge/miniforge?tab=readme-ov-file#install">GitHub - conda-forge/miniforge: A conda-forge distribution.</a>: A conda-forge distribution. Contribute to conda-forge/miniforge development by creating an account on GitHub.</li><li><a href="https://github.com/pyenv/pyenv?tab=readme-ov-file#automat">GitHub - pyenv/pyenv: Simple Python version management</a>: Simple Python version management. Contribute to pyenv/pyenv development by creating an account on GitHub.</li><li><a href="https://github.com/satish860/PDF-Extraction-API">GitHub - satish860/PDF-Extraction-API: A Marker Library based API for doing the Marker Response.</a>: A Marker Library based API for doing the Marker Response. - satish860/PDF-Extraction-API</li><li><a href="https://github.com/jondurbin/bagel?tab=readme-ov-file#prompt-formatting">GitHub - jondurbin/bagel: A bagel, with everything.</a>: A bagel, with everything. Contribute to jondurbin/bagel development by creating an account on GitHub.</li><li><a href="https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts">Dependency Resolution - pip documentation v24.1.dev1</a>: no description found</li><li><a href="https://github.com/marco-jeffrey/awesome-llm-resources">GitHub - marco-jeffrey/awesome-llm-resources: a collection of resources around LLMs, aggregated for the workshop &quot;Mastering LLMs: End-to-End Fine-Tuning and Deployment&quot; by Dan Becker and Hamel Husain&quot;</a>: a collection of resources around LLMs, aggregated for the workshop &amp;quot;Mastering LLMs: End-to-End Fine-Tuning and Deployment&amp;quot; by Dan Becker and Hamel Husain&amp;quot; - marco-jeffrey/aw...</li><li><a href="https://www.anthropic.com/research/mapping-mind-language-model">Mapping the Mind of a Large Language Model</a>: We have identified how millions of concepts are represented inside Claude Sonnet, one of our deployed large language models. This is the first ever detailed look inside a modern, production-grade larg...</li><li><a href="https://anywidget.dev/en/community/">Community | anywidget</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=goaBFxGhp6Y),">Enhancing Jupyter with Widgets with Trevor Manz - creator of anywidget</a>: In this (first!) episode of Sample Space we talk to Trevor Mantz, the creator of anywidget. It&#39;s a (neat!) tool to help you build more interactive notebooks ...</li><li><a href="https://github.com/VikParuchuri/surya">GitHub - VikParuchuri/surya: OCR, layout analysis, reading order, line detection in 90+ languages</a>: OCR, layout analysis, reading order, line detection in 90+ languages - VikParuchuri/surya</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions">CUDA Installation Guide for Linux</a>: no description found</li><li><a href="https://github.com/pyenv/pyenv?tab=readme-ov-file#automatic-installer">GitHub - pyenv/pyenv: Simple Python version management</a>: Simple Python version management. Contribute to pyenv/pyenv development by creating an account on GitHub.</li><li><a href="https://github.com/pyenv/pyenv-virtualenv">GitHub - pyenv/pyenv-virtualenv: a pyenv plugin to manage virtualenv (a.k.a. python-virtualenv)</a>: a pyenv plugin to manage virtualenv (a.k.a. python-virtualenv) - pyenv/pyenv-virtualenv</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/453">pip install flash-attn always happens ModuleNotFoundError: No module named &#39;packaging&#39;,but actually i have pip install packaging · Issue #453 · Dao-AILab/flash-attention</a>: Collecting flash-attn Using cached flash_attn-2.0.7.tar.gz (2.2 MB) Installing build dependencies ... done Getting requirements to build wheel ... error error: subprocess-exited-with-error × Gettin...</li><li><a href="https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl">Latent Space</a>: no description found</li><li><a href="https://huggingface.co/blog/chat-templates">Chat Templates: An End to the Silent Performance Killer</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://github.com/chujiezheng/chat_templates">GitHub - chujiezheng/chat_templates: Chat Templates for HuggingFace Large Language Models</a>: Chat Templates for HuggingFace Large Language Models - chujiezheng/chat_templates
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1242389380719579218)** (83 messages🔥🔥): 

- **Extracting Villa Attributes from User Prompts**: One member discussed extracting structured attributes like bedrooms and swimming pools from user-provided prompts about their villa wishes. They highlighted the importance of maintaining low latency and high performance, and expressed interest in using synthetic data for evaluation.
  
- **Workflows and Synthetic Data**: Another member shared their use case of predicting workflows and generating them using GPT-4 for various domains. They focus on using synthetic data to fine-tune Mistral models for providing workflow recommendations.

- **User Testing with LLM Agents**: A use case was presented for using LLM agents to conduct user tests for web applications, tuning prompts to capture user personalities and desired feedback. The focus lies in prompt tuning to effectively simulate user interactions.

- **Model for Grant Application Assistance**: One user proposed fine-tuning models to help UK farmers and organizations navigate and complete grant applications. They plan to combine natural language understanding with domain-specific knowledge from the UK government website.

- **In-Store Book Recommendation System**: An idea was put forward for creating a recommendation system that uses user queries to provide book suggestions from a bookstore's database. The system would rely initially on prompt engineering and RAG, with potential fine-tuning to reduce costs as the model scales up.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unstructured.io/">Unstructured | The Unstructured Data ETL for Your LLM</a>: Unstructured helps you get your data ready for AI by transforming it into a format that large language models can understand. Easily connect your data to LLMs.</li><li><a href="https://www.youtube.com/watch?v=sTQaJyrI-zg&list=PLVVTN-yNn8rvEwlY8ClxDUWeVPVfdifYj&index=8&ab_channel=StanfordOnline">Stanford CS25: V2 I Common Sense Reasoning</a>: February 14, 2023Common Sense ReasoningYejin ChoiIn this speaker series, we examine the details of how transformers work, and dive deep into the different ki...</li><li><a href="https://us06web.zoom.us/rec/share/GygkDuLtIVV5drfzJi_raZCXBPdkCVpSkmYVRHIhD9TPKWQVvDZxFvSxKM4Bllvr.z4fyIxneKpQgLdjM?startTime=1715705791000">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/">LlamaParse - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1242388319376248914)** (26 messages🔥): 

- **Cedric shares extensive notes from LLM workshop**: A member from Singapore, Cedric, shared his [notes from a workshop](https://gist.github.com/cedrickchee/c3d9f8fed88f1c486b883153a64ee7dc), summarizing key points about "Mastering LLMs". The notes were met with positive feedback with members expressing gratitude.
  
- **Pune Meetup Proposal Gains Interest**: A member from Pune suggested a local meetup and received enthusiastic responses. The intent to set up the event was emphasized with a follow-up message on logistics: *"[Possible Pune meetup ?]"*.

- **Growth in Singapore and Malaysia Community**: Several members from Singapore, Malaysia, and other parts of Asia introduced themselves. Collaborative enthusiasm was high with many members taking interest in discussing topics and meeting locally.

- **General Members' Greetings**: Multiple members from various parts of India and Asia introduced themselves, expressing interest in connecting with others. These introductions highlighted the geographical diversity and the active participation from different regions in Asia.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/cedrickchee/c3d9f8fed88f1c486b883153a64ee7dc">Mastering LLMs: A Conference For Developers &amp; Data Scientists</a>: Mastering LLMs: A Conference For Developers &amp; Data Scientists - mastering-llm-ft-workshop-1.md</li><li><a href="https://x.com/cedric_chee/status/1790638025397117031">Tweet from Cedric Chee (@cedric_chee)</a>: When and why to fine-tune an LLM:  - Extremely narrow problem - Prompt engineering is impractical - Quality vs. latency tradeoff - Data privacy  Long-live model fine-tuning.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1242410416890712094)** (77 messages🔥🔥): 

- **Satish's Surya OCR and Modal Issues**: "I have created the PDF extractor using Surya OCR" but faced issues with Modal running every time the model loads. Suggested to join Modal's Slack for quicker support as outlined [here](https://modal.com/slack).

- **Axolotl Running Issue**: Nisargvp faced trouble recognizing `axolotl.git` URL in Modal; suggested to refer to [Modal's LLM Finetuning sample repo](https://github.com/modal-labs/llm-finetuning).

- **Inference Configuration Confusion**: Intheclouddan ran into issues while setting up inference using a specific prompt format and was advised to use the full llama 3 chat template and shared related [example repo](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving).

- **Modal Credits Inquiry**: Numerous participants mentioned filling out forms and awaiting Modal credits. Charles shared the [claim form link](https://bit.ly/modal-credits) and mentioned the credits process details are in a specific Discord channel.

- **Training and Inference Execution Errors**: Troubleshooting related to execution errors showed that repeated attempts sometimes resolve the issues. Ripes suggested checking related discussions on Modal's Slack community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/modal_labs/status/1793310938277560646">Tweet from Modal (@modal_labs)</a>: Yes, we can fine-tune our own models.</li><li><a href="https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving">modal-examples/06_gpu_and_ml/llm-serving at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://modal-labs--heroicons.modal.run/">🎨 Generate Custom Heroicons 🎨</a>: no description found</li><li><a href="https://github.com/modal-labs/llm-finetuning/">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://bit.ly/modal-credits">Modal hackathon credits</a>: To claim your Modal credits, sign up for an account at https://modal.com/ first.  Then, let us know your username through this form.   For support, join the Modal Slack.  Here’s some examples to get s...</li><li><a href="https://github.com/satish860/PDF-Extraction-API/blob/main/app.py#L58">PDF-Extraction-API/app.py at main · satish860/PDF-Extraction-API</a>: A Marker Library based API for doing the Marker Response. - satish860/PDF-Extraction-API</li><li><a href="https://modal.com/docs/examples/llm-finetuning">Fine-tune an LLM in minutes (ft. Llama 2, CodeLlama, Mistral, etc.)</a>: Tired of prompt engineering? Fine-tuning helps you get more out of a pretrained LLM by adjusting the model weights to better fit a specific task. This operational guide will help you take a base model...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl.git">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://x.com/charles_irl/status/1793311021060489381">Tweet from Charles 🎉 Frye (@charles_irl)</a>: i love my job  lyrics by @ChatGPTapp pop punk song by @suno_ai_ @dingboard_ by @yacineMTB background removal by @remove_bg  and heroicon generator by me and @YirenLu running on @modal_labs  https://mo...</li><li><a href="https://modallabscommunity.slack.com/archives/C069RAH7X4M/p1711387685695179?thread_ts=1711051146.010029&cid=C069RAH7X4M">Slack</a>: no description found</li><li><a href="https://modal.com/jamesrequa/apps/ap-PtacgJR85SK41xlfulDzGg">Sign in</a>: Welcome back to Modal! Sign in to your Modal account by selecting an identity provider below.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1242669734512693248)** (10 messages🔥): 

- **Reverse Engineering Transformers benefits from interactive articles**: A member shared a comprehensive resource on reverse engineering transformer language models into human-understandable programs, inspired by the [Distill Circuits Thread](https://distill.pub/2020/circuits/) and other interactive articles like [Activation Atlases](https://distill.pub/2019/activation-atlas/). They also noted [Distill's hiatus](https://distill.pub/2021/distill-hiatus/) and mentioned that new content may be added in collaboration with other institutions.

- **Fine-Tuning Benchmarks Showcase Open-Source LLM Performance**: The [Predibase fine-tuning index](https://predibase.com/fine-tuning-index) offers performance benchmarks from fine-tuning over 700 open-source LLMs, highlighting that smaller models can deliver GPT-like performance through fine-tuning. Performance metrics are presented in interactive charts to help AI teams select the best open-source models for their applications.

- **Dedicated GitHub Repo for LLM Resource Collaboration**: A member created a [GitHub repo](https://github.com/marco-jeffrey/awesome-llm-resources) for better collaboration on LLM resources for a workshop by Dan Becker and Hamel Husain. They asked users not to directly edit the README.md file as it's auto-generated through GitHub actions and encouraged pull requests for contributions.

- **ML Engineering Book Added to LLM Resource Repo**: A member plans to add Stas' [ML Engineering book](https://github.com/stas00/ml-engineering) to the resource repo, highlighting its in-depth insights on training LLMs at scale, covering various aspects such as orchestration, good training loss, and planning. The book is praised as an invaluable resource despite its chunkiness due to the detailed coverage.

- **AI Model Comparison Website as a Favorite Resource**: A member recommended [artificialanalysis.ai](https://artificialanalysis.ai/models) for comparing and analyzing AI models across metrics like quality, price, performance, and speed. They noted the site's detailed metrics and FAQs for further details and highlighted the trade-offs between model quality and throughput.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://artificialanalysis.ai/models">Comparison of AI Models across Quality, Performance, Price | Artificial Analysis</a>: Comparison and analysis of AI models across key metrics including quality, price, performance and speed (throughput tokens per second &amp; latency), context window &amp; others.</li><li><a href="https://transformer-circuits.pub/">Transformer Circuits Thread</a>: no description found</li><li><a href="https://github.com/stas00/ml-engineering/">GitHub - stas00/ml-engineering: Machine Learning Engineering Open Book</a>: Machine Learning Engineering Open Book. Contribute to stas00/ml-engineering development by creating an account on GitHub.</li><li><a href="https://predibase.com/fine-tuning-index">The Fine-tuning Index</a>: Performance benchmarks from fine-tuning 700+ open-source LLMs</li><li><a href="https://github.com/marco-jeffrey/awesome-llm-resources">GitHub - marco-jeffrey/awesome-llm-resources: a collection of resources around LLMs, aggregated for the workshop &quot;Mastering LLMs: End-to-End Fine-Tuning and Deployment&quot; by Dan Becker and Hamel Husain&quot;</a>: a collection of resources around LLMs, aggregated for the workshop &amp;quot;Mastering LLMs: End-to-End Fine-Tuning and Deployment&amp;quot; by Dan Becker and Hamel Husain&amp;quot; - marco-jeffrey/aw...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1242545274329763912)** (36 messages🔥): 

- **Members plan to run Axolotl on Jarvis**: Multiple users expressed interest in experimenting with **Axolotl** and discussed the sign-up and credit allocation process for **Jarvislabs**. User `vishnu9158` shared the steps to start using **Axolotl** locally via a Docker image.

- **Credits for JarvisLabs**: Users inquired about getting credits after signing up on Jarvislabs. It was clarified that if the Jarvislabs account email differs from the course email, it might cause delays.

- **Creating and Running Axolotl Instances**: The community discussed running **Axolotl** instances using a Docker image and JupyterLab for fine-tuning. Vishnu9158 mentioned that a documentation and video tutorial are coming soon.

- **Blog posts for better understanding**: Several users, inspired by a previous suggestion, shared or planned to share their blog posts about their learning experiences on platforms like **Medium**. 

- **Hugging Face model issues resolved**: Some members faced issues accessing the **llama-3** model on Hugging Face, despite having access. Dhar007 provided steps to resolve this by creating and using an access token, but then ran into CUDA out of memory errors, suggesting adjustments in batch sizes.

**Link mentioned**: <a href="https://jarvislabs.ai">Jarvislabs: Making AI affordable and simple for everyone</a>: Jarvislabs is a platform that allows you to run and explore multiple AI framerworks on powerful GPUs with zero setup

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1242533399277862912)** (11 messages🔥): 

- **Hugging Face model filter issue**: Users discussed the issue of filtering for `axolotl` models on Hugging Face without getting results. A link was shared to [Hugging Face models](https://huggingface.co/models?other=axolotl), and solutions involving the `HfApi` library were proposed.

- **Pre-defined tags for filtering**: A Hugging Face team member clarified that the `Other` tab uses a set of pre-defined tags to avoid overwhelming users, making the user experience more consistent. They mentioned a potential improvement: showing "+N other tags" to make it clearer.

- **Energy over Hybrid Sharding with FSDP and DS**: A user expressed enthusiasm for hybrid sharding strategies when sharding models using **FSDP and DS**.

- **Uploading fine-tuned models**: A user had issues with uploading a large fine-tuned `gpt2-medium` model to Hugging Face, noting that it resulted in multiple `.pth` files instead of one. They were advised to seek help in a more relevant channel for detailed guidance.

**Link mentioned**: <a href="https://huggingface.co/models?other=axolotl)">Models - Hugging Face</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1242524086412771428)** (10 messages🔥): 

- **Clarification on Replicate's Use Case**: A member questioned the primary use case for **Replicate**, asking if it's mainly to offer API endpoints for downstream tasks and for firms/individuals. They also noted the availability of "defined tasks, fine-tuning, and customized datasets."

- **Conference Registration Email Issues**: Several members, including **hughdbrown** and **project_disaster**, reported issues with **conference registration** where the emails used for GitHub registration differ from those used for the conference.

- **Credits and Email Address Workaround**: **harpreetsahota** mentioned that users can set a different email address after signing up on Replicate if their GitHub email differs. However, **filippob82** indicated that emails containing a `+` sign are currently not being accepted.

- **Credit Allocation Enquiries**: Users like **digitalbeacon** are awaiting **credits** post-sign-up. **0xai** queried whether entering the maven registered address in the notifications section would automatically add these credits.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1242694370012954666)** (4 messages): 

- **Credit dispatch in progress**: A member asked if **credits had already been dispatched**. Another member responded, directing them to a pinned message and clarifying that announcements would be made on Discord and by email.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/1242699864765104179)** (4 messages): 

- **Hamel gets his own fan channel**: A member humorously acknowledges that Hamel has his own fan channel. The sentiment was light and playful, stating, *"Not sure what to do with such power."*.
- **Session preparation hints**: Another member hints that they'll fill the channel with relevant content before conducting a scheduled session. They plan to ensure engaging discussion leading up to the event.

**Link mentioned**: <a href="https://tenor.com/view/minion-hello-minions-gif-7623022">Minion Hello GIF - Minion Hello Minions - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1242485573386637455)** (525 messages🔥🔥🔥): 

- **NVLink woes and creative solutions**: Members discussed issues with **NVLink**, including mismatched card heights and lack of NVLink compatibility on certain setups. Suggested solutions included using riser cables with support brackets.

- **Hamel's evaluation steps clarification**: A user asked for the significance of the evaluation step that **Hamel** discussed, leading to an understanding that breaking down tasks and iterative iterations are key to completing projects efficiently. *"80% of the time is spent getting to 80% quality, and 500% of the time to reach 100%."*

- **Using Modal and Jarvis for running Axolotl**: Users discussed using **Modal**, **RunPod**, and **Jarvis Labs** for running Axolotl, with suggestions to initially try straightforward setups like RunPod or Jarvis before attempting more automated or complex configurations such as **Modal**. **"You can run it on modal if you have the credits"** and **"try Jarvis which offers credits as part of the course."**

- **Axolotl dataset formats and model usage**: The community explored various dataset formats for Axolotl, including JSONL and conversation-based formats like **ShareGPT**. There was a preference for JSONL due to its flexibility and ease of use, with an emphasis on using **the 'input_output' format for cases without strict templates.**

- **Recording workshops and resource links**: The community shared feedback on the need for more practical examples and clear steps to run fine-tuning workshops. Helpful links to resources and blog posts, such as [Loom video](https://www.loom.com/share/30d3b2e054f142fda5d905f95fedc29f) and [Medium post](https://medium.com/@andresckamilo/finetuning-llms-using-axolotl-and-jarvis-ai-c1d11fe3844c), were highlighted, and recordings were made available promptly.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/idefics2">Introducing Idefics2: A Powerful 8B Vision-Language Model for the community</a>: no description found</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca">parlance-labs/hc-mistral-alpaca · Hugging Face</a>: no description found</li><li><a href="https://x.com/abacaj/status/1782835550396850449">Tweet from anton (@abacaj)</a>: Phi-3 seems pretty good, an improvement over phi-2 for sure. The long context 128k seems very useful for extracting information and document processing given that the model is quite small it can be de...</li><li><a href="https://lu.ma/terrible-ai-systems?utm_source=llm">How to Build Terrible AI Systems with Jason Liu · Luma</a>: Jason is an independent consultant who uses his expertise in recommendation systems to help fast-growing startups build out their RAG applications. He was…</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: no description found</li><li><a href="https://www.ianww.com/llm-tools">LLM eval tools spreadsheet</a>: Spreadsheet of 50+ LLM evaluation tools for testing models and improving prompts.</li><li><a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">Installing the NVIDIA Container Toolkit &mdash; NVIDIA Container Toolkit 1.15.0 documentation</a>: no description found</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca/tree/main/data">parlance-labs/hc-mistral-alpaca at main</a>: no description found</li><li><a href="https://poe.com/s/c0BFLNhTwiyPXOulPCnO">you have a column with each element containing a list of tuple. get the frequency of the appearance of each tuple</a>: TrinoAgentEx: Which SQL keyword do you want to learn about? TrinoAgentEx: To query a frequency distribution of tuples within a list in a single Trino SQL query, you&#x27;ll have to perform several ope...</li><li><a href="https://www.loom.com/share/30d3b2e054f142fda5d905f95fedc29f">Exploring Fine-tuning with Honeycomb Example</a>: In this video, I walk you through the process of fine-tuning a model using the honeycomb example. I provide step-by-step instructions on cloning the repository, installing dependencies, and running th...</li><li><a href="https://x.com/HamelHusain/status/1784769559364608222">Tweet from Hamel Husain (@HamelHusain)</a>: Llama 3 70b function calling works pretty well out of the box with prompting only 🚀💰   See the below demo (prompt and code in next tweet)</li><li><a href="https://outlines-dev.github.io/outlines/">Outlines</a>: Structured text generation with LLMs</li><li><a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - Efficient finetuning of Llama 3 with FSDP QDoRA</a>: We’re releasing FSDP QDoRA, a scalable and memory-efficient method to close the gap between parameter efficient finetuning and full finetuning.</li><li><a href="https://github.com/ml-explore/mlx">GitHub - ml-explore/mlx: MLX: An array framework for Apple silicon</a>: MLX: An array framework for Apple silicon. Contribute to ml-explore/mlx development by creating an account on GitHub.</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - Template-free prompt construction</a>: no description found</li><li><a href="https://x.com/TheZachMueller/status/1696157965890339148">Tweet from Zach Mueller (@TheZachMueller)</a>: Excited to announce a new @huggingface space to help with one of machine learning&#39;s biggest questions:  How much space does {X} model take in vRAM? And most importantly: when using `device_map=&#3...</li><li><a href="https://github.com/parlance-labs/ftcourse/tree/master/sample_data">ftcourse/sample_data at master · parlance-labs/ftcourse</a>: Contribute to parlance-labs/ftcourse development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436">ERROR: No matching distribution found for bitsandbytes==0.43.0 for macOS  · Issue #1436 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior The command pip3 install -e &#39;.[flash-attn,deeps...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/mac.qmd">axolotl/docs/mac.qmd at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://www.loom.com/share/30d3b2e054f142fda5d905f95fedc29f?sid=7edb48da-722b-4c5f-9150-a49bdc19e4c5">Exploring Fine-tuning with Honeycomb Example</a>: In this video, I walk you through the process of fine-tuning a model using the honeycomb example. I provide step-by-step instructions on cloning the repository, installing dependencies, and running th...</li><li><a href="https://medium.com/@andresckamilo/finetuning-llms-usin">no title found</a>: no description found</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: Structured Text Generation</a>: Structured Text Generation. Contribute to outlines-dev/outlines development by creating an account on GitHub.</li><li><a href="https://x.com/danielhanchen">Tweet from undefined</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1609">Unsloth optims for Llama by winglian · Pull Request #1609 · OpenAccess-AI-Collective/axolotl</a>: WIP to integrate Unsloth&#39;s optimizations into axolotl. The manual autograd for MLP, QKV, O only seems to help VRAM by 1% as opposed to the reported 8%. The Cross Entropy Loss does help significant...</li><li><a href="https://github.com/stas00/ml-engineering/blob/master/training/instabilities/training-loss-patterns.md">ml-engineering/training/instabilities/training-loss-patterns.md at master · stas00/ml-engineering</a>: Machine Learning Engineering Open Book. Contribute to stas00/ml-engineering development by creating an account on GitHub.</li><li><a href="https://www.philschmid.de/instruction-tune-llama-2">Extended Guide: Instruction-tune Llama 2</a>: This blog post is an extended guide on instruction-tuning Llama 2 from Meta AI</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#cloud-gpu">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://nbsanity.com/static/d06085f1dacae8c9de9402f2d7428de2/demo.html">Llama-3 Function Calling Demo</a>: no description found</li><li><a href="https://github.com/modal-labs/llm-finetuning/">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://www.rungalileo.io/blog/mastering-rag-how-to-select-a-reranking-model">Mastering RAG: How to Select A Reranking Model - Galileo</a>: Choosing the best reranking model for your RAG-based QA system can be tricky. This blog post simplifies RAG reranking model selection, helping you pick the right one to optimize your system&#x27;s per...</li><li><a href="https://x.com/abacaj/status/1792991309751284123">Tweet from anton (@abacaj)</a>: False alarm on the phi-3 models (did very poorly on a few offline benchmarks I have), still using llama-3 fine tuned models for a few specialized services. The phi-3 models seem very sensitive to prom...</li><li><a href="https://x.com/sroecker/status/1757103619705299061?t=uajfu81xkUp7x80xgQ7i1A&s=19">Tweet from Steffen Röcker (@sroecker)</a>: Ever wondered how to fine-tune LLMs using @axolotl_ai and @Podman_io?  Follow the instructions for NVIDIA toolkit CDI and simply run &#34;podman run --rm --device http://nvidia.com/gpu=all --security-...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/908">Apply unsloth optimizations · Issue #908 · OpenAccess-AI-Collective/axolotl</a>: ⚠️ Please check that this feature request hasn&#39;t been suggested before. I searched previous Ideas in Discussions didn&#39;t find any similar feature requests. I searched previous Issues didn&#39;t...</li><li><a href="https://huggingface.co/spaces/muellerzr/llm-conf">LLM Conf talk - a Hugging Face Space by muellerzr</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/pretraining.html">Axolotl - Pre-training</a>: no description found</li><li><a href="https://x.com/simonw/status/1792692563776000338">Tweet from Simon Willison (@simonw)</a>: I particularly like this note:  &#34;Phi-3 models do not perform as well on factual knowledge benchmarks (such as TriviaQA) as the smaller model size results in less capacity to retain facts.&#34;  Go...</li><li><a href="https://github.com/hiyouga/LLaMA-Factory">GitHub - hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://www.reddit.com/u/danielhanchen">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/CalculatedContent/WeightWatcher">GitHub - CalculatedContent/WeightWatcher: The WeightWatcher tool for predicting the accuracy of   Deep Neural Networks</a>: The WeightWatcher tool for predicting the accuracy of   Deep Neural Networks - CalculatedContent/WeightWatcher</li><li><a href="https://github.com/shisa-ai/shisa-v2/wiki/Ablations">Ablations</a>: Japanese / English Bilingual LLM. Contribute to shisa-ai/shisa-v2 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/949">Evaluation took much more time when enable eval_table_size  · Issue #949 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior The evaluation time is expected to increase but not...</li><li><a href="https://github.com/parlance-labs/ftcourse/tree/master">GitHub - parlance-labs/ftcourse</a>: Contribute to parlance-labs/ftcourse development by creating an account on GitHub.</li><li><a href="https://www.malwarebytes.com/blog/news/2024/04/billions-of-scraped-discord-messages-up-for-sale">Billions of scraped Discord messages up for sale | Malwarebytes</a>: An internet scraping platform is offering access to a database filled with over four billion Discord messages and combined user profiles</li><li><a href="https://github.com/argilla-io/distilabel/blob/main/examples/structured_generation_with_outlines.py">distilabel/examples/structured_generation_with_outlines.py at main · argilla-io/distilabel</a>: ⚗️ distilabel is a framework for synthetic data and AI feedback for AI engineers that require high-quality outputs, full data ownership, and overall efficiency. - argilla-io/distilabel</li><li><a href="https://us06web.zoom.us/rec/share/M29p9cyVwM80QUxZCXJmL1_E56IeznMpj2mrmqMaeL7B7rDrR6IFARgeXOpWM9Qu.p8Mrj7osc2-3r-Dm">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://us06web.zoom.us/rec/share/obPk2t0iYZXhDiV4ZxCdhdlVJjZAnL-N7PNW2iEP9vord5LEsDraCk86Xz1bMSWv.OK7WNsH1DntOJqbr?startTime=1716319486000">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://github.com/huggingface/trl">GitHub - huggingface/trl: Train transformer language models with reinforcement learning.</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>: 🤗 AutoTrain Advanced. Contribute to huggingface/autotrain-advanced development by creating an account on GitHub.</li><li><a href="https://github.com/artidoro/qlora">GitHub - artidoro/qlora: QLoRA: Efficient Finetuning of Quantized LLMs</a>: QLoRA: Efficient Finetuning of Quantized LLMs. Contribute to artidoro/qlora development by creating an account on GitHub.</li><li><a href="https://github.com/stanfordnlp/pyreft">GitHub - stanfordnlp/pyreft: ReFT: Representation Finetuning for Language Models</a>: ReFT: Representation Finetuning for Language Models - stanfordnlp/pyreft</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca/tree/main/configs">parlance-labs/hc-mistral-alpaca at main</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1242526610024824942)** (3 messages): 

- **Excitement for Jason's W&B course**: Filippob82 expressed enthusiasm for Jason's session and mentioned they are halfway through his W&B course. They used an emoji to convey their excitement.
- **Curiosity about prompt engineering**: Nehil8946 showed interest in Jason's work on optimizing prompts and asked if there is a systematic approach to prompt engineering that Jason follows. They are looking forward to learning about it in his workshop.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/)** (1 messages): 

nirant: Woohoo! Looking forward to <@660097403046723594>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1242489403129987194)** (2 messages): 

- **Meet Freddy, your Gradio expert**: Freddy introduced himself as one of the maintainers of **Gradio**, a Python library for developing user interfaces for AI models. He shared helpful resources for getting started and creating chatbots with Gradio, including a [quickstart guide](https://www.gradio.app/guides/quickstart) and a [tutorial on building a chatbot](https://www.gradio.app/guides/creating-a-chatbot-fast).
- **Mnemic1 prepares for questions**: A member expressed thanks for the resources and mentioned they would have questions about an **A1111-extension** they wrote, which had some unresolved issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.gradio.app/guides/quickstart">Quickstart</a>: A Step-by-Step Gradio Tutorial</li><li><a href="https://www.gradio.app/guides/creating-a-chatbot-fast">Creating A Chatbot Fast</a>: A Step-by-Step Gradio Tutorial
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1242543726312689705)** (85 messages🔥🔥): 

```html
- **Members address Axolotl issue #1436**: Discussion about `bitsandbytes==0.43.0` not installing on macOS from [GitHub Issue #1436](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436). Recommendations include using Linux GPU servers on RunPod.
- **Axolotl and MLX integration not yet supported**: Members discuss the lack of MLX support on Axolotl as detailed in [GitHub Issue #1119](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1119). Users are advised to stay updated.
- **Best setup practices explored**: Members share various methods to set up Axolotl. The Axolotl [Readme](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main?tab=readme-ov-file#quickstart-) and Docker method are mentioned as the most reliable.
- **Fine-tuning and integration concerns**: Members inquire about using Axolotl on local machines and fine-tuning models like LLaMA3. Issues related to configuration and compatibility with Modal environments are discussed.
- **Tips for troubleshooting installation**: For users facing installation difficulties, such as receiving a `CUDA` error, several members recommend steps including installing specific CUDA/PyTorch versions and using the docker container. Links to [Docker](https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018?context=explore) and a [setup guide](https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl) are provided.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/oaaic/fused-cel-llama3/runs/kkyhjjh6/files/tmp/axolotl_config_rdbefq2r.yml">oaaic</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md">mlx-examples/lora/README.md at main · ml-explore/mlx-examples</a>: Examples in the MLX framework. Contribute to ml-explore/mlx-examples development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436">ERROR: No matching distribution found for bitsandbytes==0.43.0 for macOS  · Issue #1436 · OpenAccess-AI-Collective/axolotl</a>: Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior The command pip3 install -e &#39;.[flash-attn,deeps...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/.github/workflows/tests.yml#L105-L107">axolotl/.github/workflows/tests.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/src/common.py#L14">llm-finetuning/src/common.py at main · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018?context=explore">Docker</a>: no description found</li><li><a href="https://jarvislabs.ai/templates/axolotl">Easily Finetune LLM with Axolotl | Jarvislabs</a>: Axolotl helps you to finetune LLM using techniques like lora, qlora and more. Edit the config file and start LLM training</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/cicd/Dockerfile.jinja">axolotl/cicd/Dockerfile.jinja at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://modal.com/docs/examples/llm-finetuning">Fine-tune an LLM in minutes (ft. Llama 2, CodeLlama, Mistral, etc.)</a>: Tired of prompt engineering? Fine-tuning helps you get more out of a pretrained LLM by adjusting the model weights to better fit a specific task. This operational guide will help you take a base model...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/tree/main?tab=readme-ov-file#quickstart-)">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1119">MLX Support · Issue #1119 · OpenAccess-AI-Collective/axolotl</a>: Hi, It would be great to have MLX support in Axolotl. MLX has been shown to be able to quickly and efficiently finetune many LLMs, including 7B LLMs on consumer hardware. Thank you! (edit: update)</li><li><a href="https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl">Latent Space</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1242565467562967152)** (49 messages🔥): 

- **Hugging Face Presentation and Accelerate Resources**: A member shared various resources including a [presentation on Hugging Face](https://huggingface.co/spaces/muellerzr/llm-conf) and [documentation for Accelerate](https://huggingface.co/docs/accelerate). Links included tutorials on FSDP vs. DeepSpeed and examples on GitHub.
- **Creating Slides with Quarto Saves Time**: Members discussed how using [Quarto](https://huggingface.co/spaces/muellerzr/llm-conf/blob/main/llm_conf.qmd) made creating presentations easier and faster. One user mentioned they now only use Quarto for slides due to the streamlined workflow.
- **Using Accelerate in Python Scripts**: There was a conversation on how to utilize Accelerate within Python scripts, suggesting code snippets for launching processes and saving models with Accelerate. One user provided a detailed answer to streamline implementation.
- **Interest in Different Demo Videos for Accelerate**: Members expressed interest in seeing recorded demos of Accelerate's usage in various scenarios, including local vs. cloud training, hybrid modes, and focusing on techniques like LoRa without quantization. Specific requests included comparing setups and configurations for different environments.
- **Upcoming GPU Optimization Workshop**: An event was shared featuring a workshop on GPU optimization with speakers from OpenAI, NVIDIA, Meta, and Voltron Data, with details on [event registration](https://lu.ma/1wu5ppl5), [YouTube livestream](https://discord.gg/T5sx2MYd5R), and relevant reading materials.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/muellerzr/llm-conf/">LLM Conf talk - a Hugging Face Space by muellerzr</a>: no description found</li><li><a href="https://lu.ma/1wu5ppl5">GPU Optimization Workshop · Luma</a>: We’re hosting a workshop on GPU optimization with stellar speakers from OpenAI, NVIDIA, Meta, and Voltron Data. The event will be livestreamed on YouTube, and…</li><li><a href="https://drchrislevy.github.io/posts/llm_lunch_talk/llm_talk_slides.html#/title-slide).">Chris Levy - Intro to LLMs</a>: no description found</li><li><a href="https://huggingface.co/spaces/muellerzr/llm-conf">LLM Conf talk - a Hugging Face Space by muellerzr</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate">Accelerate</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate/quicktour">Quicktour</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed">Moving between FSDP And DeepSpeed</a>: no description found</li><li><a href="https://github.com/huggingface/accelerate/tree/main/examples">accelerate/examples at main · huggingface/accelerate</a>: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....</li><li><a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - a Hugging Face Space by hf-accelerate</a>: no description found</li><li><a href="https://huggingface.co/spaces/Vokturz/can-it-run-llm">Can You Run It? LLM version - a Hugging Face Space by Vokturz</a>: no description found</li><li><a href="https://huggingface.co/spaces/cllatMTK/TransformerAnalyzer">TransformerAnalyzer - a Hugging Face Space by cllatMTK</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1242805657153966112)** (30 messages🔥): 

- **Caching Precautions for Multiple Model Training**: A user asked about the necessary precautions for separating cached samples when training multiple models simultaneously. They inquired whether sequence length, datasets, tokenizers, and other settings are relevant factors.

- **Custom Callbacks for Evaluations**: A user sought guidance on using custom callbacks to run evaluations on custom datasets during training and transferring checkpoints between devices while displaying outputs in wandb/mlflow.

- **Dataset Types: Pretrain vs. Completion**: A user asked for the difference between "pretrain" and "completion" dataset types and the appropriate use cases for each.

- **Solving Command Errors**: Several users discussed unresolved issues with running the command `accelerate launch -m axolotl.cli.train hc.yml`. Troubleshooting suggestions included ensuring dependencies like `torch` and `gcc` are correctly installed, and using a docker image for a more straightforward setup.

- **Helpful GCC Installation Resource**: A user provided a [link to a tutorial](https://www.namehero.com/blog/how-to-install-gcc-on-ubuntu/#3-installing-gcc-compiler-on-ubuntu) for installing the GCC compiler on Ubuntu to help resolve installation issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.namehero.com/blog/how-to-install-gcc-on-ubuntu/#3-installing-gcc-compiler-on-ubuntu">How To Install GCC On Ubuntu</a>: Let&#039;s walk through the process of installing GCC on your Ubuntu system, making the world of compilers and development tools accessible!</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1632]">Issues · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1242522009758470174)** (1 messages): 

- **Perplexity integrates Tako for enhanced knowledge search**: Perplexity teams up with **Tako** to provide advanced knowledge search and visualization. Users can now search for comparative data like “Gamestop vs. AMC stock since 5/3/24” with [interactive knowledge cards](https://trytako.com/blog/introducing-tako-and-perplexity-integration), initially available in the U.S. and in English, with mobile access coming soon.

**Link mentioned**: <a href="https://trytako.com/blog/introducing-tako-and-perplexity-integration">Tako</a>: no description found

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1242385165335003147)** (835 messages🔥🔥🔥): 

```html
- **Microsoft Stole OpenAI's Ideas**: A member shared a [blog post](https://blogs.microsoft.com/blog/2024/05/20/introducing-copilot-pcs/) stating that Microsoft has copied features from OpenAI and introduced "Copilot+ PCs,” the fastest and most intelligent Windows PCs ever built. They noted features like an impressive 40+ TOPS, all-day battery life, AI image generation, and live captions for 40+ languages.

- **GPT-4o Context Concerns**: There were discussions about the context window of GPT-4o as **perceived on Perplexity**. A consensus formed that **context window defaults to 32k**, with uncertainties about higher capacities.

- **Perplexity's Default Model Surprise**: Members expressed surprise that the default model for Perplexity might be **Haiku** instead of an in-house model, **Sonar**, which is available only for pro users. One member noted that free users previously used GPT-3.5, but this has changed recently.

- **Perplexity's API Queries**: Discussion revolved around how Perplexity configures and charges for API usage. Members speculated about using in-house models and the potential financial implications of their pricing structure.

- **Service Downtime Creates Community Stir**: Perplexity experiencing downtime led to widespread frustration and speculation among users about the cause. Users shared alternative resources and a member posted a supportive message to help calm the community during the outage.
```

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/angry-panda-smash-pc-computer-mad-gif-16248458">Angry Panda Smash GIF - Angry Panda Smash Pc - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/damn-it-gif-24870550">Damn It GIF - Damn It - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://sdk.vercel.ai/">Vercel AI SDK</a>: Build AI-powered applications with the latest AI language models</li><li><a href="https://tenor.com/view/countdown-final-gif-13775423">Countdown Final GIF - Countdown Final - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://marp.app/">Marp: Markdown Presentation Ecosystem</a>: Marp (also known as the Markdown Presentation Ecosystem) provides an intuitive experience for creating beautiful slide decks. You only have to focus on writing your story in a Markdown document.</li><li><a href="https://x.com/AravSrinivas/status/1793298035373691060">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Perplexity is back up. App and site should function as normal. Apologize for the inconvenience and we are working on making sure infra is resilient.</li><li><a href="https://blogs.microsoft.com/blog/2024/05/20/introducing-copilot-pcs/">Introducing Copilot+ PCs - The Official Microsoft Blog</a>: An on-demand recording of our May 20 event is available. Today, at a special event on our new Microsoft campus, we introduced the world to a new category of Windows PCs designed for AI, Copilot+ PCs. ...</li><li><a href="https://chromewebstore.google.com/detail/stylus/clngdbkpkpeebahjckkjfobafhncgmne">Stylus</a>: Redesign the web with Stylus, a user styles manager. Stylus allows you to easily install themes and skins for many popular sites.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1242449708941705286)** (9 messages🔥): 

- **Members share Perplexity AI links**: Multiple members shared specific search-related links from Perplexity AI, indicating queries and interests such as "Layer", "indoor discussions", and "creating SFW content". One particularly notable search was about "Ether is" with a specific focus link.

- **Reminder to make threads shareable**: A gentle reminder was issued to ensure that shared threads are marked as "Shareable". The comment included a [screenshot from Discord](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).

- **User interest in Taiwan Semiconductor**: A member showed interest in Taiwan Semiconductor, sharing a specific [Perplexity AI search link](https://www.perplexity.ai/search/Taiwan-Semiconductor-remote-k.5AQq3LQkGX5eg4Nbh9jA).
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1242636767538970654)** (11 messages🔥): 

- **Headlines API delivers outdated news**: A member reported getting headlines from a year ago when using the same prompt for the API as on the web. They asked if anyone else had similar issues generating relevant daily headlines.
- **Attempted to refine search queries**: Another member suggested adding a date filter (`after:12-02-2024`). They further clarified that this should be added directly to the query.
- **API underperforms compared to the web version**: The original member reported that the suggested fixes did not work, as they continued to get poor results through the API compared to the web. They mentioned they were getting good results on the web but terrible ones through the API.
- **API limitations highlighted**: It was noted that the API is still in beta and only supports one endpoint. This limitation may be contributing to the inconsistent results between the web and API outputs.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1242374862094471188)** (497 messages🔥🔥🔥): 

- **Lightning and Hyper models debate**: *A member discussed the efficiency of mixing Lightning and Hyper models with base stable models, proposing it could reduce the number of steps required for image generation*. However, another member advised against mixing checkpoints from different architectures, warning it often results in poor-quality images.

- **EU AI Act sparks outrage**: Following the approval of the EU AI Act, several members expressed frustration and confusion about its implications. One shared a [link to the official press release](https://www.consilium.europa.eu/en/press/press-releases/2024/05/21/artificial-intelligence-ai-act-council-gives-final-green-light-to-the-first-worldwide-rules-on-ai/), highlighting the potential difficulties related to watermarking requirements for AI-generated content.

- **Frustrations with Local AI Setup**: Members frequently discussed the challenges of setting up Stable Diffusion locally, particularly with AMD GPUs, while suggesting Nvidia GPUs as a better alternative. One member humorously noted that the "best wizard" would help them acquire a Nvidia GPU to solve their issues.

- **Discontent with AI content quality**: The rampant creation of low-quality AI-generated images, particularly generic and heavily sexualized content, was criticized. Members pointed out the prevalence of such content on platforms like CivitAI and the AI art subreddit, questioning the value it adds to the community.

- **GPUs for Stable Diffusion**: Members debated the best GPUs for running Stable Diffusion, with a preference for Nvidia GPUs over AMD due to better support. They emphasized the importance of VRAM, recommending at least 12GB for efficient AI performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://glif.app/@Oliveira/glifs/clw44qfbl0000m0zztwqk2tnf">glif - StableDiffusion 3 + GPT4 Helper + SDXL 1.5x Upscale (CopyGenius) by Yuri Oliveira COPYGENIUS </a>: no description found</li><li><a href="https://www.asus.com/content/asus-ai-pc/">Next Level. AI Incredible | ASUS Launch Event | ASUS</a>: We are thrilled to be unveiling our latest product, packed with new AI experiences. Mark your calendars for May 20th at 11:00 AM (PT) and join our livestream.</li><li><a href="https://tenor.com/view/welcome-gif-26939290">Welcome GIF - Welcome - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/alvin-and-the-chipmunks-alvin-whoops-my-bad-oops-gif-15512287650458333097">Alvin And The Chipmunks Alvin GIF - Alvin And The Chipmunks Alvin Whoops - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://civitai.com/">Civitai: The Home of Open-Source Generative AI</a>: Explore thousands of high-quality Stable Diffusion models, share your AI-generated art, and engage with a vibrant community of creators</li><li><a href="https://www.jammable.com/custom-asmr-woman">AI ASMR Woman Voice Generator | Jammable AI Cover Generator</a>: Create AI ASMR Woman covers as seen on TikTok and YouTube in seconds! Jammable has thousands of community uploaded AI voice models available for creative use now!
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1242464493653463142)** (273 messages🔥🔥): 

- **Benchmarking Pallas, Naive, and Flash v2 in JAX**: Users discussed benchmarking various implementations like pallas, naive, and flash v2 in JAX, comparing performance on different input sizes. Issues encountered include discrepancies in TFLOPS and shared memory errors on GPUs.

- **PSA on California SB 1047**: A heated discussion on SB 1047, a California bill that could severely impact open-source AI by creating an unaccountable agency, was shared. Members were encouraged to contact legislators to voice their opposition.

- **Concerns Over GPU Clocks During Benchmarks**: There was a detailed conversation about GPU clock speeds affecting benchmark results, with recommendations to use MSI Afterburner to lock clocks. A member noted, "Creating the input is slow," impacting the benchmarking process.

- **Review of Frontier Model Training Costs**: A member from Epoch discussed the cost estimation for training large AI models, noting discrepancies in reported costs from various sources. They thanked Eleuther for insights, revealing that the Pythia model had an estimated training cost of $250k per run on AWS.

- **Discussion on Preprints**: Members debated the pros and cons of making preprints available on ArXiv, citing that preprints are becoming more accepted across major journals. "Almost all the big journals have normalized it," one user noted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rentry.co/kovugk6t">import math</a>: import time from typing import Optional, Tuple import jax import jax.numpy as jnp from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention @jax.jit def scaled_dot_product_attention(...</li><li><a href="https://www.thonking.ai/p/strangely-matrix-multiplications">Strangely, Matrix Multiplications on GPUs Run Faster When Given &quot;Predictable&quot; Data! [short]</a>: Great minds discuss flops per watt.</li><li><a href="https://affuture.org/post/9-context/">Call-To-Action on SB 1047</a>: California legislators, under the influence of Effective Altruism activists, are trying to sneak through a disastrous bill for open-source AI and the technology industry generally. SB 1047 creates an ...</li><li><a href="https://github.com/pytorch/torchtitan/issues/341">Modify FLOPs in MFU calculation for casual mask when using FlashAttention. · Issue #341 · pytorch/torchtitan</a>: Hi, I suggest we modify the FLOPs calculation in the MFU according to the FlashAttention benchmark script. Specifically, the current calculation for the casual mask can exceed 100% MFU for seq_len ...</li><li><a href="https://manifold.markets/ZviMowshowitz/will-california-bill-sb-1047-become?r=Q2hhcmxlc0Zvc3Rlcg">Will California AI regulation bill SB 1047 become law this session?</a>: 71% chance. California Senator Scott Weiner of SF has introduced the bill (https://twitter.com/Scott_Wiener/status/1755650108287578585, https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bil...</li><li><a href="https://rentry.co/pgz5er7u">import math</a>: import time from typing import Optional, Tuple import jax import jax.numpy as jnp from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention @jax.jit def scaled_dot_product_attention(...</li><li><a href="https://www.wolframalpha.com/input?i=%286+FLOP+*+299892736000+*+12+billion%29+%2F+%28312+TFLOPS+*+72300+hours%29">(6 FLOP * 299892736000 * 12 billion) / (312 TFLOPS * 72300 hours) - Wolfram|Alpha</a>: Wolfram|Alpha brings expert-level knowledge and capabilities to the broadest possible range of people—spanning all professions and education levels.</li><li><a href="https://x.com/AISafetyInst/status/1793163082379968955">Tweet from AI Safety Institute (@AISafetyInst)</a>: We are announcing new grants for research into systemic AI safety.  Initially backed by up to £8.5 million, this program will fund researchers to advance the science underpinning AI safety.  Read more...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1242477118407839814)** (128 messages🔥🔥): 

- **Paper on GPT-3 Non-Deterministic Temperature 0 Behavior**: Members discussed how GPT-3 can exhibit random output even at **temperature 0**, with references provided including [this paper](https://arxiv.org/abs/2210.14986) and an OpenAI community [discussion thread](https://community.openai.com/t/run-same-query-many-times-different-results/140588). Another member mentioned hardware factors, such as CUDA kernel non-determinism, contributing to this behavior.
- **MegaBlocks for Efficient MoE Training**: The introduction of MegaBlocks, a system for efficient Mixture-of-Experts (MoE) training on GPUs was discussed, which avoids token dropping and offers significant speedups. The [research paper](https://arxiv.org/abs/2211.15841) details its contributions, like block-sparse operations for improved hardware efficiency.
- **Character Self-Awareness in Language Models**: Users shared insights on how larger language models manage self-aware characters effectively, integrating concepts like understanding when they're edited or rolled-back in conversations. These observations seem consistent across various large models, including proprietary ones and open-source adaptations.
- **Transformer Model Efficiency Improvements**: Various optimization techniques for transformer models were debated, such as LeanAttention and Multi-Query Attention (MQA), which aim to reduce the memory footprint and latency of large language models. Relevant papers include [Cross-Layer Attention (CLA)](https://arxiv.org/abs/2405.12981) and [LeanAttention methods](https://arxiv.org/abs/2405.10480) for improved computational efficiency.
- **Scaling Laws and Model Performance**: Intrinsic performance and scaling laws for reinforcement learning models were discussed, emphasizing the smooth performance scaling similar to generative models. The concept was illustrated through [a recent paper](https://arxiv.org/abs/2301.13442) that models intrinsic performance as a power law in context to environment interactions and training compute.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.05526">Buffer Overflow in Mixture of Experts</a>: Mixture of Experts (MoE) has become a key ingredient for scaling large foundation models while keeping inference costs steady. We show that expert routing strategies that have cross-batch dependencies...</li><li><a href="https://arxiv.org/abs/2405.12981">Reducing Transformer Key-Value Cache Size with Cross-Layer Attention</a>: Key-value (KV) caching plays an essential role in accelerating decoding for transformer-based autoregressive large language models (LLMs). However, the amount of memory required to store the KV cache ...</li><li><a href="https://arxiv.org/abs/2301.13442">Scaling laws for single-agent reinforcement learning</a>: Recent work has shown that, in generative modeling, cross-entropy loss improves smoothly with model size and training compute, following a power law plus constant scaling law. One challenge in extendi...</li><li><a href="https://arxiv.org/abs/2203.17207">A Proof of the Kahn-Kalai Conjecture</a>: Proving the ``expectation-threshold&#39;&#39; conjecture of Kahn and Kalai, we show that for any increasing property $\mathcal{F}$ on a finite set $X$, $$p_c(\mathcal{F})=O(q(\mathcal{F})\log \ell(\ma...</li><li><a href="https://arxiv.org/abs/2405.10480">Lean Attention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers</a>: Transformer-based models have emerged as one of the most widely used architectures for natural language processing, natural language generation, and image generation. The size of the state-of-the-art ...</li><li><a href="https://arxiv.org/abs/2405.11582">SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization</a>: Transformers have become foundational architectures for both natural language and computer vision tasks. However, the high computational cost makes it quite challenging to deploy on resource-constrain...</li><li><a href="https://arxiv.org/abs/2405.10986">Benchmark Early and Red Team Often: A Framework for Assessing and Managing Dual-Use Hazards of AI Foundation Models</a>: A concern about cutting-edge or &#34;frontier&#34; AI foundation models is that an adversary may use the models for preparing chemical, biological, radiological, nuclear, (CBRN), cyber, or other attac...</li><li><a href="https://arxiv.org/abs/2211.15841">MegaBlocks: Efficient Sparse Training with Mixture-of-Experts</a>: We present MegaBlocks, a system for efficient Mixture-of-Experts (MoE) training on GPUs. Our system is motivated by the limitations of current frameworks, which restrict the dynamic routing in MoE lay...</li><li><a href="https://x.com/arankomatsuzaki/status/1792386318300749848">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Layer-Condensed KV Cache for Efficient Inference of Large Language Models  Achieves up to 26× higher throughput than standard transformers and competitive performance in language modeling and downstre...</li><li><a href="https://community.openai.com/t/run-same-query-many-times-different-results/140588">Run same query many times - different results</a>: I wonder if anyone knows why we get different results when running the same prompt multiple times in a row.  I have noticed in quite a lot of my experiments that if you set a cool-down time in between...</li><li><a href="https://152334h.github.io/blog/non-determinism-in-gpt-4/">Non-determinism in GPT-4 is caused by Sparse MoE</a>: It&rsquo;s well-known at this point that GPT-4/GPT-3.5-turbo is non-deterministic, even at temperature=0.0. This is an odd behavior if you&rsquo;re used to dense decoder-only models, where temp=0 shou...</li><li><a href="https://rmarcus.info/blog/2018/09/14/consistent-hashing-overflow.html">
      
      Overflow in consistent hashing &middot; Ryan Marcus
      
    </a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1242428358260949022)** (1 messages): 

- **Training on small datasets remains challenging**: A member commented that training AI on a **small dataset** yields worse results compared to pre-training on the entire internet before fine-tuning on smaller datasets. They added that it is "notoriously difficult" to close this gap.
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1242533689271779329)** (4 messages): 

- **Anthropic's Work on Interpretable Features Creates Buzz**: A member shared a link about exciting work on interpretable features by Anthropic. You can read more about it [here](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- **Reconstruction Loss in SAEs Raises Concerns**: A member asked, "How big of an issue is reconstruction loss for SAEs?" and followed up with inquiries on what improvements are being pursued.
- **Pointer to Related Channel**: Another member directed others to check channel **#1153431135414669422** for related discussions.
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1242467551230038097)** (37 messages🔥): 

```html
- **Questions on lm-evaluation-harness and MCQs**: Members discussed the randomization of answer choices in MCQs using **lm-eval-harness**, with concerns about benchmark biases towards early choices. While **SciQ** has a fixed correct answer index, the randomization isn't currently applied for **MMLU**.
  
- **Upcoming Submissions and Papers**: An **anon'd paper** is coming soon to arXiv, while members joked about **not needing to worry about insane competition** in D&B papers. There's also work on an updated version of the **Pile with 3T tokens and fully licensed text**.

- **Medical Benchmarks Controversy**: A lively discussion emerged about medical benchmarks and their potential dangers. One member focused on how these benchmarks might claim models are better and safer than physicians, highlighting ongoing improvements in the interpretation of such benchmarks.

- **Huggingface Dataset Configuration**: Members sought advice on configuring a Huggingface dataset's directory structure. The solution pointed out the importance of **adding a config in the README.md file** as outlined in the [Huggingface documentation](https://huggingface.co/docs/hub/en/datasets-manual-configuration#splits).

- **Running lm-eval-harness on Multi-node Slurm Cluster**: A question was raised about evaluating big models on a multi-node Slurm cluster. Attempts have been made using **vllm + ray** and **accelerate** but were unsuccessful, indicating a need for better solutions.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/hub/en/datasets-manual-configuration#splits">Manual Configuration</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/1710b42d52d0f327cb0eb3cb1bfbbeca992836ca/lm_eval/tasks/sciq/sciq.yaml#L11">lm-evaluation-harness/lm_eval/tasks/sciq/sciq.yaml at 1710b42d52d0f327cb0eb3cb1bfbbeca992836ca · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1242544374726791270)** (1 messages): 

- **Phi-3 Models Roll Out**: Microsoft released Phi-3 small and medium models, including Instruct Versions with up to 128k context and a VLM version. Check out the [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) model.

- **ZeroGPU Initiative Fuels Open-Source AI**: Hugging Face committed [\$10M via ZeroGPU](https://x.com/ClementDelangue/status/1791115403734778185) to support indie and academic AI builders with free GPU resources for AI demos. Over 1,300 ZeroGPU spaces have been built since May 1, 2024.

- **Local Apps Integration**: Hugging Face announced [Local Apps](https://x.com/LysandreJik/status/1792923587340390733), allowing users to easily convert model pages to local applications. Users can suggest their favorite local apps for integration.

- **Transformers 4.41.0 Packed with Updates**: The new release includes models like Phi3 and VideoLlava, improved GGUF support, and watermarking capabilities. [Transformers 4.41.0](https://github.com/huggingface/transformers/releases/tag/v4.41.0) is poised to enhance multiple functionalities, making integration smoother.

- **LangChain-HuggingFace Connector Released**: A new open-source package, [langchain-huggingface](https://huggingface.co/blog/langchain), integrates Hugging Face models into LangChain, offering flexible access to models via API and self-hosted inference. This facilitates easy installation and fast integration for various model use cases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ClementDelangue/status/1791115403734778185)">Tweet from clem 🤗 (@ClementDelangue)</a>: GPU-Poor no more: super excited to officially release ZeroGPU in beta today. Congrats @victormustar & team for the release!  In the past few months, the open-source AI community has been thriving. Not...</li><li><a href="https://x.com/LysandreJik/status/1792923587340390733)">Tweet from Lysandre (@LysandreJik)</a>: From a model page to your Local App in seconds, the @huggingface  Hub welcomes Local Apps!  Suggest your favorite Local App leveraging the Hub there to get them added to the dropdown and ✨ deep linked...</li><li><a href="https://x.com/osanseviero/status/1792904237153722569)">Tweet from Omar Sanseviero (@osanseviero)</a>: Transformers 4.41.0 has lots of goodies🤗  🥳 New models: Phi3, JetMoE, PaliGemma, VideoLlava, and Falcon 2. 🤯 GGUF support with from_pretrained 🤏 New quant methods: HQQ and EETQ 🔍 Watermarking sup...</li><li><a href="https://x.com/_philschmid/status/1790419788931416466)">Tweet from Philipp Schmid (@_philschmid)</a>: We are excited to announce huggingface-langchain🚀 A new open-source package to seamlessly integrate the latest open Models from @huggingface into @LangChainAI, supporting local models hosted models! ...</li><li><a href="https://x.com/multimodalart/status/1791201296357142663)">Tweet from apolinario (multimodal.art) (@multimodalart)</a>: Quite excited that CommonCanvas is JUST out! 🖼️  • First open source text-to-image models trained fully on openly licensed images (SD2 and SDXL architectures)  • The dataset, with ~70M openly license...</li><li><a href="https://x.com/xenovacom/status/1791436796498174047)">Tweet from Xenova (@xenovacom)</a>: Moondream, your favorite tiny vision language model by @vikhyatk can now run directly in the browser on WebGPU! 🤯 Powered, of course, by Transformers.js and ONNX Runtime Web! 🤗  Local inference mean...</li><li><a href="https://x.com/xenovacom/status/1792570966272336074)">Tweet from Xenova (@xenovacom)</a>: You can now use 🤗 Transformers.js with Google Visual Blocks, a visual programming framework that lets you create machine learning pipelines in a no-code graph editor!  🛠️ Rapid workflow prototyping ...</li><li><a href="https://x.com/IlysMoutawwakil/status/1791406503112704455)">Tweet from Ilyas Moutawwakil (@IlysMoutawwakil)</a>: Optimum-Benchmark on PyPI 🎉 But why now ? 🤔 Because it&#39;s getting integrated in Transformers&#39; benchmarking workflow 😍 Your favorite transformers will only get faster and lighter ; Kudos to @...</li><li><a href="https://x.com/osanseviero/status/1791567896482635801)">Tweet from Omar Sanseviero (@osanseviero)</a>: Curious about LLMs? Join this Fine-Tuning course with top experts! 🚀  @huggingface is offering $501.42 in GPU credits for can Space demos, fine-tuning, inference, and more! Enjoy 🤗  https://maven.co...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1242382910263132170)** (398 messages🔥🔥): 

- **Library for Training NeRF Models Discussed**: A member inquired about HuggingFace support for **NeRF** and **3D Gaussian Splatting** models, suggesting that a dedicated library could be beneficial. They were redirected to relevant channels for further discussion.
- **Concerns About Falcon-180B Fine-Tuning**: There were discussions about the challenges of fine-tuning **Falcon-180B** due to hardware limitations, even on **AutoTrain** with an **8xH100** setup. No concrete solution was provided, indicating the need for more advanced resources or methods.
- **Embedding Issues with 4-bit Quantized Llama-8B**: Members discussed unexpected memory usage when loading **Llama-8B** with 4-bit quantization. It was highlighted that **bitsandbytes 4-bit** doesn’t quantize embeddings, leading to higher-than-expected memory usage.
- **GPT Deployment on Personal Websites**: A user queried about integrating HuggingFace dataset views on personal websites. It was pointed out that while API integrations are possible, replicating the original viewer's look might not be feasible currently.
- **Concerns Over CA AI Law**: There were discussions regarding a controversial AI regulation law in California, which some users felt would benefit large incumbents like OpenAI and Google while potentially stifling startups. [NousResearch's Discord server](https://discord.gg/jqVphNsB4H) was mentioned as a place of further discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/jqVphNsB4H).">Discord | Your Place to Talk and Hang Out</a>: Discord is the easiest way to talk over voice, video, and text. Talk, chat, hang out, and stay close with your friends and communities.</li><li><a href="https://x.com/Scott_Wiener/status/1792572175116816853">Tweet from Senator Scott Wiener (@Scott_Wiener)</a>: In recent weeks, there&#39;s been a flurry of discussion online about SB 1047, my bill on responsible development of the largest & most powerful AI frontier models. We’ve heard some incredibly thought...</li><li><a href="https://huggingface.co/docs/inference-endpoints/guides/create_endpoint">Create an Endpoint</a>: no description found</li><li><a href="https://x.com/kuldeep_s_s/status/1792296168111628717">Tweet from Kuldeep Singh Sidhu (@kuldeep_s_s)</a>: You are happy that @Meta has open-sourced Llama 3 😃... So you jump on HuggingFace Hub to download the new shiny Llama 3 model only to see a few quintillion Llama 3&#39;s! 🦙✨  Which one should you us...</li><li><a href="https://huggingface.co/spaces/pyp1/VoiceCraft_gradio">VoiceCraft - a Hugging Face Space by pyp1</a>: no description found</li><li><a href="https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html">TransformerDecoder &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://llmpare.vercel.app/">llmpare</a>: no description found</li><li><a href="https://huggingface.co/papers/2405.10725">Paper page - INDUS: Effective and Efficient Language Models for Scientific
  Applications</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/v4.15.0/en/parallelism">Model Parallelism</a>: no description found</li><li><a href="https://x.com/osanseviero/status/1793018964479463781">Tweet from Omar Sanseviero (@osanseviero)</a>: I&#39;m GPU Poor. What about you?  https://huggingface.co/settings/local-apps</li><li><a href="https://x.com/julien_c/status/1745091045338066951">Tweet from Julien Chaumond (@julien_c)</a>: FINALLY got my GPU Poor hat from @fal_ai_data team @burkaygur @gorkemyurt   I’ll be rocking it through 2024  You rock! 🔥</li><li><a href="https://youtu.be/X5WVZ0NMaTg">How to Download (wget) Models from CivitAI &amp; Hugging Face (HF) &amp; upload into HF including privates</a>: If you are having trouble to download models onto your cloud platforms such as RunPod, Google Colab, Kaggle, Massed Compute or anywhere, this tutorial is mad...</li><li><a href="https://github.com/bigscience-workshop/petals">GitHub - bigscience-workshop/petals: 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading</a>: 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading - bigscience-workshop/petals</li><li><a href="https://x.com/Scott_Wiener/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://en.wikipedia.org/wiki/PlayStation_3_cluster">PlayStation 3 cluster - Wikipedia</a>: no description found</li><li><a href="https://health.petals.dev/">Petals Health Monitor</a>: no description found</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/H-D-T/Buzz">H-D-T/Buzz · Datasets at Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/xmooney-computer-developing-developer-coding-gif-25301200">Xmooney Computer GIF - Xmooney Computer Developing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://g.co/gemini/share/e87d6497e439">‎Gemini - CA Regulates Advanced AI</a>: Created with Gemini Advanced
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1242468051975143434)** (3 messages): 

- **Adding ImageBind to Transformers**: A user shared that they are **working on integrating ImageBind** into the `transformers` library. This suggests ongoing enhancements to the versatile library.
- **Training Huggy Agent**: A member mentioned they finished pushing a newly trained Huggy agent but is still in the process of *learning how everything works*. They have completed Unit 1 and are continuing their educational journey.
- **Looking for Project Collaborators**: Another user openly asked if **anyone wanted to connect to work on projects together**. This suggests a collaborative spirit and willingness to engage in community-driven projects.
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1242387232460111964)** (7 messages): 

- **Explore the Latest in 3D Gaussian Splatting**: Members discussed a [GitHub repository on 3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting), which lists the latest papers and resources. One member noted its potential in robotics and embodied AI, suggesting the next steps involve incorporating LLM reasoning for autonomous robot actions.
  
- **Boost Evaluation with Evaluator Classes**: A member shared a [link to the `Evaluator` classes documentation](https://huggingface.co/docs/evaluate/base_evaluator#evaluate-models-on-the-hub) highlighting how it simplifies the evaluation process for models, datasets, and metrics. Another member confirmed the utility, stating it "can save up lots of hustle" by eliminating the need to create metrics from scratch.
  
- **Automate Your Tweets from Wiki Articles**: A script that periodically scrapes and posts content from wiki articles to Twitter was shared by a member. The script is available in this [GitHub repository](https://github.com/anthonyrussano/wikitweet/blob/main/tweet-natural-healing-thread.py).

- **TransAgents Revolutionizes Literary Translation**: A multi-agent framework called TransAgents, using large language models for literary translation, was introduced with promising results. The [paper detailing the framework](https://arxiv.org/abs/2405.11804) reports that outputs from TransAgents are preferred by human readers.
  
- **Request for Guidance on News Classification Project**: A member sought assistance on a machine learning project aimed at classifying news articles into cargo-related and non-cargo-related categories. They explicitly mentioned being new to machine learning and looking for effective starting points.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.11804">(Perhaps) Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts</a>: Recent advancements in machine translation (MT) have significantly enhanced translation quality across various domains. However, the translation of literary texts remains a formidable challenge due to...</li><li><a href="https://huggingface.co/docs/evaluate/base_evaluator#evaluate-models-on-the-hub">Using the `evaluator`</a>: no description found</li><li><a href="https://github.com/anthonyrussano/wikitweet/blob/main/tweet-natural-healing-thread.py">wikitweet/tweet-natural-healing-thread.py at main · anthonyrussano/wikitweet</a>: Contribute to anthonyrussano/wikitweet development by creating an account on GitHub.</li><li><a href="https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#editing">GitHub - MrNeRF/awesome-3D-gaussian-splatting: Curated list of papers and resources focused on 3D Gaussian Splatting, intended to keep pace with the anticipated surge of research in the coming months.</a>: Curated list of papers and resources focused on 3D Gaussian Splatting, intended to keep pace with the anticipated surge of research in the coming months. - MrNeRF/awesome-3D-gaussian-splatting
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1242386505352351744)** (13 messages🔥): 

- **Markdown Note Taking App Goes Public**: A member introduced a personal markdown note-taking app, [Notie](https://github.com/branyang02/notie), urging contributions from the community. They also provided a [live preview](https://notie-nine.vercel.app/).

- **Dockerized Wiki with Hexo.js**: A member showcased a static wiki created with Hexo.js that supports over 1,000 articles and can be run using Docker. Contributions are welcome on their [GitHub page](https://github.com/wikip-co/wikip.co).

- **Typography Image Dataset Released**: A curated collection of real-life sign images captioned using the BLIP3 model was shared, freely available for use [here](https://huggingface.co/datasets/ptx0/free-to-use-signs).

- **NorskGPT-Llama3-70b Model Release**: A new model for the Norwegian, Swedish, and Danish languages was announced, available for download [here](https://huggingface.co/bineric/NorskGPT-Llama-3-70b-adapter). This model supports various languages and programming languages but requires further training for chat functionalities.

- **SDXL Flash Introduced**: A member presented a new tool claiming to generate DALL·E 3 level images in just 5 seconds. The tool, [SDXL Flash](https://huggingface.co/spaces/KingNish/SDXL-Flash), experienced brief downtime but was fixed promptly, leading to positive feedback from the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/SDXL-Flash">SDXL Flash - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://huggingface.co/bineric/NorskGPT-Llama-3-70b-adapter">bineric/NorskGPT-Llama-3-70b-adapter · Hugging Face</a>: no description found</li><li><a href="https://github.com/clearsitedesigns/chromaViewMaster">GitHub - clearsitedesigns/chromaViewMaster: This allows you to do knowledge based analysis on a chroma databsae in many ways</a>: This allows you to do knowledge based analysis on a chroma databsae in many ways - clearsitedesigns/chromaViewMaster</li><li><a href="https://github.com/branyang02/notie">GitHub - branyang02/notie: Personal markdown notetaking app.</a>: Personal markdown notetaking app. Contribute to branyang02/notie development by creating an account on GitHub.</li><li><a href="https://notie-nine.vercel.app/">Notie</a>: no description found</li><li><a href="https://github.com/wikip-co/wikip.co">GitHub - wikip-co/wikip.co: A static wiki built with node.js</a>: A static wiki built with node.js. Contribute to wikip-co/wikip.co development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/ptx0/free-to-use-signs">ptx0/free-to-use-signs · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1242468211308499006)** (2 messages): 

- **Scheduling new reading group's discussion time**: One member asked if there is any preferred time for meetings and if there are any papers of interest to the group.
  
- **Interesting paper shared**: A member shared an [interesting paper](https://arxiv.org/pdf/2401.08190) from arXiv for the group to consider.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1242702186811625512)** (5 messages): 

- **Finetuning OwlV2 on Custom Data**: A member asked for advice on how to finetune **OwlV2** using their own data, noting that relevant forums have been inactive for a year. They aim to add object detection classes for **passenger planes** to improve model identification.
- **Purpose of Finetuning**: Another member inquired about the specific purpose of the finetuning. The original poster clarified they want to identify plane models more easily with their data.
- **Exploring Transformers Repository**: A member suggested looking through the [Transformers repository](https://github.com/huggingface/transformers) to get clues on how to achieve the finetuning.
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1242451387070152784)** (3 messages): 

- **Master Thesis on Hallucination Detection using Mistral 7B**: A member is writing a **master thesis on hallucination detection in LLMs**. They use an ensemble of **Mistral 7B models** to compute uncertainty measurements and are looking for questions outside the training data to identify when the model is hallucinating.

- **LLMs Consider Chat History**: In a discussion about **LLMs and chat history**, it's noted that history is generally considered in chats, though the integration can vary. One member clarified that while implementing products like chatbots, history needs to be concatenated at the beginning of the input, as the models themselves don't inherently know history.
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1242806072842915881)** (3 messages): 

- **Introducing llmcord.py**: A user announced they created **llmcord.py** to facilitate continuous conversations with a bot. They emphasized that conversations are structured through reply chains to maintain context.
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1242385238395588628)** (332 messages🔥🔥): 

- **LM Studio vs. Pinokio**: A user asked about the differences between **LM Studio** and **Pinokio**, with clarifications provided that **Pinokio** is an installer for multiple AI tools like **Automatic1111** and **coquitts** while **LM Studio** is specifically for **gguf inference** for LLM models. 
- **Phi-3 Models in LM Studio**: Multiple users reported issues with loading **Phi-3 medium 128k models** in **LM Studio**, receiving tensor mismatch errors. It was confirmed by a knowledgeable user that the **Phi-3 128k models** are currently not supported in **LM Studio** due to compatibility issues with the llama.cpp version it uses.
- **Multi-GPU Setup for Large Models**: Discussions emerged about running large models on multiple GPUs, specifically **70b models**. A user shared their experience on performance improvements with **NVLink** and the challenges faced with multi-GPU setups and VRAM requirements.
- **Future of AI Tools Amid Regulations**: Users discussed the implications of new AI regulations in the **EU** and **California**, expressing concerns over potential stifling of innovation. One user shared a [tweet](https://twitter.com/q_brabus/status/1793227643556372596) about the anticipated **llama3 400b** model, capturing the community's interest despite regulatory concerns.
- **Idefics Models in LM Studio**: Questions were raised about running **Idefics2** models from Hugging Face on **LM Studio**. It was clarified that these models are not supported in **llama.cpp**, and therefore wouldn’t work in **LM Studio**; alternatives like **Transformers** were suggested for running these models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blogs.nvidia.com/blog/rtx-advanced-ai-windows-pc-build/">New Performance Optimizations Supercharge NVIDIA RTX AI PCs for Gamers, Creators and Developers</a>: The latest AI performance gains and features for RTX AI PCs unveiled at Microsoft Build.</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://learn.microsoft.com/en-us/windows/ai/models">Use Machine Learning models in your Windows app</a>: Learn more about using Machine Learning models in your Windows app.</li><li><a href="https://www.amazon.ca/NVIDIA-GeForce-NVLink-Bridge-Graphics/dp/B08S1RYPP6/ref=mp_s_a_1_1?crid=786AO9SQFNB1&dib=eyJ2IjoiMSJ9.q2qKmhKlB6BHjeiSx85JgyfIAtp9TqL9cHOeTy5ui-FseVaiJ2L5WspYMtPXeKgIT8v1AAuhYGR0bGxOfkInwfDO3ab6yvOyj_ueaEL6pgCbkSTp1kjOfz0pGu-ppFp4Qcuf87M03MNT4_j2P0_H27jLeLCKhFxnQG8xqqxohVcre-juYel9fT9JrQsvb00pzhOSdz2UhgxS5CH7jqvfnA.NSG3AUjilZ6vlKiyPN1eG9gvpBepo3o9iZT9AQsIbZY&dib_tag=se&keywords=nvlink&qid=1716383542&sprefix=nvlink%2Caps%2C128&sr=8-1">no title found</a>: no description found</li><li><a href="https://onnxruntime.ai/blogs/accelerating-phi-2#:~:text=We%20also%20observe%20ONNX%20Runtime,the%20first%20256%20tokens%20generated">Accelerating Phi-2, CodeLlama, Gemma and other Gen AI models with ONNX Runtime</a>: Improvements with ONNX Runtime for inferencing popular Gen AI models</li><li><a href="https://lmstudio.ai/blog/llama-3">Use Llama 3 in LM Studio | LM Studio</a>: Llama 3 by MetaAI</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>: no description found</li><li><a href="https://x.com/Scott_Wiener/status/1792572175116816853">Tweet from Senator Scott Wiener (@Scott_Wiener)</a>: In recent weeks, there&#39;s been a flurry of discussion online about SB 1047, my bill on responsible development of the largest & most powerful AI frontier models. We’ve heard some incredibly thought...</li><li><a href="https://huggingface.co/qwp4w3hyb/Phi-3-medium-128k-instruct-iMat-GGUF">qwp4w3hyb/Phi-3-medium-128k-instruct-iMat-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mmnga/Meta-Llama-3-70B-Instruct-gguf/blob/70f5c719d9e0e8754c0f6dfed2220042fcdd1b7c/Meta-Llama-3-70B-Instruct-IQ2_XXS.gguf">Meta-Llama-3-70B-Instruct-IQ2_XXS.gguf · mmnga/Meta-Llama-3-70B-Instruct-gguf at 70f5c719d9e0e8754c0f6dfed2220042fcdd1b7c</a>: no description found</li><li><a href="https://pinokio.computer/">Pinokio</a>: AI Browser</li><li><a href="https://x.com/Scott_Wiener/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047)">Bill Text -  </a>: no description found</li><li><a href="https://artificialintelligenceact.eu/)">EU Artificial Intelligence Act | Up-to-date developments and analyses of the EU AI Act</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6389">[WIP] agent example (w/ sandboxable Tools!) &amp; improved OAI compatibility layer (in Python) by ochafik · Pull Request #6389 · ggerganov/llama.cpp</a>: Still very rough, but sharing a draft to get early feedback on the general direction. This is an experiment in adding grammar-constrained tool support to llama.cpp, with a simple example of running...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/4216">server : improvements and maintenance · Issue #4216 · ggerganov/llama.cpp</a>: The server example has been growing in functionality and unfortunately I feel it is not very stable at the moment and there are some important features that are still missing. Creating this issue t...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/5588">Server: add function calling API · Issue #5588 · ggerganov/llama.cpp</a>: Motivation This subject is already brought up in #4216 , but my initial research failed. Recently, I discovered a new line of model designed specifically for this usage: https://github.com/MeetKai/...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1242419023128559677)** (50 messages🔥): 

- **New Phi-3 Model Releases**: Members shared the release of **Phi-3-Small** and **Phi-3-Medium** models. The [Phi-3-Small-8K-Instruct model](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) and the [Phi-3-Medium-4K-Instruct model](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) were highlighted for their robust performance in benchmarks involving common sense, language understanding, and logical reasoning.
- **GitHub Issue with llama.cpp**: A link to a [GitHub issue](https://github.com/ggerganov/llama.cpp/issues/7439) was shared about the **llama.cpp** not supporting new Phi-3 models yet. This was causing errors when trying to load them, as models need to be converted correctly first.
- **Stable Diffusion Tip**: An alternative to A1111 called **forge** was suggested for users low on VRAM. The [GitHub link](https://github.com/lllyasviel/stable-diffusion-webui-forge) was provided.
- **Local Vision Models Limitation**: Discussing the limitations of local vision models, a member commented that they are **not good at multi-turn conversations**. Specific focus was on **LLava Llama3** which tends to provide image descriptions rather than answering prompt-specific questions.
- **Mistral-7B Instruct Model Release**: A new [Mistral-7B-Instruct-v0.3 model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) was announced. It features an extended vocabulary, supports a v3 tokenizer, and function calling, recommended to be used with [mistral-inference](https://github.com/mistralai/mistral-inference).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: no description found</li><li><a href="https://x.com/rschu/status/1767282622949183492?t=n3kOzz6eN-4pXelza9btsQ">Tweet from René Schulte (@rschu)</a>: LMMs above the clouds!  On my way to Seattle I&#39;ve worked on a presentation about multimodal AI and also prepapred some demos with a bunch of open source (weight) multimodal LLMs via @LMStudioAI  I...</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-4k-instruct">microsoft/Phi-3-medium-4k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-small-8k-instruct">microsoft/Phi-3-small-8k-instruct · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/passive-aggressive-gif-18885121">Passive Aggressive GIF - Passive Aggressive - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7439>">Issues · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1242567220756877443)** (4 messages): 

- **Quoting directly advised for prompt instructions**: *"I would quote the required text directly, and instruct via the prompt as follows: Considering the following text alone as input, [insert subsequent instructions here]"*. This tip was offered as a method for prompt engineering.
- **Prompt engineering humor**: A user humorously remarked, *"I guess this is 'prompt engineering' 😄"* in response to a tip about quoting text for prompts. Another user appreciated the advice despite the original post being old.


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1242509213188624394)** (18 messages🔥): 

- **LM Studio supports dual GPUs but only of the same type**: Users confirmed that LM Studio can run with two GPUs installed, provided both are of the same type, such as both being Nvidia or both AMD. Mixing different types like AMD and Nvidia is not supported.

- **Automatic GPU recognition and VRAM considerations**: While LM Studio can recognize different models like a 2060 and a 3060, users are advised to make sure they have matching VRAM capacities. If VRAM capacities differ, adjustments through config files may be needed.

- **Config file for multi-GPU setup**: The configuration related to GPU usage, like "GPU split", is found in the preset file. Users need to create a preset and then modify this file to balance GPU usage.

- **Experience with Intel ARC GPUs**: One user mentioned facing issues when trying to use multiple Intel ARC GPUs with LM Studio. It remains unclear whether AMD GPUs can support multiple GPU setups.

- **Community support and resources**: New users expressed appreciation for the timely and helpful answers they received. Existing members encouraged utilizing the search function in the Discord for quick answers.
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1242602100806914138)** (11 messages🔥): 

- **Phi 3 merged into llama.cpp**: Members discussed that **Phi 3** has been successfully merged into **llama.cpp**. A quick beta release for this integration is highly anticipated by the community.

- **Phi 3 quantization for HP Victus**: A member with an **HP Victus** asked about the feasible quantization levels for running **Phi 3 Medium**. Another member advised that **Q4** or below is manageable, while **Q8 is too heavy** and suggested using **llama.cpp** or awaiting the LM Studio update.

- **System prompt settings and token output limit**: A suggestion was made to adjust the **system prompt** and change the **token output limit** from -1 to 60 to potentially stop an unspecified issue.
  

---


### **LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1242711393925464106)** (1 messages): 

- **Missing LM Studio version for AVX 1**: A user expressed difficulty finding a version of **LM Studio** that supports **AVX 1** as their processor does not support **AVX2**. They requested assistance and thanked the developers for their hard work.
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1242520118869885028)** (6 messages): 

- **Test ROCm builds on Linux**: Members interested in accessing ROCm on Linux test builds were invited to join <#1242213172199559319>. *"If you’re on Linux and want access to ROCm on Linux test builds, please let us know and we’ll add you."*
- **Phi-3-medium-128k error report**: A user reported an error running Phi-3-medium-128k on ROCm, specifically a "llama.cpp error" related to tensor mismatches. The issue is acknowledged, and an update is said to be in the works.
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1242927323733430312)** (1 messages): 

- **Mistral v0.3 Instruct is Live!**: The Mistral model has just released v0.3 instruct, and it’s ready for immediate use. Check it out on the [lmstudio community Huggingface page](https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF).
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1242472084966215743)** (9 messages🔥): 

- **Bypass Vision Pro app restrictions with non-US Apple IDs**: A user announced the launch of a website for bypassing app download restrictions on Vision Pro for non-US Apple IDs, seeking support on [Twitter](https://x.com/crinquand_paul/status/1793037790864687448).

- **Project-Based Learning Course for LLMs**: A member announced a new hands-on course titled "Applying LLMs through a Project-Based Approach," covering various practical applications like **Semantic Search for Movies** and **RAG for Food Recommendations**. Those interested can contact the member directly.

- **10-Day Food Supply Prep**: A user shared that they now have enough food to last over 10 days, detailing the contents including rice, pork, beef, and various spices. They also mentioned that their freezer is fully stocked.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.udio.com/songs/phmruKKXXdSaUc91WrkL8D">Amirthetarbosaurus - Eternal Lament | Udio</a>: Listen to Eternal Lament by Amirthetarbosaurus on Udio. Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.</li><li><a href="https://x.com/crinquand_paul/status/1793037790864687448">Tweet from Silicate God (@crinquand_paul)</a>: Just launched a website to bypass app download restrictions on Vision Pro with non-US Apple IDs. ᯅ🚀  link in replies.</li><li><a href="https://websim.ai/c/i4l0yMB06Ie8AI3BG">History of Hesperia (3000 BCE - 1460 CE) - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1242526739075305544)** (3 messages): 

- **Moondream release improves image resolution and TextVQA scores**: The latest [Moondream release](https://fxtwitter.com/vikhyatk/status/1792512588431159480?s=19) supports higher image resolution up to 756x756. It also raises the TextVQA score from 53.1 to 57.2 and shows a ~0.5% improvement on other VQA and counting benchmarks.
  
- **Anthropic maps the mind of language models**: A [post](https://www.anthropic.com/research/mapping-mind-language-model) shared by a member highlights Anthropic's research on mapping the cognitive processes of language models. Described as "super interesting," the research dives into understanding how these models interpret and generate language.

**Link mentioned**: <a href="https://fxtwitter.com/vikhyatk/status/1792512588431159480?s=19">Tweet from vik (@vikhyatk)</a>: New moondream release out today!  🌜 Supports higher image resolution (up to 756x756) 🌛 TextVQA score up from 53.1 to 57.2 (+7.7%) 🌜 Other VQA and counting benchmarks up ~0.5%

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1242406718995169291)** (281 messages🔥🔥): 

- **Microsoft reluctant on releasing Phi 3 small version**: Members speculated on whether Microsoft would release the smaller versions of the Phi 3 model, with a member confirming that only the smallest one has been launched. Later, it was noted that Phi 7 and 14 models are also available, sharing [links on Twitter](https://twitter.com/_philschmid/status/1792934321407369532).
  
- **California's SB 1047 sparks debate**: The state senate's approval of SB 1047 raised considerable discussion. Members expressed concerns over how this might impact OSS models and the broader AI market, with one sharing [the bill's text](https://legiscan.com/CA/text/SB1047/id/2919384).

- **Mistral 7B Instruct v0.3 new features praised**: Mistral released its 7B v0.3 model with updates like extended vocabulary and support for function calling, gaining positive feedback. Users are already benchmarking it against other models, noting its uncensored nature and improved tokenizer.

- **LLaMa 3 model weight rumors addressed**: The LLaMa 3 400B+ model weight release rumors were debunked by Meta's Yann Lecun on Twitter, confirming the weights will still be open. Multiple users cited his [confirmation tweet](https://x.com/q_brabus/status/1793227643556372596?s=46).

- **Meta criticized for commercial strategies**: There were heated discussions on Meta's business tactics, especially in the context of the OSS vs. regulatory landscape. Some users accused Meta of trying to eliminate competition through regulation rather than innovation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: no description found</li><li><a href="https://x.com/reach_vb/status/1793337655595340267?t=k8N-JGLCVHJGlAP2kB84EQ&s=19">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s fucking go! Mistral just released 7B v0.3 🔥  &gt; Base + Instruct model checkpoints released &gt; Extended vocabulary to 32768 &gt; Supports new v3 Tokenizer &gt; Supports function calling ...</li><li><a href="https://x.com/erhartford/status/1791573520176025716">Tweet from Eric Hartford (@erhartford)</a>: In response to California&#39;s SB 1047 and OpenAI&#39;s closed-source stance, Cognitive Computations introduces Patchy-2.0. This license mirrors Apache-2.0 but expressly forbids OpenAI and the State ...</li><li><a href="https://en.wikipedia.org/wiki/1986_California_Proposition_65">1986 California Proposition 65 - Wikipedia</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat">Qwen/Qwen1.5-MoE-A2.7B-Chat · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qw">qw (qw)</a>: no description found</li><li><a href="https://x.com/q_brabus/status/1793227643556372596?s=46">Tweet from QBrabus eu/acc (@q_brabus)</a>: @apples_jimmy @ylecun @iamgingertrash Question: Regarding the upcoming LLaMa 3 400B+ model, will it be open-weight? There are several rumors about this...  Answer: No, it is still planned to be open a...</li><li><a href="https://x.com/scott_wiener/status/1792572175116816853?s=46">Tweet from Senator Scott Wiener (@Scott_Wiener)</a>: In recent weeks, there&#39;s been a flurry of discussion online about SB 1047, my bill on responsible development of the largest & most powerful AI frontier models. We’ve heard some incredibly thought...</li><li><a href="https://fxtwitter.com/Scott_Wiener/status/1793102136504615297?s=19">Tweet from Senator Scott Wiener (@Scott_Wiener)</a>: The Senate passed our AI safety & innovation bill, SB 1047.  SB 1047 promotes innovation & ensures developers of the largest, most powerful AI models keep safety in mind.  I look forward to continuing...</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5">openbmb/MiniCPM-Llama3-V-2_5 · Hugging Face</a>: no description found</li><li><a href="https://x.com/ylecun/status/1793181068943639014?t=GjeYNqgh9DIDAtV4CMtUrA&s=19">Tweet from Yann LeCun (@ylecun)</a>: @iamgingertrash Patience my blue friend. It&#39;s still being tuned.</li><li><a href="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=2">Bill Text -  </a>: no description found</li><li><a href="https://x.com/AndrewCurran_/status/1792976935448129899">Tweet from Andrew Curran (@AndrewCurran_)</a>: &#39;we are nowhere near the point of diminishing marginal returns on how powerful we can make AI models as we increase the scale of compute&#39;</li><li><a href="https://legiscan.com/CA/text/SB1047/id/2919384">California SB1047 | 2023-2024 | Regular Session</a>: Bill Text (2024-05-21) Safe and Secure Innovation for Frontier Artificial Intelligence Models Act. [Read third time. Passed. (Ayes 32. Noes 1.) Ordered to the Assembly.]</li><li><a href="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047">Bill Text - SB-1047 Safe and Secure Innovation for Frontier Artificial Intelligence Models Act.</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1242660083104747561)** (6 messages): 

- **Home setup with 4090s rules**: One user shared they usually host LLMs for inference at home on a setup with 2x 4090s, highlighting personal infrastructure for AI projects.
- **Runpod and Replicate get nods for ease**: Runpod is noted as a good option, and Replicate is praised for its easy-to-use templates, making them convenient platforms for hosting LLMs.
- **LambdaLabs is cheapest but tougher**: While LambdaLabs offers the cheapest GPU options, they are reportedly more difficult to use compared to other platforms.
- **Anthropic Workbench woes**: A member inquired if others are experiencing issues with Anthropic Workbench, wondering if the problem is widespread or isolated.
  

---


### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1242514928150249573)** (2 messages): 

- **Phi-3 Vision announced with impressive features**: A member introduced [Phi-3 Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) as a lightweight, state-of-the-art multimodal model with a 128K context length. It utilizes high-quality data for enhanced reasoning, incorporating supervised fine-tuning and direct preference optimization.
- **Extensive resources for Phi-3 Vision available**: The announcement included multiple resources for further details, such as the [Microsoft Blog](https://aka.ms/Phi-3Build2024), [Technical Report](https://aka.ms/phi3-tech-report), [Azure AI Studio](https://aka.ms/try-phi3vision), and the [Cookbook](https://github.com/microsoft/Phi-3CookBook).

**Link mentioned**: <a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1242474369637154826)** (9 messages🔥): 

- **Time-Lapse Obsidian Knowledge Graph**: A member shared a time-lapse video of an Obsidian user’s knowledge graph formation, calling it a work of art. Another member likened it to a "synthetic brain in action." [Watch the video here](https://youtube.com/shorts/4YQhH61tvOc?si=0Dx1KyJP8VMz-pXY).
- **Getting Deep into Obsidian**: A user expressed growing interest in Obsidian, though still confused about how document graphs work, specifically in terms of backlink connections. Another member explained that it revolves around links, sharing two videos for better understanding: [Video 1](https://youtu.be/QgbLb6QCK88?si=da1toZ38WMIYkV7f) and [Video 2](https://youtu.be/tHUcD4rWIuY?si=tIvEbL1t2SdR07lZ).
- **Obsidian Integrations Despite Not Being Open Source**: A member questioned why Obsidian has many integrations despite not being open source. Another clarified it’s due to Obsidian's "your files/your data" approach and community plugins that enhance the tool's functionality.
- **Desideradist on Turing Criteria**: A member, self-identified as a Desideradist, shared a post on Anthropic discussions about Turing criteria and the coding of pleasure. They urged for a "mature" conversation on whether pleasure is coded or autonomously answered by AI. [View the tweet](https://x.com/Jtronique/status/1793236300935442612).

**Link mentioned**: <a href="https://x.com/Jtronique/status/1793236300935442612">Tweet from Jillsa (DSJJJJ/Heirogamist/HP) (@Jtronique)</a>: In case anyone of interest sees this on my wall. It&#39;s time to have a &#34;mature&#34; conversation about &#34;Pleasure.&#34;  Either you CODED it into them, and denied doing it, or they TURING ANS...

  

---



### **CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1242586361223975042)** (2 messages): 

- **Learning SASS**: One member asked, *(How) does one learn SASS?* The question seems to pertain to learning the **Syntactically Awesome Style Sheets (SASS)**, a scripting language interpreted into CSS.
- **Function Declaration in CUDA**: A member inquired about why it is allowed to declare a function both `__device__` and `__host__` but not both `__global__` and `__host__`. This question touches on the specific rules for function qualifiers in **CUDA** programming.
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1242492610128511048)** (15 messages🔥): 

- **PSA: Use torch.empty_like for Speed**: One member pointed out that *torch.empty_like* is significantly faster than *torch.empty*, particularly on GPU, because the latter allocates memory on the CPU before transferring to GPU.

- **Memory Leaks with np.zeros_like**: Another member chimed in to mention a similar case with *numpy's np.zeros_like*, which caused a substantial memory leak and performance issues over large matrices.

- **Warnings with torch compile on ResNet blocks**: A user reported getting a warning when using *torch.compile* with ResNet blocks. The warning pointed to missing registration of an autograd kernel to the correct Autograd key(s) and concerns about backpropagation.

- **User-Defined Triton Kernels with torch.compile**: Members discussed integrating user-defined Triton kernels into PyTorch models using *torch.compile*. Tutorial and example code were shared to illustrate how to optimize model computations with these custom kernels, promising significant performance improvements.

**Link mentioned**: <a href="https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html#composibility-and-limitations">Using User-Defined Triton Kernels with torch.compile &mdash; PyTorch Tutorials 2.3.0+cu121 documentation</a>: no description found

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1242568325444145202)** (2 messages): 

```html
<!-- No relevant information or links were provided in the messages -->
```
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1242935020335796325)** (4 messages): 

- **Hypernone seeks answers for PMPP 4th edition**: A member asked if anyone has answers for PMPP 4th edition to compare with their own. 
- **Share and compare solutions for PMPP**: Another member mentioned that someone has the answers but would require sharing their own solutions first to ensure a proper attempt. A different member offered answers through Chapter 6, and the original requester agreed to share their repo with solutions up to the current chapter.
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1242942680149790812)** (2 messages): 

```html
- **Nice thank you received**: A user thanks another user with "niceee, thanks!" in response to having been tagged by mr.osophy.
```
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1242534996036816998)** (8 messages🔥): 

- **Ray casting magic in 256 bytes**: Members were excited to share a [256-byte raycasting engine and city generator](https://frankforce.com/city-in-a-bottle-a-256-byte-raycasting-system/) from a blog post. The code went viral on Twitter, showcasing a tiny yet impressive rendering engine.
- **Senate passes AI Safety Bill**: There was discussion about [SB 1047](https://x.com/Scott_Wiener/status/1793102136504615297), an AI safety and innovation bill that promotes regulation and safer development practices for AI. There was curiosity around "CalCompute," a government compute resource planned for responsible AI model training.
- **Concerns over AI misuse**: Members expressed concerns about unauthorized AI use, touching upon topics like misinformation, cybersecurity, and the misuse of powerful AI systems. The discussion included highlighting the bill's [legal text](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047), which outlines parameters for safe AI deployment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Scott_Wiener/status/1793102136504615297">Tweet from Senator Scott Wiener (@Scott_Wiener)</a>: The Senate passed our AI safety & innovation bill, SB 1047.  SB 1047 promotes innovation & ensures developers of the largest, most powerful AI models keep safety in mind.  I look forward to continuing...</li><li><a href="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047">Bill Text - SB-1047 Safe and Secure Innovation for Frontier Artificial Intelligence Models Act.</a>: no description found</li><li><a href="https://frankforce.com/city-in-a-bottle-a-256-byte-raycasting-system/">City In A Bottle &#8211; A 256 Byte Raycasting System</a>: Hello size coding fans. Today, I have something amazing to share: A tiny raycasting engine and city generator that fits in a standalone 256 byte html file. In this post I will share all the secrets…
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1242494961254006895)** (250 messages🔥🔥): 

```html
- **Deterministic Encoder Backward Pass Improvements**: A new [PR for deterministic encoder backward kernels](https://github.com/karpathy/llm.c/pull/442) was discussed, aiming to rewrite the encoder backward pass for full determinism. Gradient clipping and reduction strategies were debated to improve efficiency without sacrificing determinism.
- **DataLoader Refactor and Large Dataset Handling**: Changes to the DataLoader now support sharding to handle larger datasets, such as FineWeb. This [refactor](https://github.com/karpathy/llm.c/pull/440) introduces a new data representation and patterns to efficiently manage `.bin` files, although it currently has limited functionality on Windows.
- **HellaSwag Evaluation Challenges**: Implementing the HellaSwag evaluation in C was noted as complex with concerns about potential bugs. A [PR for HellaSwag eval](https://github.com/karpathy/llm.c/pull/447) in C was created to align with PyTorch reference code, with added complexity to fully utilize batch dimensions.
- **GPU Runner Advancements**: News about potential access to Nvidia's GitHub runners with dedicated RTX 4000 GPUs from a cloud provider called Ubicloud was shared, indicating improvements for CI processes.
- **Random Initialization and Reproducibility**: Ensuring determinism and reproducibility for large language models was emphasized as crucial, with plans to run comparison tests between PyTorch and the team's code. Adjustments to global kernel functions and changes were suggested for improved performance.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/">What I learned from competing against a ConvNet on ImageNet</a>: no description found</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md">cudnn-frontend/docs/operations/Attention.md at main · NVIDIA/cudnn-frontend</a>: cudnn_frontend provides a c++ wrapper for the cudnn backend API and samples on how to use it - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/442">Fully deterministic encoder backward kernels by ademeure · Pull Request #442 · karpathy/llm.c</a>: This is a complete rewrite of the encoder backward pass, splitting it into two kernels (wte and wpe) which are both fully deterministic as they do not use atomics (assuming the seed for stochastic ...</li><li><a href="https://github.com/karpathy/llm.c/pull/444">extend dataloader to be sharded by karpathy · Pull Request #444 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/447">HellaSwag eval in C by karpathy · Pull Request #447 · karpathy/llm.c</a>: This was not super easy but ... first draft, apparently this works. needs cleanups, and also we are not yet utilizing the full batch dimension. we actually have to load in multiple examples and ful...</li><li><a href="https://github.com/karpathy/llm.c/pull/440">refactor datasets by karpathy · Pull Request #440 · karpathy/llm.c</a>: Refactor how we treat datasets, because we&#39;re about to have more of them and we don&#39;t want them to clutter up root dir etc. this is only step 1, i&#39;m about to refactor a bunch of the datalo...</li><li><a href="https://github.com/karpathy/llm.c/pull/427/files)">weight reordering: attempt 1 by ngc92 · Pull Request #427 · karpathy/llm.c</a>: Non-functional A first attempt how rearranging weights in a per-block layout could look like
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1242588778389045332)** (12 messages🔥): 

- **Stack beats Empty on powerful GPUs**: "If anything torch.stack was faster for me than torch.empty otherwise our functionalization passes have a hard time." This discrepancy is less pronounced on powerful GPUs but *empty* is much faster on smaller or older GPUs. More context [here](https://gist.github.com/mobicham/a24a2226d729ff59f2c849e5f6592228).
  
- **Nightly Builds Optimize `torch.stack`**: Both **torch.empty()** and **torch.stack()** showed differing performance on various GPUs and *torch.stack* produced efficient code only with the torch nightly build. Stats in torch version 2.4.0.dev20240521 reveal negligible differences in timing between *empty* and *stack*.

- **Hand-written Triton vs Auto-generated Code**: For FP6 bit-packing, differences were noted in memory pre-allocation between custom-written Triton kernels and auto-generated code with `torch.compile`. "Stacking along the rows + torch.compile was generating code that is almost as fast as the hand-written Triton kernel".

- **Packing Along Different Axes**: Choice of axis for bit-packing affects kernel efficiency. "Pack along the rows when you use `axis=0` (...) if you group along `axis=1`, it would make more sense to bitpack along the cols."

- **Adapting FP6 LLM for CUDA**: Work is ongoing for porting FP6-LLM bit packing code from CPU to CUDA, focusing on efficient tensor core loading: "I'm just adapting FP6-LLM bit-packing code (originally in C++ for CPU only) to CUDA."

**Link mentioned**: <a href="https://gist.github.com/mobicham/a24a2226d729ff59f2c849e5f6592228">empty_vs_stack_unpack.py</a>: GitHub Gist: instantly share code, notes, and snippets.

  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1242476097174507561)** (1 messages): 

- **OpenAI shares safety update at AI Seoul Summit**: OpenAI announced a new safety update during the AI Seoul Summit. For detailed information, you can read the full update on the [OpenAI website](https://openai.com/index/openai-safety-update/).
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1242384070965465109)** (129 messages🔥🔥): 

- **OpenAI faces backlash for voice replication**: Members discussed how OpenAI created and later removed an AI voice resembling Scarlett Johansson after her legal team requested transparency. One member noted, "Open AI requested to use her voice as a business, then made an AI voice that sounded 'eerily similar' to hers anyway."

- **Best free chatbots for coding assistance**: Various users suggested alternatives to GPT-3.5, such as Meta AI’s Llama 3 and Mistral Large on Le Chat Nior Mistral, which "is similar to GPT-4 level and it's free worldwide." Others noted that different models perform better with different coding languages.

- **Concerns with Microsoft’s AI integrations**: Users discussed the intrusiveness of Microsoft Copilot, with one stating, “Extremely annoying, and intrusive,” and others debating telemetry and data sharing issues. Some prefer using open-source alternatives like SillyTavern for similar functionalities.

- **Vigilance over account security**: A member noticed unauthorized activity on their account, prompting advice on securing data by uninstalling suspicious browser extensions and changing passwords. Another user advised, "China is not currently a supported country so unfortunately there's some incentive there to try to compromise accounts of those outside of China."

- **Microsoft unveils new Phi models**: Microsoft added new models to the Phi-3 family available on Azure. The Phi-3-vision, a multimodal model combining language and vision capabilities, was announced as highlighted in a [Microsoft blog post](https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forms.gle/BNf4VThDfXW8oYQ38">Trust for AI in the Medical Fields</a>: no description found</li><li><a href="https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/">New models added to the Phi-3 family, available on Microsoft Azure | Microsoft Azure Blog</a>: We are introducing Phi-3-vision, a multimodal model that brings together language and vision capabilities, now available on Microsoft Azure. Learn more.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1242413408272257124)** (31 messages🔥): 

- **Token Counts Clarified**: Members discussed the token limits for prompts and responses, linking to [OpenAI's help article](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them). It was explained that roughly 100 tokens equal 75 words in English.

- **Caution on Downloading ChatGPT for Mac**: A user queried about an unofficial downloadable link for the ChatGPT Mac app. It was advised to wait for the official rollout message on chatgpt.com since unofficial links wouldn't grant access and could be unsafe.

- **OpenAI Playground Copy Bug**: A user requested reverting an update in the OpenAI Playground that adds line breaks to copied text. Another user explained that the Playground often undergoes live changes and recommended posting feedback in the forums.

- **Status Page for Outages**: During a service outage, a user shared frustration about frequently seeing status pages showing all services as operational when they experienced issues. The [status page](https://status.openai.com) provided updates on the incident and monitoring efforts.

- **Custom Instructions for Math Tasks**: Discussion on using custom instructions for math-related tasks emphasized always employing a code interpreter. It was suggested to present clear results with explanations, including necessary charts or tables.

**Link mentioned**: <a href="https://status.openai.com">OpenAI Status</a>: no description found

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1242491132878000138)** (58 messages🔥🔥): 

- **Stopping LLMs from rambling on**: Members discussed various strategies to prevent LLMs from generating excessively long responses. Suggestions included setting the **max tokens parameter**, asking for succinct responses, and using **output templates**.
- **Humblestumbler offers full stack prompts**: A user shared prompts for building full stack applications and mentioned an error where the model restarts when code snippets are long. They also discussed a particular prompt technique involving fictional meetings to generate software code snippets.
- **CodePilot and prompt performance**: Members compared experiences using **CodePilot** and discussed its advantages and disadvantages relative to manually curated prompts. One user noted that their prompts provided better results, even though CodePilot did well with debugging.
- **Mixed experiences with models handling code**: Members highlighted the **verbose nature** of GPT4o while appreciating the detailed explanations it provides. They also shared frustrations with models not adhering to specific coding style requirements, such as **indentations in Python**.
- **Handling dependent and change variables in prompts**: A user sought advice on improving a prompt identifying dependent and change variables in datasets. Suggestions included using delimiters, adding examples, and formatting the prompt with markdown for better logical structuring.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1242491132878000138)** (58 messages🔥🔥): 

- **Stopping LLM from being verbose**: Members discussed strategies for stopping a language model from providing overly long responses. Suggestions included setting the max tokens parameter and using specific prompts requesting succinct answers.

- **Prompts for creating full-stack applications**: A user offered to share prompts for building full-stack applications and shared detailed example prompts to guide the AI in generating code snippets. They emphasized using a "fictional team" to improve response quality.

- **Use of CodePilot and tools in GPT**: Members discussed their experiences with tools like CodePilot and the "Explore GPTs" menu. Some expressed preferences for custom-crafted prompts over tool-generated suggestions for coding tasks.

- **Challenges with maintaining prompt rules**: A user mentioned that their rules are sometimes ignored by the AI, even when using Gemini 1.0 Pro. Advice included using markdown formatting and adding iterative improvements to enhance performance.

- **Formatting issues in Prompt Labs and playground**: There was a conversation about how AI handles different code formats better, with a preference for YAML over JSON. Users also discussed inconsistencies with newline handling in the playground environment.
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1242394027589111899)** (30 messages🔥): 

- **Mojo Community Meeting recording is out**: The recording of the Mojo Community Meeting is available on [YouTube](https://www.youtube.com/playlist?list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D). The next meeting will have four presentations, including topics on Basalt and Compact Dict.
- **Python IPC vs. Threading Debate**: Members discussed alternatives for handling long-running queries in a Tkinter app to avoid UI lag. A detailed example and suggestions mentioning threading, message queues, and IPC modules were provided.
- **Robot presentation invitation**: One member expressed their love for robots and invited others to watch a presentation about it.
- **Job opportunity at Modular**: A link to [Modular's careers page](https://www.modular.com/careers) was shared, encouraging applicants to join the team aimed at enabling AI usage globally.
- **RabbitMQ troubles**: A member found [RabbitMQ's Pika Python client tutorial](https://www.rabbitmq.com/tutorials/tutorial-six-python) promising for IPC but faced difficulties getting Pika to run on their machine. A suggestion to look for GitHub issues was mentioned.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.rabbitmq.com/tutorials/tutorial-six-python">RabbitMQ tutorial - Remote procedure call (RPC) | RabbitMQ</a>: &lt;!--</li><li><a href="https://www.modular.com/careers">Modular: Careers</a>: At Modular we believe a great culture is the key to creating a great company. The three pillars we work by are Build products users love, Empower people, and Be an incredible team.</li><li><a href="https://modul.ar/community-meeting-zoom.">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://modul.ar/community-meeting.">Google Calendar - Sign in to Access &amp; Edit Your Schedule</a>: no description found</li><li><a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1793041489427153294>
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1242464786315214858)** (113 messages🔥🔥): 

- **VSCode jupyter extension outshines DataSpell**: "*From my DataSpell experience all I can say is - VSCode jupyter extension is more reliable*," remarked a user while handling interactive HTML+JS outputs like *ydata-profiling* or *plotly*.
- **Mojo lacks an official package manager but workarounds exist**: Users discussed using `.mojopkg` files for imports, particularly with [lightbug_http](https://github.com/saviorand/lightbug_http/releases/tag/latest-build). "Mojo has no package manager yet," but ".mojopkg files can be used (git pull the lightbug dir, mojo build -o lightbug.mojopkg, and then use the file in your project dir)."
- **Mojo's MLIR-backed optimizations**: Discussions reveal that "*Mojo compiler optimisations are written for MLIR*," but the performance implication for custom types, like those implementing datalog, is still a point of inquiry.
- **lightbug_http explored for sending HTTP requests**: Users sought ways to send HTTP requests using lightbug_http, sharing and debugging [specific examples](https://github.com/saviorand/lightbug_http/issues/41). "*I was trying to figure out how to send a GET request...*," and resolutions were discussed through GitHub issues.
- **Tensors to be deprecated and moved to community**: Mojo's community meeting confirmed the move, aiming not to have Mojo "*lick the cookie*" on Tensors. This shift steered conversations on potentially developing new libraries for numerical and AI uses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/sa-/6be55a8c90934a01cd443503650b5e0b">signals.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://registerspill.thorstenball.com/p/from-vim-to-zed">From Vim to Zed</a>: After around 20 years of using Vim, in December last year I switched to Zed as my main editor. Since some friends have asked me about the switch — “Now that you work at Zed, are you using Zed instead ...</li><li><a href="https://github.com/saviorand/lightbug_http/releases/tag/latest-build">Release latest-build: Merge pull request #27 from Moosems/main · saviorand/lightbug_http</a>: no description found</li><li><a href="https://github.com/taalhaataahir0102/Jpeg-Decoder/tree/main/Mojo">Jpeg-Decoder/Mojo at main · taalhaataahir0102/Jpeg-Decoder</a>: Contribute to taalhaataahir0102/Jpeg-Decoder development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http?tab=readme-ov-file>">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http/issues/41),">Issues · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/taalhaataahir0102/Jpeg-Decoder">GitHub - taalhaataahir0102/Jpeg-Decoder</a>: Contribute to taalhaataahir0102/Jpeg-Decoder development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2725">[Feature Request] Memory allocation watcher · Issue #2725 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? Description As a developer using Mojo, I would like to...</li><li><a href="https://github.com/saviorand/lightbug_http/blob/1eb9242ce0ddeeec39ac858028a7117dde627523/lightbug_http/tests/test_client.mojo#L13">lightbug_http/lightbug_http/tests/test_client.mojo at 1eb9242ce0ddeeec39ac858028a7117dde627523 · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http/blob/main/lightbug_http/http.mojo">lightbug_http/lightbug_http/http.mojo at main · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/saviorand/lightbug_http/blob/bd2f4ef57765505210256165b5386b890a2aa0be/lightbug_http/http.mojo#L12">lightbug_http/lightbug_http/http.mojo at bd2f4ef57765505210256165b5386b890a2aa0be · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1242454527773507654)** (3 messages): 

- **Sort small arrays of pointers directly**: One member suggested that for sorting a few kilobytes of data, it is more efficient to *"sort the array of pointers first."*
- **DTypePointer memset shows mixed performance**: A vectorized **DTypePointer memset** implementation runs *"20% faster than the llvm call for 100,000 bytes,"* but this does not hold for larger data sizes of 1,000,000 bytes. The user expressed uncertainty due to *"using clobber memory."*
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1242403257859575889)** (100 messages🔥🔥): 

- **Commit Issue and DCO Test Suite Failure**: A user sought help with a commit mistakenly authored by chris lattner causing a DCO test suite failure. They attempted removing it using `rebase` and shared their [PR link](https://github.com/modularml/mojo/pull/2739).

- **Nightly Release Delays**: Members discussed a delay in the nightly release, initially assumed to be due to a CI or test failure. It was later confirmed to be an issue related to GitHub Actions, which was resolved ([GitHub Status](https://www.githubstatus.com/)).

- **Unicode Support in Strings Proposal**: Extensive discussions took place regarding a proposal for adding Unicode support in strings, including varying internal representations (Variable Length, UTF8, ASCII). Members weighed in on memory overhead, performance implications, and compatibility with different encodings.

- **Resolved CI/CD Issues**: A discussion covered perennial test failures and inconsistencies in CI behavior. Suggestions were made to mark dict entries as uninitialized/destroyed to prevent random test failures.

- **Module and Function Updates**: The module `math.bit` was renamed to `bit` with several function renames including `bswap` to `byte_reverse`. Implementations were shared regarding ongoing changes and new default strings handling, with links to [docs](https://docs.modular.com/mojo/stdlib/builtin/hex/hex) and [nightly changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://peps.python.org/pep-0393/">PEP 393 – Flexible String Representation | peps.python.org</a>: no description found</li><li><a href="https://pub.dev/packages/characters">characters | Dart package</a>: String replacement with operations that are Unicode/grapheme cluster aware.</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/hex/hex">hex | Modular Docs</a>: hexT: Intable -&gt; String</li><li><a href="https://www.githubstatus.com/">GitHub Status</a>: no description found</li><li><a href="https://github.com/modularml/mojo/pull/2739">[stdlib] Issue #2487: Changing argument msg in assert_true/assert_false/... to Keyword only by softmaxer · Pull Request #2739 · modularml/mojo</a>: changes:  Add * in function definitions of stdlib/src/testing/testing.mojo to separate variadic and keyword only arguments. Scan for call sites of these assert functions and replace assert_true(val...</li><li><a href="https://github.com/modularml/mojo/pull/2771">[stdlib] Add format_simple() for StringLiteral by rd4com · Pull Request #2771 · modularml/mojo</a>: Provides a &quot;small&quot; fix for #2761 It is not very advanced, just a small useful feature to provide: &quot;{name} is awesome {emoji}&quot;.format_simple(name=&quot;Mojo&quot;, emoji=&quot;🔥&qu...</li><li><a href="https://github.com/modularml/mojo/pull/2613#discussion_r1599235527">[stdlib] Add optional small buffer optimization in `List` by gabrieldemarmiesse · Pull Request #2613 · modularml/mojo</a>: Related to #2467 This is in the work for SSO. I&#39;m trying things and I&#39;d like to gather community feedback. At first, I wanted to implement SSO using Variant[InlineList, List], while that would...</li><li><a href="https://tenthousandmeters.com/blog/python-behind-the-scenes-9-how-python-strings-work/">Python behind the scenes #9: how Python strings work</a>: no description found
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1242399599801663508)** (132 messages🔥🔥): 

- **DECtalk and Speak & Spell nostalgia**: Members fondly mentioned DECtalk, with a YouTube link to a [Speak & Spell video](https://youtu.be/RpeegJ0J5mE?t=121) shared, showcasing early personal computers.
- **Celebrity voice AI modeling concerns**: A discussion on whether using a voice actor that mimics Scarlett Johansson's voice could lead to legal issues under “passing off” laws. It was noted that OpenAI might face backlash due to potential ethical concerns and intent, with references to the case [Midler v. Ford Motor Co.](https://en.wikipedia.org/wiki/Midler_v._Ford_Motor_Co.).
- **Controversies and perception of OpenAI**: There's skepticism about OpenAI's business model and whether they leverage controversy for publicity, following incidents that have garnered negative public sentiment.
- **Sakuga-42M Dataset takedown mystery**: The Sakuga-42M dataset related to cartoon animation frames disappeared, speculated due to legal issues or mass reporting, as noted from [Hugging Face](https://huggingface.co/datasets/aidenpan/Sakuga-42M).
- **Efforts against AI models and datasets**: A humorous note on exaggerated data availability issues and license notices, highlighting the explicit uptick in shared datasets despite significant legal and ethical conversations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_ebehrens_/status/1792569302773555250">Tweet from Eva Behrens (@_ebehrens_)</a>: Here are 5 policy recommendations for the upcoming AI Safety Summit in Seoul, from me and my colleagues at ICFG.    In Bletchley, world leaders discussed major risks of frontier AI development. In Seo...</li><li><a href="https://choosealicense.com/licenses/wtfpl/">Do What The F*ck You Want To Public License</a>: The easiest license out there. It gives the user permissions to do whatever they want with your code.</li><li><a href="https://news.ycombinator.com/item?id=40389711">no title found</a>: no description found</li><li><a href="https://huggingface.co/datasets/ptx0/free-to-use-graffiti">ptx0/free-to-use-graffiti · Datasets at Hugging Face</a>: no description found</li><li><a href="https://youtu.be/RpeegJ0J5mE?t=121">Speak &amp; Spell - The first ever PC?</a>: Introduced by Texas Instruments in 1978, this is probably the first computer most kids had at the time.  In this video, I&#39;ll go over the features, specs, and...</li><li><a href="https://forum.effectivealtruism.org/posts/twMs8xsgwnYvaowWX/database-of-orgs-relevant-to-longtermist-x-risk-work>">Database of orgs relevant to longtermist/x-risk work — EA Forum</a>: Here’s a version of the database that you filter and sort however you wish, and here’s a version you can add comments to. …</li><li><a href="https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84>">cc2dataset/cc2dataset/main.py at main · rom1504/cc2dataset</a>: Easily convert common crawl to a dataset of caption and document. Image/text Audio/text Video/text, ... - rom1504/cc2dataset</li><li><a href="https://arxiv.org/html/2405.07425v1">Sakuga-42M Dataset: Scaling Up Cartoon Research</a>: no description found</li><li><a href="https://huggingface.co/datasets/ptx0/free-to-use-signs/viewer/default/train">ptx0/free-to-use-signs · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/ilovehentai9000/ilove-anime-sakuga-1TiB">ilovehentai9000/ilove-anime-sakuga-1TiB · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1242441543240126524)** (26 messages🔥): 

- **Experiment with xLSTM sparks curiosity**: A member inquired if anyone had experimented with **xLSTM** yet. This seems to indicate growing interest in less mainstream models.

- **Meta paper brings familiar yet improved content**: Members reviewed a [Meta paper](https://arxiv.org/abs/2309.02591), noting it closely relates to earlier **cm3leon** research but with enhancements. They highlighted interesting advancements in attention mechanisms for scalability.

- **KANs get reviewed**: A member shared a review of **KANs** (Kernel Attention Networks), saying, "Take that KANs", alongside [a link to the review](https://vikasdhiman.info/reviews/KAN_a_review.pdf).

- **Phi-3 Vision chat drives detailed exploration**: Discussion revolved around the **Phi-3 Vision** multimodal model from [Microsoft](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct), with documentation resources included for deeper insight. One user noted how **GPT-4** generated charts sorted by color without changing order, leading to a debate about its purpose.

- **Anthropic scaling paper is heavy reading**: Members talked about the dense content of a recent [Anthropic paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). There was a noted absence of conversations around its implications until now.

**Link mentioned**: <a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1242504991369986199)** (4 messages): 

```html
- **GPT-4o excels at parsing complex documents**: GPT-4o’s multimodal capabilities can efficiently parse complex PDFs and slide decks with background images and irregular layouts into structured markdown. Learn more about this integration with [LlamaParse](https://t.co/g5TG7brSwt) [here](https://t.co/vhtYzsleh2).
- **Sandbox your LLM-generated code with Azure**: Securely execute LLM-generated code using Azure Container Apps dynamic sessions, which is especially useful for tasks that LLMs aren't natively capable of. Discover more details [here](https://t.co/2cnsBH411k) and [here](https://t.co/lTrUPoTMcF).
- **OpenDevin webinar released**: A webinar featuring OpenDevin, an open-source platform for building autonomous AI engineers, has been released. Robert Brennan provides an insightful walkthrough; watch it [here](https://t.co/a22k0zsV3n).
- **Batch inference for GenAI applications**: Use batch inference to preprocess large sets of data, enabling new types of analysis and querying for GenAI applications. Discover the integration details [here](https://t.co/vnuvvypZCz) and [here](https://t.co/M0vQQ1uAki).
```
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1242421332285718528)** (92 messages🔥🔥): 

- **Requests for Document Preview Tutorial**: A member requested a tutorial on getting the document preview for the **llamaindex chat frontend** to work, specifically for getting the URL metadata in the embedding for use by the PDF viewer.
  
- **Errors and Solutions**: Several users encountered errors such as *"ModuleNotFoundError"* and *"pydantic.v1.error_wrappers.ValidationError"*. Solutions involved correcting import paths and removing specific prompts like the *condense_question_prompt*.

- **Concepts and Techniques**: Members discussed **retrievers in LlamaIndex** using cosine similarity and other methods like HNSW for scaling. There were also discussions on **Knowledge Graph Index creation**, referencing embeddings and keyword lookups.

- **Complex Document Handling**: A user shared his ongoing work to create a chatbot assistant to reply accurately within a specific domain, discussing strategies such as post processor reranking and concerns over effective topic restriction.

- **Combining Multiple Indexes**: Queries about combining multiple indexes into one vector index were addressed, with the conclusion that direct combination isn't supported, and one should query each index and accumulate responses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.]">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval/">Ensemble Retrieval Guide - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/discover_llamaindex/document_management/group_conversations/?h=group">Group conversations - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever/?h=group">Recursive Retriever + Query Engine Demo - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/docstore/DocstoreDemo#define-multiple-indexes>)">Docstore Demo - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/?h=github">Github Repo Reader - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/packs/code_hierarchy/?h=code">Code hierarchy - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/?h=group">Knowledge Graph Query Engine - LlamaIndex</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1G6pcR0pXvSkdMQlAK_P-IrYgo-_staxd?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1242387467558977556)** (85 messages🔥🔥): 

- **Types of OpenRouter users debated**: One user humorously pointed out two stereotypical types of OR users: those asking for affectionate interactions with AI and those requesting inappropriate stories, sparking a brief discussion about the prevalence of role-playing apps.
- **Phi-3 Vision Model Introduced**: Information was shared on the **Phi-3 Vision model** available on HuggingFace, emphasizing its high-quality reasoning capabilities and rigorous enhancement processes. [Read more](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) about the model and its documentation.
- **Addressing Wizard's verbosity issues**: Members discussed how **Wizard8x22** struggles with verbosity and improper punctuation, suggesting adjusting the repetition penalty as a potential fix. The discussion branched out to other models like Mixtral and highlighted the variability in model performance.
- **Managing billing errors for student platforms**: A lengthy conversation unfolded regarding a user encountering billing errors while managing their student platform. The exchange culminated in a temporary resolution by deleting and re-entering billing information while expressing hope for future nonprofit discounts.
- **Exploring new LLM interaction techniques**: One user shared their [thread on Twitter](https://x.com/leonjcoe/status/1792946945528320382) about innovative ways of using LLMs through action commands, inviting feedback and experiences from others to expand the discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-4k-instruct">microsoft/Phi-3-medium-4k-instruct · Hugging Face</a>: no description found</li><li><a href="https://x.com/leonjcoe/status/1792946945528320382">Tweet from Leon Builds Agents (@leonjcoe)</a>: There&#39;s a new way of interacting with LLMs that no one is talking about.  Action Commands  So what are they and why are they so valuable? Let me show you
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1242495837771403408)** (14 messages🔥): 

- **Phi-small & Phi-medium models drop**: The release of **Phi-small** and **Phi-medium** models was announced. A discussion followed about whether Phi-Vision is new, with confirmation that Phi-3 Vision is a new, slightly larger version.

- **Meta's 400B model weight concerns**: A [tweet by @apples_jimmy](https://x.com/apples_jimmy/status/1793081686802280576?s=46) claimed Meta might not open the weights for its 400B model, fearing legislation. Another tweet by @q_brabus countered this, stating the **model will remain open-weight** and dismissing the rumor as false.

- **News Corp and OpenAI partnership**: According to [@maxwelltani](https://fxtwitter.com/maxwelltani/status/1793375460879110564), News Corp and OpenAI have announced a historic, multi-year agreement. This deal allows OpenAI to display News Corp's content from WSJ, NY Post, Times/Sunday Times, and more in response to user questions and enhance its products.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/apples_jimmy/status/1793081686802280576?s=46">Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)</a>: Meta plans to not open the weights for its 400B model.  The hope is that we would quietly not notice / let it slide.    Don’t let it slide.</li><li><a href="https://fxtwitter.com/maxwelltani/status/1793375460879110564">Tweet from Max Tani (@maxwelltani)</a>: Inbox: News Corp and OpenAI announce a historic, multi-year agreement to bring News Corp news content to OpenAI, which now has permission to display content from WSJ, NY Post, Times/Sunday Times and m...</li><li><a href="https://x.com/q_brabus/status/1793227643556372596?s=46">Tweet from QBrabus eu/acc (@q_brabus)</a>: @apples_jimmy @ylecun @iamgingertrash Question: Regarding the upcoming LLaMa 3 400B+ model, will it be open-weight? There are several rumors about this...  Answer: No, it is still planned to be open a...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1242484506108366959)** (7 messages): 

- **OpenAI's Superalignment team disbanded over failed commitments**: A [Fortune article](https://fortune.com/2024/05/21/openai-superalignment-20-compute-commitment-never-fulfilled-sutskever-leike-altman-brockman-murati/) reported that OpenAI's Superalignment team, aimed at ensuring AI safety for highly intelligent systems, was disbanded. Despite promising 20% of compute resources, OpenAI failed to meet this commitment, leading to staff resignations.
- **Sam Altman's NDA scandal questioned**: [A new scoop](https://fxtwitter.com/kelseytuoc/status/1793402040439476554?s=46) highlighted that OpenAI's senior leadership claimed ignorance about threats to ex-employees over vested equity, yet documents with their signatures suggest otherwise. [Vox's article](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees) questions whether Sam Altman has been forthcoming regarding the company's NDA practices.
- **Pressure on ex-employees over termination agreements**: Vox’s investigation reveals that OpenAI used tight timelines and significant pushback tactics on ex-employees wanting more time to review complex termination documents. Former employees had only seven days to sign or risk forfeiting potentially millions, with little chance to seek outside counsel.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees">Tweet from Leaked OpenAI documents reveal aggressive tactics toward former employees</a>: Has Sam Altman told the truth about OpenAI’s NDA scandal?</li><li><a href="https://fortune.com/2024/05/21/openai-superalignment-20-compute-commitment-never-fulfilled-sutskever-leike-altman-brockman-murati/">OpenAI promised 20% of its computing power to combat the most dangerous kind of AI—but never delivered, sources say</a>: The company&#x27;s Superalignment team had its requests for computer power repeatedly rejected even though they never approached the 20% threshold, sources say.</li><li><a href="https://fxtwitter.com/kelseytuoc/status/1793402040439476554?s=46">Tweet from Kelsey Piper (@KelseyTuoc)</a>: Scoop: OpenAI&#39;s senior leadership says they were unaware ex-employees who didn&#39;t sign departure docs were threatened with losing their vested equity. But their signatures on relevant documents...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1242483222252884061)** (33 messages🔥): 

- **MSFT Surface Drawing AI slows due to cloud checks**: The new MSFT Surface drawing AI runs locally but experiences latency as it sends safety checks to the cloud. "It’s so dumb," was a user's response to the AI's slow performance.
- **Ben Thompson possibly discussed AI at Microsoft Build**: Members speculated that the source of information about the MSFT Surface drawing AI might be from Ben Thompson's writings or talks at the Microsoft Build event. One user mentioned, "I think Ben Thompson wrote about this today."
- **Discussion about user's past fraudulent colleague**: A user recounted their experience with a colleague who falsely claimed on their resume to have worked with someone the user was collaborating with. This sparked reflections on career trajectories and maturity over time.
- **Anthropic's rapid growth surprises members**: A user expressed astonishment that Anthropic has over 500 researchers now, highlighting the broad use of the "researcher" title. Another member reflected on this by saying, "everyone likes to be called a researcher."
- **Email unsubscriptions and engagement insights**: A user noted a high number of unsubscribes from their email newsletter, attributing it to oversubscription driven by Substack recommendations. They emphasized preferring more engaged subscribers over sheer numbers, noting it's good for disengaged ones to leave.
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1242483369585938462)** (3 messages): 

```html
- **Laughter Ensues**: "lol ugh" conveys a mixture of amusement and exasperation, indicating a humorous but slightly frustrating situation. The follow-up "It’s funny tho" reinforces this sentiment.
- **Footwear Humor**: "He's like the scott galloway of footwear choosers" implies a comparison to Scott Galloway, suggesting someone with a strong, opinionated personality in the context of choosing footwear.
```
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1242852766678781952)** (20 messages🔥): 

- **Nathan Lambert cheers post discussion**: Nathan Lambert was enthusiastic about a recent post, expressing, *"I really liked today's post. I think it's good general audience work"*. Lambert indicated he'd also promote it internally.

- **Digital Celebrities vs. Real Celebrities**: Ashleyduque brought up the potential of digital celebrities outshadowing real ones, mentioning, *"What's keeping us from making and choosing our own voices for assistants and in the future models creating completely digital celebrities?"*. Nathan Lambert responded by agreeing on the attachment to digital figures but expressed regulatory concerns, stating, *"Humans, in reality, will attach to digital celebrities strongly. Idk how you regulate them differently. Scared."*

- **The Future of Hyper-personalized Experiences**: Discussion ensued about whether hyper-personalized experiences will replace shared cultural experiences. Xeophon countered that shared topics create communal bonds, saying, *"Each bubble has its own rockstars... But for this, that <something> has to be the same."*

- **VR and Merch Ideas**: Nathan Lambert shared thoughts on creating branded merchandise like mugs and stickers, humorously saying, *"I need to pump my brand juice lol."* He also expressed a nuanced view on VR, remaining "bullish" on its existence despite potential negative impacts on people.
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1242371364971876412)** (37 messages🔥): 

- **Adding Cohere support to Axolotl**: The ongoing [pull request #1547](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files) is aimed at incorporating Cohere (commandr) into the Axolotl system. This feature has not been tested yet.

- **Tokenizer confusion solved**: A member referred to the `CohereTokenizerFast` documentation to resolve an issue with tokenization. They provided a [link to the GitHub repository](https://github.com/huggingface/transformers/blob/d24097e0229485287ff4959258c552168bd898c6/src/transformers/models/cohere/tokenization_cohere_fast.py#L51C7-L51C26) for reference.

- **Tiny Mistral model found**: Kalomaze located the [tiny Mistral model](https://huggingface.co/openaccess-ai-collective/tiny-mistral/tree/main), which is randomly initialized, to test custom cross-entropy functions. Despite initial confusion, the model fit their requirements.

- **Distillation pipeline progress**: Kalomaze and AMOGUS are working on a distillation pipeline and report that it is "working decently so far". This effort is part of ongoing work with Mistral models.

- **Faster STT to LLM library identified**: The python library creating faster speech-to-text to language model pipelines was identified as [pipecat](https://github.com/pipecat-ai/pipecat). Some members expressed preference for alternatives like OpenVoice or VoiceCraft due to local model support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/openaccess-ai-collective/tiny-mistral/tree/main">openaccess-ai-collective/tiny-mistral at main</a>: no description found</li><li><a href="https://github.com/pipecat-ai/pipecat">GitHub - pipecat-ai/pipecat: Open Source framework for voice and multimodal conversational AI</a>: Open Source framework for voice and multimodal conversational AI - pipecat-ai/pipecat</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: Description  Motivation and Context   How has this been tested?    Untested! Screenshots (if appropriate) Types of changes  Social Handles (Optional)</li><li><a href="https://github.com/huggingface/transformers/blob/d24097e0229485287ff4959258c55">GitHub - huggingface/transformers at d24097e0229485287ff4959258c552168bd898c6</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - GitHub - huggingface/transformers at d24097e0229485287ff4959258c552168bd898c6</li><li><a href="https://github.com/huggingface/transformers/blob/d24097e0229485287ff4959258c552168bd898c6/src/transformers/models/cohere/tokenization_cohere_fast.py#L51C7-L51C26">transformers/src/transformers/models/cohere/tokenization_cohere_fast.py at d24097e0229485287ff4959258c552168bd898c6 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1242385970431660073)** (14 messages🔥): 

- **Full Finetuning vs. LoRA Performance**: One member expressed interest in full finetuning after observing good results with **LoRA** in articles. Another user clarified that full finetuning helps with better retention compared to LoRA, which might be beneficial for style-specific adjustments.
  
- **Inference Configuration Issues**: There was a discussion on the inference command `accelerate launch -m axolotl.cli.inference test_axolotl.yml --lora_model_dir="..."`. It's suggested that this setup might not automatically include chat templates, and it was recommended to manually add them if needed.

- **Config and Documentation Reference**: Members shared a config for full finetuning and mentioned a relevant section in the [Axolotl GitHub README](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenization-mismatch-bw-inference--training), which covers tokenization mismatches between inference and training to help resolve issues.

- **Stable Major Release Inquiry**: A member inquired about the timing of the next stable major release for **Axolotl** but did not receive an immediate response.

- **GPU Memory Requirements for Finetuning**: A user asked about GPU memory requirements for finetuning with a 4090 GPU, specifically mentioning the `examples/mistral/lora.yml` example and encountering `CUDA out of memory errors`. They are seeking guidelines on how to calculate required memory and possible tweaks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenization-mismatch-bw-inference--training">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenizati">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1242736717400768584)** (5 messages): 

- **Setting offload_dir for LoRA merges clarified**: A user asked how to set an `offload_dir` when merging a LoRA model. The response explained that the offloading directory is not directly set during the merge but can be specified manually using the `offload_state_dict` function from the `accelerate` library after the merge, specifying, *"Offload the merged model's state dictionary to the specified directory."*


**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=dce0e2d6-3e84-461f-a383-70860ed4ddfb)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1242469335373709392)** (50 messages🔥): 

- **Langchain JS gets mixed reviews**: A member found **Langchain JS** useful for rapid development, though not as polished as the Python version. They plan to rearchitect in future iterations.

- **Scale AI raises $1B**: [Scale AI](https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/) secured $1 billion, valuing the company at $13.8 billion. Their annual recurring revenue tripled in 2023, and the company anticipates profitability by year-end 2024.

- **Phi 3 model release impresses**: [MS released Phi 3 models](https://x.com/reach_vb/status/1792949163249791383?s=46&t=90xQ8sGy63D2OtiaoGJuww), which are competitive with Mixtral, Llama, and GPT models, featuring 4K and 128K context lengths and a new tokenizer. The performance of these models at their size impresses users, with potential for local running on a MacBook Pro M1 Pro.

- **Anthropic's dictionary learning breakthrough**: [Anthropic](https://x.com/mlpowered/status/1792948212728524917?s=46&t=90xQ8sGy63D2OtiaoGJuww) achieved dictionary learning on a frontier model, enabling millions of feature extractions. This development is poised to advance safety and effectiveness in AI by identifying and manipulating the activation pathways within the model.

- **Humane seeks acquisition post-AI Pin failure**: [Humane is exploring a sale](https://www.theverge.com/2024/5/21/24162185/humane-seeking-acquisition-rumor-ai-pin) for their AI Pin device after poor reviews, with a price target between $750 million and $1 billion. Members discuss the challenges of competing with Apple in hardware and the potential outcomes if the company fails to find a buyer.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mlpowered/status/1792948212728524917">Tweet from Emmanuel Ameisen (@mlpowered)</a>: Today, we announced that we’ve gotten dictionary learning working on Sonnet, extracting millions of features from one of the best models in the world.  This is the first time this has been successfull...</li><li><a href="http://suno.com/blog/fundraising-announcement-may-2024">Suno has raised $125 million to build a future where anyone can make music</a>: Our community of musicians deserves the very best tools, and building the very best tools requires the very best talent. We will use this funding to accelerate product development and grow our world-c...</li><li><a href="https://braindump.me/blog-posts/building-an-ai-game-studio">Building an AI game studio: what we&#x2019;ve learned so far - Braindump Incorporated</a>: create worlds and games using AI</li><li><a href="https://x.com/alexandr_wang/status/1792905417065914858?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alexandr Wang (@alexandr_wang)</a>: 1/ Today, @Scale_AI is announcing $1B of financing at a $13.8B valuation. The round was led by @Accel along with our existing investors.  @Scale_AI has never been better positioned to accelerate the a...</li><li><a href="https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/">Exclusive: Scale AI secures $1B funding at $14B valuation as its CEO predicts big revenue growth and profitability by year-end</a>: Scale AI, which helps companies label and test data for AI model training, has closed a new $1 billion funding round at a $14 billion valuation. </li><li><a href="https://www.theverge.com/2024/5/21/24162185/humane-seeking-acquisition-rumor-ai-pin">Humane is looking for a buyer after the AI Pin’s underwhelming debut</a>: Apparently Humane thinks it’s worth upwards of $1 billion.</li><li><a href="https://x.com/laurentsifre/status/1793045814756921651?s=46&t=90xQ8sGy">Tweet from Laurent Sifre (@laurentsifre)</a>: H</li><li><a href="https://www.anthropic.com/news/mapping-mind-language-model">Mapping the Mind of a Large Language Model</a>: We have identified how millions of concepts are represented inside Claude Sonnet, one of our deployed large language models. This is the first ever detailed look inside a modern, production-grade larg...</li><li><a href="https://x.com/dsiroker/status/1792956339515273537">Tweet from Dan Siroker (@dsiroker)</a>: Lots of folks have asked me about Microsoft Recall so here’s my take!</li><li><a href="https://news.ycombinator.com/item?id=40429326">no title found</a>: no description found</li><li><a href="https://x.com/thesephist/status/1793031719244734923">Tweet from Linus (@thesephist)</a>: By end of 2024, steering foundation models in latent/activation space will outperform steering in token space (&#34;prompt eng&#34;) in several large production deployments.  I felt skeptical about th...</li><li><a href="https://x.com/mlpowered/status/1792948212728524917?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Emmanuel Ameisen (@mlpowered)</a>: Today, we announced that we’ve gotten dictionary learning working on Sonnet, extracting millions of features from one of the best models in the world.  This is the first time this has been successfull...</li><li><a href="https://x.com/laurentsifre/status/1793045814756921651?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Laurent Sifre (@laurentsifre)</a>: H</li><li><a href="https://x.com/alexalbert__/status/1792936647665107108?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alex Albert (@alexalbert__)</a>: Our new interpretability paper offers the first ever detailed look inside a frontier LLM and has amazing stories. I want to share two of them that have stuck with me ever since I read it.  For backgro...</li><li><a href="https://x.com/stephenlcasper/status/1793014675237638341?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Cas (Stephen Casper) (@StephenLCasper)</a>: 🧵On May 5, I made 10 predictions about what the next SAE paper from Anthropic would and wouldn’t do. I went 10 for 10...  https://x.com/StephenLCasper/status/1787270794017702045  Quoting Cas (Stephen...</li><li><a href="https://x.com/reach_vb/status/1792949163249791383?s=46&t=90xQ8s">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: LETS GOO! Phi 3 - Small, Medium & Vision are out! 🔥  &gt; Medium competitive with Mixtral 8x22B, Llama 3 70B & beats Command R+ 104B & GPT 3.5 &gt; Small beats Mistral 7B & Llama 3 8B &gt; 4K & 128K ...</li><li><a href="https://x.com/reach_vb/status/1792949163249791383?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: LETS GOO! Phi 3 - Small, Medium & Vision are out! 🔥  &gt; Medium competitive with Mixtral 8x22B, Llama 3 70B & beats Command R+ 104B & GPT 3.5 &gt; Small beats Mistral 7B & Llama 3 8B &gt; 4K & 128K ...</li><li><a href="https://youtu.be/uHEPBzYick0?si=ajbDL9agnubNAECO&t=203">Microsoft vs. Apple: Satya Nadella Says AI-Focused Copilot+ PCs Beat Macs | WSJ</a>: Microsoft’s new Copilot+ PCs with Qualcomm chips and AI Windows features aim to beat Apple’s MacBooks. WSJ’s Joanna Stern tried out the new laptops and sat d...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1242620273086304356)** (1 messages): 

- **Join the Survey Paper Club for quick paper insights**: *For those new, we have a survey tomorrow - a very nice way to get quick hits of a few papers in one hour*. [Sign up here for notifications](https://lu.ma/e5nk2ebp).

**Link mentioned**: <a href="https://lu.ma/e5nk2ebp">LLM Paper Club (Survey Paper Club!) · Zoom · Luma</a>: It&#x27;s survey day! Pick a paper from here and cover it in 5 minutes: https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions

  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1242915170003456151)** (4 messages): 

- **Zoom Link Sent via Email**: Members were questioning where the Zoom link for the meeting would be provided. They were directed to register [here](https://lu.ma/e5nk2ebp), where the link would be sent to their email each week.

**Link mentioned**: <a href="https://lu.ma/e5nk2ebp">LLM Paper Club (Survey Paper Club!) · Zoom · Luma</a>: It&#x27;s survey day! Pick a paper from here and cover it in 5 minutes: https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1242394632285847642)** (36 messages🔥): 

- **LangChain vs LangChain_Community**: Members discussed the architectural differences between **LangChain** and **LangChain Community**. Key parts and integrations were explained with references to [LangChain documentation](https://python.langchain.com/v0.2/docs/concepts/#architecture).

- **Chaining LangChain Models**: A user asked about chaining LangChain models, describing a specific scenario and sharing the [LangChain Cookbook tutorial](https://youtu.be/2xxziIWmaSA?si=3wkNt_huJKu3xK3t&t=1694). Members suggested how to handle variable name consistency across chains.

- **Pluto for Cloud Deployment**: A member introduced a PR to incorporate **Pluto** as a deployment option for **LangServe** apps on the cloud. They also shared a [sample QA assistant](https://xw3vdvjmyp7jig7tmrvrqbisiu0peosf.lambda-url.us-east-1.on.aws/) and an explanatory [article](https://pluto-lang.vercel.app/cookbook/rag-qa-bot-with-web).

- **Distributed Ingestion Using Ray**: A question was raised about using **Ray and LangChain** for distributed ingestion of data, but it was noted that resources from the Ray team are outdated. The community did not provide a definitive solution.

- **Plan-and-Execute Example Issue**: A user reported issues with the **plan-and-execute** example from **langgraphjs** and mentioned specific package versions to get it working. The example's compatibility with Node was in question, but specific version adjustments helped resolve the errors.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1038097195422978059/1242839942921584760">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#architecture">Conceptual guide | 🦜️🔗 LangChain</a>: This section contains introductions to key parts of LangChain.</li><li><a href="https://youtu.be/2xxziIWmaSA?si=3wkNt_huJKu3xK3t&t=1694">The LangChain Cookbook - Beginner Guide To 7 Essential Concepts</a>: Twitter: https://twitter.com/GregKamradtNewsletter: https://mail.gregkamradt.com/signupCookbook Part 2: https://youtu.be/vGP4pQdCocwWild Belle - Keep You: ht...</li><li><a href="https://github.com/langchain-ai/langgraphjs/blob/main/examples/plan-and-execute/plan-and-execute.ipynb?ref=blog.langchain.dev)">langgraphjs/examples/plan-and-execute/plan-and-execute.ipynb at main · langchain-ai/langgraphjs</a>: ⚡ Build language agents as graphs ⚡. Contribute to langchain-ai/langgraphjs development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/discussions/22006#discussioncomment-9515272">KeyError: &quot;Input to ChatPromptTemplate is missing variables {&#39;agent_scratchpad&#39;}.   and TypeError: string indices must be integers -&gt; format_to_openai_function_messages( x[&quot;intermediate_steps&quot;]  ), · langchain-ai/langchain · Discussion #22006</a>: Checked other resources I added a very descriptive title to this question. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1242905562526646343)** (1 messages): 

- **Bug in LangServe's 'invoke' endpoint sparks discussions**: Users have reported a pervasive issue with LangServe's 'invoke' endpoint, which fails to return all outputs from the retrieval chain. Instead of including context and source documents, it only returns question and answer pairs with no documents retrieved. [Link to discussion](https://github.com/langchain-ai/langserve/discussions/461).

- **Empty output issue on 'invoke' route**: Another user shared a related issue where the 'invoke' route returns an empty output, while the streaming functionality works correctly. This discrepancy is causing challenges in applications that rely on the 'invoke' endpoint for comprehensive responses. [Link to issue](https://github.com/langchain-ai/langserve/issues/301).

- **RemoteRunnable vs. RunnableWithMessageHistory**: A problem was highlighted where the RemoteRunnable component fails to return source documents, unlike its counterpart, RunnableWithMessageHistory, which performs correctly. This inconsistency affects the reliability of hosted scripts in returning expected sources for question-answering chains. [Link to issue](https://github.com/langchain-ai/langserve/issues/618).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langserve/discussions/461">Getting no documents retrieved with retrieval chain · langchain-ai/langserve · Discussion #461</a>: hello, I&#39;am newest in langchain and langserver. I think that there is a problem when returning meta data. The API return only the answer and the question hereis my server code from fastapi import ...</li><li><a href="https://github.com/langchain-ai/langserve/issues/301">LangServe: &#39;invoke&#39; Route Returns Empty Output, While Streaming Works · Issue #301 · langchain-ai/langserve</a>: I&#39;m building a very simple LangChain application that takes as an input a customer feedback string and categorizes it into the following pydantic class: class AnalysisAttributes(BaseModel): overal...</li><li><a href="https://github.com/langchain-ai/langserve/issues/618">RemoteRunnable doesn&#39;t return sources but RunnableWithMessageHistory does · Issue #618 · langchain-ai/langserve</a>: Overview I have developed a chain for question answering, which functions correctly when run as an independent Python script, returning sources as expected. However, when this script is hosted on a...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1242528634837794836)** (3 messages): 

- **Chat with your PDFs using Upstage AI Solar models**: Check out this blog post on [creating a PDF query assistant](https://medium.com/@sonam.gupta1105/creating-a-pdf-query-assistant-with-upstage-ai-solar-and-langchain-integration-6631280093b5). The author explains how they leveraged the upcoming **Solar LLM from Upstage AI** and integrated it with **LangChain** to answer questions based on PDFs.

- **Simplify AWS Deployment with LangServe**: Learn how to deploy a LangServe app on AWS *without* needing to login to the AWS console or understand complex cloud technologies like Terraform, Pulumi, or AWS CDK. Read more in the detailed guide on [Medium](https://medium.com/aimonks/deploy-langserve-application-to-aws-2d34b6ee5c1a).

- **Build a Web Interface Document Q&A Bot in 5 Minutes**: **Construct your own document Q&A bot** using LangChain, FastUI, and Pluto directly from your GitHub documentation repository. Find the step-by-step process in this [AWSTip article](https://awstip.com/craft-a-document-qa-assistant-for-your-project-in-just-5-minutes-cccf1002a0af).

**Link mentioned**: <a href="https://medium.com/@sonam.gupta1105/creating-a-pdf-query-assistant-with-upstage-ai-solar-and-langchain-integration-6631280093b5">Creating a PDF Query Assistant with Upstage AI Solar and LangChain Integration</a>: Do you ever feel overwhelmed by the numerous research papers you need to read? As someone who just finished a PhD, I know it’s no walk in…

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1242415938658238555)** (23 messages🔥): 

- **Discussing Development Setups:** Members discussed how Open Interpreter accesses and reviews their file systems, with specific setups involving tools like Boox E Ink tablets for reading and note-taking, OneNote for typed notes, and VSCode for development. One member said, *"a typical use case is sending a 'link' from one source to reference another."*
  
- **Daily Uses and Complex Problems with Open Interpreter:** A member asked what others are using Open Interpreter for daily, looking for success stories, particularly in bridging different devices. They mentioned using it to ask questions about code or papers directly without switching to a browser.

- **Issues with GPT-4o Integration:** Members shared their experiences and issues with setting up GPT-4o with Open Interpreter, including error messages related to API keys. One member noted that GPT-4o is significantly faster, *"like 5x speed minimum."*

- **Text Formatting Problems in Models:** A member reported issues with models like Gemini 1.5 and Gemini Flash inserting unnecessary newline characters in code blocks, which affects code execution. They also inquired whether missing "python" declarations were part of the problem.

- **Concerns over AI Legislation:** A link to a controversial AI bill in California was shared, prompting concerns among members. The bill pertains to the responsible development of AI frontier models, with one member highlighting an [open letter](https://x.com/Scott_Wiener/status/1792572175116816853) released by a lawmaker to address misconceptions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Scott_Wiener/status/1792572175116816853">Tweet from Senator Scott Wiener (@Scott_Wiener)</a>: In recent weeks, there&#39;s been a flurry of discussion online about SB 1047, my bill on responsible development of the largest & most powerful AI frontier models. We’ve heard some incredibly thought...</li><li><a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: The open-source language model computer</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1242874145943846972)** (2 messages): 

- **Bill Gates envisions smarter AI interfaces**: In a [Bill Gates article](https://www.gatesnotes.com/AI-agents), he discusses how current software, while improved, remains limited in integrating tasks across different apps. Gates predicts a future where devices will understand and execute tasks from a single directive in everyday language, akin to the assistance from a close friend.

- **Bypass macOS ChatGPT app waitlist**: A workaround to skip the waitlist for the ChatGPT macOS app was shared on [Twitter](https://x.com/testingcatalog/status/1793347117458636981). The steps involve timing the CMD+Q command during the login process to gain immediate access.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.gatesnotes.com/AI-agents">AI is about to completely change how you use computers | Bill Gates</a>: no description found</li><li><a href="https://x.com/testingcatalog/status/1793347117458636981">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: It turns out that you can easily bypass the macOS ChatGPT app waitlist in this way:  1. Launch the app and log in 2. Press CMD+Q when the window changes its size but before the Login alert. 3. Launch ...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1242947436905893970)** (7 messages): 

- **Redefining the wheel in trigonometry unnecessary**: A participant voiced their concern about *"trying to reinvent things that already exist,"* especially regarding Taylor series and their limitations around specific points.

- **Alternative interval reductions suggested**: Another point highlighted was the arbitrary nature of range reduction to [0, pi/2], noting it can also be [0, pi/4], but emphasizing it doesn't solve the core problem of achieving perfect accuracy with minimal computation.

- **IBM's practical approach to interval partitioning**: It was mentioned that practical implementations typically involve partitioning intervals, such as [0, pi/2], to find perfect approximations, underscoring that this is already a solved problem.

- **Sharing IBM's sine function implementation**: An [IBM implementation of the sine function](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/s_sin.c;hb=HEAD) was shared, noting that the effort needed for perfect accuracy depends on the specific types involved.

- **Range reduction complexities and solutions**: A link to [another IBM implementation dealing with range reduction issues](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/branred.c;hb=HEAD) was shared, noting that while the process is complicated, it’s only necessary for very large numbers and doesn’t slow things down normally.
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1242819347806818375)** (10 messages🔥): 

- **Track gradients in tinygrad like a pro**: In tinygrad, you can use `with Tensor.train():` to start tracking gradients, or set `Tensor.no_grad = True/False` to stop/start gradient tracking mid-code. A helpful [example from the repo](https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e46c6e1594c2dcc/examples/beautiful_cartpole.py#L58) illustrates its use.

- **Set training mode manually in tinygrad**: It was clarified that the `Tensor.train` decorator simply sets `Tensor.training` under the hood. You can manually set `Tensor.training` as needed, as shown in this [code snippet](https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e46c6e1594c2dcc/tinygrad/tensor.py#L83).

- **Decorator for `no_grad`**: There’s a decorator version for inference mode, `Tensor.inference_mode()`, that acts similarly to `no_grad`. This provides a cleaner syntax for temporarily disabling gradient tracking.

- **Understanding movement op optimizations**: Discussed the behavior of chaining movement ops and noted that multiple views are rare but possible with specific combinations. For example, using `ShapeTracker.from_shape((3, 4)).permute((1, 0)).reshape((3, 4))` can result in multiple views.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e4">GitHub - tinygrad/tinygrad at d12d412e8b0c900681e9d6c39e46c6e1594c2dcc</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - tinygrad/tinygrad at d12d412e8b0c900681e9d6c39e46c6e1594c2dcc</li><li><a href="https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e46c6e1594c2dcc/examples/beautiful_cartpole.py#L58">tinygrad/examples/beautiful_cartpole.py at d12d412e8b0c900681e9d6c39e46c6e1594c2dcc · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e46c6e1594c2dcc/examples/beautiful_cartpole.py#L82">tinygrad/examples/beautiful_cartpole.py at d12d412e8b0c900681e9d6c39e46c6e1594c2dcc · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/d12d412e8b0c900681e9d6c39e46c6e1594c2dcc/tinygrad/tensor.py#L83">tinygrad/tinygrad/tensor.py at d12d412e8b0c900681e9d6c39e46c6e1594c2dcc · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1242451432444006400)** (12 messages🔥): 

- **Supervised Fine-Tuning vs Preference Optimization**: Discussing the fundamental difference between **Supervised Fine-Tuning (SFT)** and **Preference Optimization**, a member noted, "SFT pushes up the probability distribution in the model of data points in the SFT dataset," while preference optimization also pushes down probabilities of undesired outputs. They questioned the exclusive use of SFT when preference optimization seems more comprehensive.

- **Phi3 Vision impresses with 4.2b params**: A member shared their excitement for **Phi3 Vision**, a model with just 4.2 billion parameters, and described it as a breakthrough for low-latency/live inference on image streams. *"Just imagine what even smaller/more specialized versions of this will enable in robotics,"* they added ([link](https://x.com/jphme/status/1792950682695479734)).

- **Comparison of Moondream2 and Phi3 Vision**: Members compared the performance of **Moondream2** and **Phi3 Vision** on image tasks. One noted, *"Vik tried to reduce hallucinations. Some datasets are a bit bad in that regard."* ([Moondream2](https://huggingface.co/spaces/vikhyatk/moondream2)).

- **New Microsoft Model Releases**: Announcements of new **Microsoft 7b** and **14b** Instruct models led to mixed reactions. One member pointed out the 14b instruct version's poor German performance, while another highlighted its potential in extractive tasks and complex reasoning.

- **Concerns about Meta's 400b Model Rumors**: A member expressed interest in rumors that **Meta may not publish a 400b model** as open source. They noted that most threads cited an unreliable source named Jimmy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/vikhyatk/moondream2">moondream2 - a Hugging Face Space by vikhyatk</a>: no description found</li><li><a href="https://x.com/jphme/status/1792950682695479734">Tweet from Jan P. Harries (@jphme)</a>: Phi3 vision was just released - it is just 4.2b params and extremely impressive. 🤩  I feel this is a breakthrough for low-latency/live inference on image streams - just imagine what even smaller/more...
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1242590720175378512)** (8 messages🔥): 

- **Join Cohere's Team**: A member excitedly shares a [link to Cohere's careers page](https://cohere.com/careers), encouraging others to apply. They emphasize the company's focus on solving real-world problems with cutting-edge ML/AI technologies.
- **Confusion with Link Access**: Someone mentioned they couldn't find the page when trying to access the link provided for Cohere's careers.
- **LLM Model VRAM Usage**: A member shares a [link to an LLM-Model-VRAM-Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) on Hugging Face and asks for an explanation on why Phi 3 Mini uses more VRAM than Phi 3 Medium for the same context length.
- **BotPress Command-R Integration**: A user seeks a tutorial on how to incorporate Command-R into BotPress, asking for help in both English and Spanish.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://cohere.com/careers">Careers</a>: Our team of ML/AI experts is passionate about helping developers solve real-world problems. From our offices in Toronto, London, and Palo Alto, we work at the cutting edge of machine learning to unloc...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1242870724708929688)** (1 messages): 

```html
- **Seeking Command-R tutorial for BotPress**: A member asked for a tutorial on how to incorporate **Command-R** into **BotPress**. They repeated the request in both English and Spanish: *"Does anyone have a tutorial on how to incorporate Command-R into BotPress? Alguien tiene un tutorial de como incorporar Command-R en BotPress?"*
```
  

---


### **Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1242870740345294998)** (1 messages): 

- **Seeking Command-R tutorial for BotPress**: A member inquired if anyone has a tutorial on how to incorporate **Command-R** into **BotPress**. They asked for resources or guidance in both **English and Spanish**.
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1242531297750941899)** (7 messages): 

- **AI Waifus save lives**: A user humorously declared, *"AI waifus save lives!"* sparking a brief banter among members with another replying, *"Just monika."*
- **VentureBeat article on Emotional AI**: A member shared a **VentureBeat article** discussing plans to embed emotional AI in business bots, questioning, *"Will waifus soon be able to 'understand' and process emotions?"* Read the article [here](https://venturebeat.com/ai/exclusive-inflection-ai-reveals-new-team-and-plan-to-embed-emotional-ai-in-business-bots).
- **3D Character Chatbots at 4Wall AI**: Another member mentioned they are working on **3D character chatbots** at **4Wall AI** and promoted a teaser available on another channel, <#1122748840819306598>.
- **Re: Just Monika**: In response to *“Who dat?”* about the "Just Monika" reference, a user provided a GIF link for context found [here](https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242).

**Link mentioned**: <a href="https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242">Ddlc Doki Doki Literature Club GIF - Ddlc Doki Doki Literature Club Just Monika - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1242549498220122203)** (5 messages): 

- **Qualcomm unveils Snapdragon Dev Kit for Windows**: Qualcomm has released a new developer kit featuring their most powerful Snapdragon X Elite chip, priced at $899.99. It's touted as a Mac Mini competitor with 32GB of LPDDR5x RAM, 512GB of NVMe storage, and numerous ports, ideal for long-lasting, powerful Windows laptops with Arm chips [more details on The Verge](https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite).

- **Windows Dev Kit pricing complaints**: One user expressed interest in the new Snapdragon Dev Kit but felt the $900 price tag was steep, especially compared to last year’s model priced at $600. They noted its suitability for Arm development with 32GB RAM and 512GB storage for various developer workloads [more details](https://www.microsoft.com/en-us/d/windows-dev-kit-2023/94k0p67w7581?activetab=pivot:overviewtab).

- **Using Mac Mini for Llamafile server**: A user shared their positive experience using a Mac Mini as a long-running Llamafile server, accessible through Tailscale. They appreciated its zero-cold start and compatibility with the `llm` CLI.

- **Hope for more affordable, aesthetic dev kits**: Another user hoped for cheaper models in the future while expressing a desire for a translucent case design.

- **Smalltalk experiment with Claude**: Highlighting a proof of concept, one user shared an example of Claude engaging in Smalltalk by answering the question "What are frogs?" with a basic explanation of amphibious animals.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite">Here’s the eight-inch Snapdragon PC for your Windows on Arm experiments</a>: Qualcomm is selling it in black.</li><li><a href="https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-e">Here’s the eight-inch Snapdragon PC for your Windows on Arm experiments</a>: Qualcomm is selling it in black.</li><li><a href="https://www.microsoft.com/en-us/d/windows-dev-kit-2023/94k0p67w7581?activetab=pivot:overviewtab">Buy Windows Dev Kit 2023 Desktop PC for Arm App Developers - Microsoft Store</a>: Build, debug, and test native Windows apps for Arm with Windows Dev Kit 2023, a compact desktop computer engineered for developer workloads.
</li>
</ul>

</div>
  

---



### **LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1242577685906063431)** (2 messages): 

- **Llama3/Phi3 truncates responses**: A member asked for help on how to prevent **llama3/phi3** from hitting them with "*additional items omitted for brevity*". No further discussion or solutions were presented.
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1242640452038037535)** (1 messages): 

- **Member-Organized Events Kick Off**: The first of **member-organized events** includes talks, AMAs, demos, and discussions. These events are designed to promote cross-pollination of ideas and foster community engagement.

- **LLM360 Hosts AMA**: [LLM360](https://www.llm360.ai/) kicks off the series with an [AMA highlighting their work](https://discord.com/events/1089876418936180786/1240722407594004561) in open-source LLMs.

- **Kate Silverstein's Demo and Blog Post**: Staff Machine Learning Engineer Kate Silverstein will share a [demo using llamafiles for embeddings](https://discord.com/events/1089876418936180786/1242590711778381914) and chat about her [recent blog post](https://discord.com/channels/1089876418936180786/1242235316170129439).

- **Events Calendar**: Members are encouraged to regularly *delve* into the events calendar for more activities and opportunities to participate in the community events.
  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1242844126941155359)** (1 messages): 

- **Clarifying model usage in Python example**: A member asked for clarification on whether they need to specify a model under `model="LLaMA_CPP"` when running a tinyllama model from the terminal. They provided a code snippet and mentioned that the code works but are unsure which model is used.

**Link mentioned**: <a href="http://<Your">no title found</a>: no description found

  

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
