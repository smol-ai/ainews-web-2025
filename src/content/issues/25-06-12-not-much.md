---
id: MjAyNS0w
title: not much happened today
date: '2025-06-12T05:44:39.731046Z'
description: >-
  **Bytedance** showcased an impressive state-of-the-art video generation model
  called **Seedance 1.0** without releasing it, while **Morph Labs** announced
  **Trinity**, an autoformalization system for Lean. **Huggingface
  Transformers** deprecated Tensorflow/JAX support. **Andrew Ng** of
  **DeepLearning.AI** highlighted the rise of the **GenAI Application Engineer**
  role emphasizing skills in **AI building blocks** and **AI-assisted coding
  tools** like **Codex** and **Claude Code**. Engineering teams are increasingly
  testing API designs against LLMs for usability. **Figure AI**'s CEO stressed
  speed as a key competitive advantage, and **LangChain** introduced the concept
  of **Context Engineering** for AI agents. Reinforcement learning on LLMs shows
  transformative potential, and the community values **AI evals** and data work.
  **Sakana AI** released **Text-to-LoRA**, a hypernetwork method for generating
  task-specific LoRA adapters from natural language, enabling efficient model
  customization. The video generation race heats up with **Bytedance**'s
  Seed-based model praised for quality, challenging American labs, alongside
  models like **Kling 2.1** and **Veo 3**.
companies:
  - bytedance
  - morph-labs
  - huggingface
  - deeplearning.ai
  - figure-ai
  - langchain
  - sakana-ai
models:
  - seedance-1.0
  - codex
  - claude-code
  - kling-2.1
  - veo-3
topics:
  - video-generation
  - autoformalization
  - ai-assisted-coding
  - api-design
  - context-engineering
  - reinforcement-learning
  - ai-evals
  - hypernetworks
  - model-fine-tuning
  - foundation-models
people:
  - andrew_ng
  - hwchase17
  - adcock_brett
  - clementdelangue
  - akhaliq
  - jxmnop
  - hamelhusain
  - sh_reya
---


a quiet day

> AI News for 6/11/2025-6/12/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (218 channels, and 7130 messages) for you. Estimated reading time saved (at 200wpm): 579 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Bytedance showed off, but did not release, an impressive [SOTA videogen model called Seedance 1.0](https://seed.bytedance.com/en/seedance), Morph Labs announced [Trinity](https://x.com/morph_labs/status/1933181394588483868?s=46), an autoformalization system for Lean, and Huggingface Transformers [deprecated Tensorflow/JAX](https://x.com/LysandreJik/status/1933201171130593530).

---

# AI Twitter Recap

**AI Engineering Skills, Roles, and Development Philosophy**

- **The Rise of the GenAI Application Engineer**: In a detailed thread, **Andrew Ng** of [**DeepLearning.AI**](http://deeplearning.ai/) outlines the key skills for the emerging role of a **GenAI Application Engineer**. He emphasizes two criteria: the ability to use new **AI building blocks** (like RAG, agentic frameworks, evals, MCPs) to build powerful applications, and the ability to use **AI-assisted coding tools** like **Codex** and **Claude Code** for rapid engineering. [Ng notes that while AI building blocks have a longer shelf-life, AI-assisted coding techniques become obsolete much faster](https://twitter.com/AndrewYNg/status/1933185193059516442), making the ability to continuously learn a highly predictive skill for success.
- **Designing APIs for LLMs**: A growing trend observed by [@alexalbert__/](https://twitter.com/alexalbert__/status/1933177502777913596) is that engineering teams at large companies are now **testing their API designs against LLMs** before release. They run evaluations to see which API structures are easiest for models to use, suggesting a future where software is designed with models as the primary user.
- **The Need for Speed in Development**: [@adcock_brett](https://twitter.com/adcock_brett/status/1933226344156221746), CEO of **Figure AI**, argues that **speed is the ultimate advantage** and moat in tech. He reflects that his team made **5-7 years of progress in the last 3 years** by prioritizing speed over perfection, which builds momentum and focus.
- **The Importance of "Context Engineering"**: **LangChain's** [@hwchase17](https://twitter.com/hwchase17/status/1933278290992845201) highlights the concept of **"Context Engineering"** as the next level of prompt engineering. He defines it as the process of dynamically and automatically providing a system with the necessary context, calling it the "**#1 job of engineers building AI agents**."
- **RL's Transformative Potential**: [@jxmnop](https://twitter.com/jxmnop/status/1933359925415325980) notes that it's becoming clear what incredible possibilities open up when **reinforcement learning (RL) on LLMs works**, suggesting "we’re just getting started."
- **The Value of Evals and Data Work**: While acknowledging that [**eval work and staring at data** are "incredibly important and incredibly boring,"](https://twitter.com/finbarrtimbers/status/1933278968859468161) the community emphasizes their necessity. A popular course on **AI Evals** by [@HamelHusain](https://twitter.com/HamelHusain/status/1932964208100180239) and [@sh_reya](https://twitter.com/HamelHusain/status/1932964208100180239) is frequently mentioned as a key resource for engineers and PMs to master this critical skill.

**Model & Research Breakthroughs**

- **Text-to-LoRA from Sakana AI**: **Sakana AI** introduced **Text-to-LoRA (T2L)**, a novel technique using a **hypernetwork** to generate task-specific **LoRA adapters** directly from a natural language description of a task. The approach meta-learns from hundreds of existing LoRAs, enabling rapid, parameter-efficient customization of foundation models without large datasets or expensive fine-tuning. [The announcement states T2L can generalize to unseen tasks and lowers the barrier for non-technical users to specialize models](https://twitter.com/SakanaAILabs/status/1932972420522230214). The release was met with excitement, with **Hugging Face's** [@ClementDelangue](https://twitter.com/ClementDelangue/status/1932977773582106973) simply exclaiming, "Text to models!"
- **The Video Generation Race**: A **ByteDance** model based on the **Seed** architecture is being praised for its quality, with [@scaling01](https://twitter.com/scaling01/status/1933048431775527006) claiming it "**destroys Veo 3**" and questioning if American labs can compete. This follows **Kling AI** sharing generations from its **Kling 2.1** model, and [@_akhaliq](https://twitter.com/_akhaliq/status/1933069477807337771) showcasing a **Veo 3** video of a polar bear explaining the Concorde's failure. Meanwhile, **ByteDance** also introduced **APT2**, an autoregressive adversarial post-training method for real-time interactive video generation.
- **Eliciting Latent Capabilities from Pretrained Models**: New research from **Anthropic**, shared by [@jeremyphoward](https://twitter.com/jeremyphoward/status/1932959121915195842), demonstrates how to **elicit capabilities from pretrained models using no external supervision**. The resulting models are often competitive with or even superior to SFT models on tasks like math and coding. [@jiaxinwen22](https://twitter.com/jeremyphoward/status/1933364618371739948) clarified this is about elicitation, not self-improvement.
- **Meta's V-JEPA 2 World Model**: [@omarsar0](https://twitter.com/omarsar0/status/1932993784683303272) shared **Meta's** release of **V-JEPA 2**, a new world model aimed at accelerating physical AI by learning from video to understand and predict the physical world.
- **Model Merging in Pretraining**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1933255559668772941) suggests that **model merging during pretraining** is one of the most underdiscussed and understudied aspects of current foundation model training in high-compute environments.
- **Weekly Model Roundup**: [@mervenoyann](https://twitter.com/mervenoyann/status/1933101803274477600) provided a weekly summary of open model releases, including **Alibaba's Qwen3-Reranker-4B** and **Qwen3-Embedding** models, **OpenBMB's MiniCPM4** family, **Arcee AI's Homunculus 12B**, **MonkeyOCR** for document parsing, **NVIDIA's Llama-3.1-Nemotron-Nano-VL-8B-V1**, and **ByteDance's ContentV-8B** video model.
- **Mind-Reading Benchmark for Imagination**: Researchers at **UMN** have created the [first benchmark dataset for decoding mental images directly from imagination via fMRI](https://twitter.com/iScienceLuvr/status/1932945933521817988), moving beyond reconstructing what a person is actively seeing.

**Tooling, Frameworks, and Integrations**

- **Hugging Face Transformers Deprecates TensorFlow and Flax**: In a major ecosystem shift, **Hugging Face** announced that its popular `transformers` library will become **PyTorch-only**, [deprecating support for **TensorFlow** and **Flax**](https://twitter.com/_lewtun/status/1933226225620885818). The team cited the high maintenance burden and the desire to reduce library bloat as key reasons for the change.
- **LangGraph Powers Enterprise AI Agents**: **LangChain** showcased how the **$11 trillion asset manager BlackRock** built its **Aladdin Copilot** orchestration system on **LangGraph**, [supporting over 4,000 engineers across 100+ applications](https://twitter.com/LangChainAI/status/1933216936730722794). They also announced a new **LangGraph integration with Tensorlake**, a document ingestion engine, to improve agentic understanding of data.
- **Perplexity Integrates with Fireflies for Meetings**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1933248190326976542) announced that **Perplexity** can now be used on video calls through an integration with [**Fireflies.ai**](http://fireflies.ai/), bringing its search and reasoning capabilities to meetings.
- **Runway Introduces "Chat Mode"**: **Runway** launched **Chat Mode** for its **Gen-4** model, providing a new conversational interface to generate images and videos. Co-founder [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1933238580400537698) explained this is a step towards adaptive interfaces that make media generation more natural and intuitive.
- **The "Cursor + Claude Code" Stack**: There is significant praise for the developer experience of using the **Cursor IDE** in conjunction with **Anthropic's Claude Code**. Users like [@cloneofsimo](https://twitter.com/cloneofsimo/status/1933177834119610427) report a drastic increase in productivity, while a **Y Combinator** podcast [featured Anysphere CEO Michael Truell discussing the product's vision](https://twitter.com/dilipkay/status/1933099751370613185).
- **Instagram's 3D Photo Integration**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1933199948759146810) noted that the beta **3D photo integration with Instagram** is "very well done," turning static photos into AI-generated stereoscopic images. He sees this as a stepping stone towards full 6DOF model generation.
- **TorchAO Enables FP8 for RTX 4090**: [@RisingSayak](https://twitter.com/RisingSayak/status/1933187476509917471) shared that **TorchAO** now supports **FP8** for **SM89** architecture GPUs like the **RTX 4090**, showing significant speedups for models like **Flux**.
- **UnslothAI for Reward Model Serving**: [@danielhanchen](https://twitter.com/danielhanchen/status/1932965003621204391) announced that using **UnslothAI** can achieve **2x faster inference** for reward model serving and sequence classification.

**Infrastructure, Industry Events & Funding**

- **Major Cloud Outage Hits AI Services**: A massive internet outage, seemingly originating from cloud providers like **GCP** and **Cloudflare**, [caused widespread disruption across the AI ecosystem](https://twitter.com/gregisenberg/status/1933242926337077272). **OpenAI** reported issues affecting SSO and login methods before [announcing a full recovery](https://twitter.com/OpenAI/status/1933260549045039549). Other affected services included **Weights & Biases**, **LangSmith**, **Replit**, and **Cursor**. The event prompted commentary from **DHH** on the dangers of [cloud concentration, which he argued "wrecked the internet's primary design goal: aggregate resilience"](https://twitter.com/vikhyatk/status/1933258625327509646).
- **Perplexity Labs Presented by Jensen Huang at GTC Paris**: **Perplexity's** CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1932968936938537223) shared a photo of **NVIDIA CEO Jensen Huang** presenting **Perplexity Labs** at the **GTC** event in Paris.
- **Sam Altman and Lisa Su at AMD's Advancing AI Event**: **OpenAI CEO Sam Altman** appeared at **AMD's #AdvancingAI** keynote alongside **AMD CEO Dr. Lisa Su**, [as shared by Lamini's Sharon Zhou](https://twitter.com/realSharonZhou/status/1933231029516648554).
- **Google's Open Source Push**: **Google's Jeff Dean** highlighted that [**Google has released 999 open models** on Hugging Face](https://twitter.com/ClementDelangue/status/1933107694585487803), a significant contribution to the open-source community. **Hugging Face CEO Clément Delangue** noted this compares to 387 from Meta and 250 from Microsoft.
- **Funding Momentum**: [@scottastevenson](https://twitter.com/scottastevenson/status/1933117996068905457) announced receiving **four term sheets in two weeks** before returning his focus to building.
- **Perplexity Teases "Comet" Release**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1933289407705960697) stated that **Perplexity's** upcoming product, **Comet**, is "peerless" and that more invites will be going out as it nears its final testing stage.

**Geopolitics, Critiques, and Broader Commentary**

- **Ambition in the AI Race**: **Perplexity's** [@AravSrinivas](https://twitter.com/AravSrinivas/status/1933283015586951623) posted an aspirational take, stating "**Google showed the world** you can have your own search, AI, data center, chip, phone, OS, browser...So, don’t aim low. Be ambitious."
- **Critique of Modern Model Capabilities**: [@corbtt](https://twitter.com/corbtt/status/1932977024882389253) expressed frustration with the current state of LLMs, asking "**why are modern models still so bad at writing?**" and noting their inability to summarize a blog post without producing "emoji-overloaded slop." In a similar vein, [@teortaxesTex](https://twitter.com/teortaxesTex/status/1933371065863909638) found it "pretty embarrassing for OpenAI" that o3 could be defeated by simple trick questions.
- **US-China Tech Tensions**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1932955304188076081) commented on reports that **China demanded the U.S. allow ASML to export mature lithography machines** to ensure SMIC's 14nm production capacity, seeing it as a plausible request.
- **Human vs. LLM Reasoning**: [@goodside](https://twitter.com/goodside/status/1932965557214851229) offered a new analogy for the reasoning debate: "Human and LLM reasoning are a different as apples and potatoes. That is, one is a sweeter version of the other, with a shinier peel, but both grow on vines and are often interbred."
- **Zuckerberg's Past Layoffs**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1933329447853437251) commented that if **Mark Zuckerberg** "hadn't laid off Erik's team of exceptional AI talent a few years ago, they would have less of an AI talent problem today."

**Humor & Memes**

- **OpenAI 🤝 Mattel**: [@gdb](https://twitter.com/gdb/status/1933221591350964633) posted an image of a **Barbie**branded computer with the caption "OpenAI 🤝 Mattel:".
- **The Modern Coder**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1933273732212003237) posted an image of a person coding with the comment, "This feels so old already".
- **The Weights & Biases Experience**: [@vikhyatk](https://twitter.com/vikhyatk/status/1932962492696965626) posted a highly complex, chaotic diagram with the caption, "every time i open weights and biases".
- **PM Code Suggestions**: [@cto_junior](https://twitter.com/cto_junior/status/1933131249083875373) noted that a tragedy of the "democratisation of progress has been PMs pinging you on slack with gpt-4o code suggestions."
- **AI Acronyms**: [@goodside](https://twitter.com/goodside/status/1932990995638976668) joked about the return of **Y2K-style acronyms** in AI circles, such as "tmol-faq. logi. gisai. cev. sysop. ufai. foom. rpop. flare."
- **Internet Outage Reactions**: In response to the massive cloud outage, [@matanSF](https://twitter.com/matanSF/status/1933232190147706952) joked, "That prod migration you’ve been postponing for 3 years? Now's your chance."

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. OpenAI and Industry Model Release Activity and Delays

- [**Petition: Ban 'announcement of announcement' posts**](https://www.reddit.com/r/LocalLLaMA/comments/1l9lddr/petition_ban_announcement_of_announcement_posts/) ([Score: 664, Comments: 78](https://www.reddit.com/r/LocalLLaMA/comments/1l9lddr/petition_ban_announcement_of_announcement_posts/)): **The post criticizes the proliferation of 'announcement of announcement' threads regarding AI model releases, especially from organizations like OpenAI, highlighting repetitive, non-substantive updates. Commenters technically note the issue with unverified sources generating noise, such as screenshots from low-follower Twitter accounts and emphasize the need for reputable, verified news sources instead of speculative or hype-driven posts.** A technical debate emerges on community moderation best practices, with suggestions including stricter verification for posting petitions/announcements and using personal blocking to curate feeds, rather than relying on broader bans.
    - One commenter draws attention to the prevalence of speculative leaks regarding AI models, such as early rumors about DeepSeek v0.2.1.2 releases, and suggests that news posts should be limited to information from reputable or official sources to reduce misinformation and unsubstantiated hype.
    - A point is raised proposing stricter posting requirements for announcement-type posts, such as only allowing accounts older than three months to post, which aims to curb spam and increase trust in update notifications about models and tools.
    - Discussion includes the idea that outright bans on 'announcement of announcement' posts could suppress legitimate information about upcoming model releases (e.g., DeepSeek R2) and that instead, nuanced moderation or verification of source trustworthiness may be more appropriate.
- [**OpenAI delays their open source model claiming to add "something amazing" to it**](https://techcrunch.com/2025/06/10/openais-open-model-is-delayed) ([Score: 344, Comments: 145](https://www.reddit.com/r/LocalLLaMA/comments/1l9fec7/openai_delays_their_open_source_model_claiming_to/)): **OpenAI announced a delay in the release of their open-source model, citing plans to "add something amazing" to it, though no technical specifics or new benchmarks have been publicly disclosed ([source](https://www.reddit.com/r/LocalLLaMA/comments/1daxzbh/openai_delays_their_open_source_model_claiming_to/)). The community is awaiting further details on implementation, model architecture, or expected advancements.** Top comments speculate the delay is due to additional alignment, safety, or restrictive guardrails, potentially trading off openness for increased safety, with some skepticism about the practical utility of the eventual release.
    - One user notes that **OpenAI's most openly released LLM is still GPT-2**, while more recent offerings like GPT-3 have not even been made available in GGUF format for local use. This is contrasted with companies like **Alibaba**, which has released state-of-the-art open models that are accessible and localizable, *despite hardware and embargo limitations*.
    - Another technical point raised is the suggestion that new open source models, even if released, may come heavily guardrailed—potentially impacting their utility. There is skepticism about whether additional safety and security features ('guardrails') might render the model less useful for technical and experimental applications.
- [**Google and Microsoft vs OpenAI and Anthropic, a fun visualization of their open releases on Hugging Face in the past year (Julien Chaumond on LinkedIn)**](https://i.redd.it/2vdfa3f5sg6f1.jpeg) ([Score: 473, Comments: 45](https://www.reddit.com/r/LocalLLaMA/comments/1l9hzb5/google_and_microsoft_vs_openai_and_anthropic_a/)): **The image is a calendar-style visualization comparing the volume of open model, dataset, and space releases by Google, Microsoft, OpenAI, and Anthropic on Hugging Face over the past year. It highlights a significant disparity: Google and Microsoft exhibit much higher release activity (dense clusters of colored tiles) compared to the sparse release frequency of OpenAI and Anthropic. Accompanying data includes follower counts and total releases per organization, reinforcing the contrast in open-source engagement. The visualization is attributed to Julien Chaumond, and a tool for creating similar heatmaps is referenced in the comments (https://huggingface.co/spaces/cfahlgren1/model-release-heatmap).** Commenters emphasize that Google, Microsoft, and Facebook have historically contributed more to open source beyond just AI, and that Anthropic does not claim to be open. The linked Hugging Face Heatmap tool by Caleb Fahlgren provides a way for users to further explore release data.
    - A user highlights the absence of major Chinese AI companies, pointing out significant open model releases from Alibaba (`qwen`) and DeepSeek, suggesting the heatmap underrepresents the global landscape of open AI contributions—especially recent advancements from China.
    - Another user shares a visualization tool: [Hugging Face Model Release Heatmap](https://huggingface.co/spaces/cfahlgren1/model-release-heatmap) by Caleb Fahlgren, which tracks and visually represents public model release activity across organizations, providing granular insights into contribution timelines and organizational trends.
- [**Qwen3-72B-Embiggened**](https://huggingface.co/cognitivecomputations/Qwen3-72B-Embiggened) ([Score: 112, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1l9rejn/qwen372bembiggened/)): **Qwen3-72B-Embiggened is an open-source experimental LLM created by expanding the Qwen3-32B model to match the full Qwen3-72B architecture using a two-stage method: structure-aware interpolation (which rescales hidden and intermediate activations) and simple mid-layer duplication, circumventing the need for training from scratch. The model retains architectural features (such as Group Query Attention), is distributed in 145GB bf16 sharded weights ([Hugging Face link](https://huggingface.co/cognitivecomputations/Qwen3-72B-Embiggened)), achieves preliminary** `80% coherence` **and** `24.25` **perplexity, but includes many duplicated layers (identical until further tuning), and is mainly intended for prototyping or research requiring massive LLMs before full training or distillation. Significant compute is required (145GB VRAM bf16), and post-embedding fine-tuning or distillation (e.g., from Qwen3-235B) is planned to enhance capability and differentiation.** Top technical comments suggest concern over naming conventions (to avoid confusion with official Qwen3 releases), and propose distilling models such as Qwen3-235B or Deepseek into this architecture to revive and expand the underrepresented 70B parameter segment. Interest is high in how such a distilled 72B model would benchmark and perform relative to other large-scale LLMs.
    - The embiggening process for Qwen3-72B-Embiggened involves a two-stage technique: structure-aware interpolation and simple layer duplication to scale the Qwen3-32B model up to match the full Qwen3-72B architecture, effectively creating a 72B-scale model from smaller weights.
    - Planned next steps include distilling the much larger Qwen3-235B model into this 72B-scale architecture, a process that could yield a new model (Qwen3-72B-Distilled), offering a potentially valuable upgrade in performance in an architecture-efficient form factor currently lacking in the open-source model landscape.
    - There is a technical and branding debate: concern is raised about naming conventions, suggesting that models with significant architectural modifications should be clearly differentiated from the official Qwen3 series to avoid misleading users about their provenance and characteristics.

### 2. Open Source Model Releases and Ecosystem Tools

- [**Nanonets-OCR-s: An Open-Source Image-to-Markdown Model with LaTeX, Tables, Signatures, checkboxes & More**](https://www.reddit.com/r/LocalLLaMA/comments/1l9p54x/nanonetsocrs_an_opensource_imagetomarkdown_model/) ([Score: 224, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1l9p54x/nanonetsocrs_an_opensource_imagetomarkdown_model/)): **Nanonets-OCR-s is an open-source, 3B-parameter VLM model capable of converting diverse document features (including tables, equations, images, signatures, watermarks, checkboxes) into structured Markdown and HTML. Notable technical capabilities include accurate LaTeX equation recognition (with inline/block distinction), semantic image tagging, signature/watermark extraction, and robust handling of complex tables and form elements into Markdown-compatible formats. More details and model resources are available via the [Hugging Face Model Card](https://huggingface.co/nanonets/Nanonets-OCR-s), [full announcement](https://nanonets.com/research/nanonets-ocr-s/), and [Colab quickstart](https://github.com/NanoNets/docext/blob/main/PDF2MD_README.md#quickstart).** Commenters report superior table extraction performance compared to Gemini VLM and express interest in format support such as GGUF for local deployment.
    - A user benchmarks Nanonets-OCR-s against Gemini VLM, reporting superior table extraction performance from "weird tables" and highlighting its efficacy in complex structured data scenarios.
    - One request is for consistent document structure control, suggesting that the model could be further enhanced with formatting features that ensure output Markdown adheres strictly to the input document's layout if the source is consistent.
    - Another suggestion proposes native Markdown image output that includes automatic image tag construction with bounding box, page references, and support for easy extraction, as well as the feature for footnote/reference extraction and correct formatting within the Markdown output.
- [**Mistral.rs](http://mistral.rs/) [v0.6.0 now has full built-in MCP Client support!](https://www.reddit.com/r/LocalLLaMA/comments/1l9cd44/mistralrs_v060_now_has_full_builtin_mcp_client/)** ([Score: 105, Comments: 15](https://www.reddit.com/r/LocalLLaMA/comments/1l9cd44/mistralrs_v060_now_has_full_builtin_mcp_client/)): **The release of [mistral.rs](http://mistral.rs/) [v0.6.0](https://github.com/EricLBuehler/mistral.rs/) introduces tightly integrated MCP (Model Context Protocol) client support, streamlining LLM tool integration. The update enables automatic discovery and connection to a wide variety of external tools and services (filesystem, HTTP/SSE, WebSocket) using simple configuration (**`mcp-config.json`**), removing the need for manual integration code and separate tool registries. This support is native to both the Rust library and its Python bindings (available via [PyPI](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/_README.md#installation-from-pypi)), and allows seamless use of the standard OpenAI API interface, with full support for multiserver, authentication, and timeouts. Notable [quickstart](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/MCP_QUICK_START.md) and [Python examples](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/mcp_client.py) are provided.** One commenter asks about key-value (KV) cache compression options akin to `llama.cpp` parameters (`fa -ctk q4_0 -ctv q4_0`), indicating a need for advanced memory optimization features. There is also a debate about packaging, questioning the appropriateness of distributing a Rust library via PyPI rather than Cargo, highlighting cross-ecosystem build and deployment concerns.
    - A user inquires about progress on key-value cache compression, specifically asking for features analogous to llama.cpp's support for cache tensor quantization (`fa -ctk q4_0 -ctv q4_0`), which reduces memory usage and increases inference efficiency. This indicates interest in feature parity with established projects and highlights optimization requests for deployment scenarios.
    - There's a technical question regarding the choice of installation/distribution: one user notes the release of a Rust library via PyPi (typically associated with Python packages), querying the absence of a Cargo distribution method, which would be expected for Rust libraries. This points towards concerns with packaging, language integration, and preferred deployment workflows.
    - Another comment requests guidance on the best deployment route—Docker or local installation—reflecting a technical interest in ease of deployment and potentially containerization for consistent environment setup and reproducibility.

### 3. Unique LLM Deployments and Industry Investment in Superintelligence

- [**Running an LLM on a PS Vita**](https://v.redd.it/we6m8zvv4f6f1) ([Score: 173, Comments: 13](https://www.reddit.com/r/LocalLLaMA/comments/1l9cwi5/running_an_llm_on_a_ps_vita/)): **A developer has successfully ported the minimal Llama2.c large language model inference engine to the PlayStation Vita, leveraging the VitaSDK toolchain. Key technical adaptations include PS Vita-specific syscalls and on-device model downloading/deletion functionality, allowing direct management of LLM model files without manual transfer (see [psvita-llm repo](https://github.com/callbacked/psvita-llm)). A precompiled .vpk is available for quick installation, showcasing LLM deployment on a low-memory (<512MB RAM) embedded platform.** Discussion in the comments centers around the learning curve and setup complexity of VitaSDK versus other homebrew environments, particularly the Nintendo Switch, and community interest in porting LLMs to older consoles like PSP and PS2.
    - One commenter highlights the technical challenge of porting `llama2.c` to the PS Vita, noting that adapting a CPU-intensive language model implementation originally tuned for desktop/server environments to a handheld console likely required substantial modification and optimization. This includes adjustments for memory constraints, CPU instruction set compatibility, and potentially rewriting low-level system calls for the Vita's unique hardware and OS APIs.
    - There is an insightful question about the difficulty of ramping up on the PS Vita SDK compared to modern homebrew development like on the Switch, referencing pain points in setting up development environments and possibly hinting at differences in toolchain maturity, documentation, and community support across platforms.
    - Several users mention that the repository link for the project is broken (404), which raises questions about access to the actual implementation details, code, and documentation—critical for those interested in technical replication or reviewing the approach taken for LLM inference on constrained hardware.
- [**Meta Is Offering Nine Figure Salaries to Build Superintelligent AI. Mark going All In.**](https://www.reddit.com/r/LocalLLaMA/comments/1l9wbaw/meta_is_offering_nine_figure_salaries_to_build/) ([Score: 123, Comments: 67](https://www.reddit.com/r/LocalLLaMA/comments/1l9wbaw/meta_is_offering_nine_figure_salaries_to_build/)): **Meta is reportedly offering compensation packages in the nine-figure range (**`$100M+`**) to assemble a top-tier team to pursue superintelligent AI research, referencing a public drive by Mark Zuckerberg to aggressively compete in the frontier model space. The referenced Entrepreneur article notes Meta's efforts to recruit prominent names in AI, but recent high-profile hires appear to focus on securing former startup founders rather than foundational researchers like Sutskever or Hasabis.** Comments express skepticism regarding Meta's approach, suggesting high compensation is partly intended to retain top talent and prevent competition, and drawing parallels to the lackluster adoption of Meta's metaverse initiatives. There is also critical discussion regarding whether Meta has attracted elite AI researchers or founders with proven technical impact.
    - There's skepticism about Meta's talent recruitment compared to leading AI teams—users question whether Meta is securing world-class researchers like *Ilya Sutskever* (OpenAI), *Demis Hassabis* (DeepMind), or leveraging in-house experts like *Yann Lecun*. Instead, it's noted that they've hired a billionaire founder, which may not correlate with bringing leading-edge research or technical leadership.
    - The mention of 'nine figure salaries' suggests aggressive compensation strategies, potentially aimed at preventing top talent from joining competitors or launching their own startups/labs. This reflects broader industry trends where retaining elite AI talent has become highly competitive and financially intensive due to the perceived strategic importance of advanced AI research.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Claude Code: User Experiences, Productivity, and Agent Techniques

- [**ClaudeCode made programming fun again**](https://www.reddit.com/r/ClaudeAI/comments/1l9ta7s/claudecode_made_programming_fun_again/) ([Score: 137, Comments: 31](https://www.reddit.com/r/ClaudeAI/comments/1l9ta7s/claudecode_made_programming_fun_again/)): **The user highlights that using ClaudeCode (Anthropic's AI code assistant) significantly reduces time spent on monotonous aspects of programming such as reading unclear documentation, bug hunting, and tooling issues, allowing for more direct progress in building projects. Users echo that Claude Code (especially on the Max plan) excels at handling 'busy work' and edge case bug fixes, improving overall productivity and enjoyment of coding.** Technical commenters agree that the value proposition of ClaudeCode lies in automating repetitive troubleshooting, increasing focus on actual development. The main debate centers around how much this changes the programmer's workflow and whether it risks overreliance.
    - Users report a significant reduction in time spent on debugging and handling edge cases when using Claude Code, emphasizing that the tool excels at automating tedious programming tasks. This efficiency notably restores focus on creative aspects of development, as reflected by several developers with extensive experience. One user on the Max plan notes a substantial productivity increase after just a week of adoption.
- [**PSA - don't forget you can invoke subagents in Claude code.**](https://www.reddit.com/r/ClaudeAI/comments/1l9ja9h/psa_dont_forget_you_can_invoke_subagents_in/) ([Score: 124, Comments: 50](https://www.reddit.com/r/ClaudeAI/comments/1l9ja9h/psa_dont_forget_you_can_invoke_subagents_in/)): **The post emphasizes the effective use of Claude's subagents, as described in Anthropic's official documentation ([Claude code best practices](https://www.anthropic.com/engineering/claude-code-best-practices)), for complex tasks, verification, and multi-file/document review—highlighting their role in reducing task hallucination and improving context retention. Explicitly instructing Claude to use subagents for designated subtasks (e.g., code review, file analysis, or testing) reportedly enhances resource efficiency (lower virtual memory usage) and provides more reliable results, likely due to backend optimizations in information handling. The official docs also discuss security considerations of agent tools ([Anthropic security docs](https://docs.anthropic.com/en/docs/claude-code/security)).** Commenters note that specifying the number and scope of subagents for each task step is crucial for optimal performance, and that subagent invocation strategy is a skill requiring experience. There's an unanswered technical question about whether subagent calls consume the parent agent’s context window, and a community resource expands on task/agent tool mechanics ([claudelog.com](http://claudelog.com/) [article](https://claudelog.com/mechanics/task-agent-tools)).
    - A technical user discusses strategies for sub-agent implementation in Claude, emphasizing that defining the number of sub-agents per task step and specifying their responsibilities can improve task throughput and efficiency. Experimentation is encouraged to optimize sub-agent usage to match task complexity and parallelization potential, as outlined in posts and official documentation ([Anthropic docs](https://docs.anthropic.com/en/docs/claude-code/security)).
    - A commenter raises a technical concern about context budget, questioning whether invoking sub-agents in Claude shares the same context size as the parent agent or has a separate context limitation, a crucial issue for scaling complex or data-heavy tasks.
    - A user shares anecdotal evidence that using sub-agents in tandem with code browsing tools (specifically MCP → LSP, likely referring to Language Server Protocol integration) enables more efficient code exploration and search compared to traditional grep-based methods, suggesting improved productivity and automation in code analysis workflows.

### 2. AI Video Generation, Animation, and Creative Uses (Veo, i2v, Midjourney, Kling, etc.)

- [**Added i2v support to my workflow for Self Forcing using Vace**](https://www.reddit.com/gallery/1l9kt2t) ([Score: 103, Comments: 54](https://www.reddit.com/r/StableDiffusion/comments/1l9kt2t/added_i2v_support_to_my_workflow_for_self_forcing/)): **The post announces integration of image-to-video (i2v) support into a workflow for Self Forcing using Vace, highlighting that while output video quality isn't top-tier, the generation speed is notably high (videos produced in ~40s according to user feedback). The workflow and model are available via [CivitAI](https://civitai.com/models/1668005/self-forcing-simple-wan-i2v-and-t2v-workflow).**  Commenters express anticipation for larger (`14b`) model releases, acknowledge the workflow as a 'game changer' for rapid I2V generation, and show interest in understanding the technical details of 'self forcing'.
    - A user reports achieving i2v (image-to-video) generation in `40 seconds` with the new workflow, highlighting a substantial performance improvement and describing it as a 'game changer' for I2V tasks.
    - Another user confirms rapid generation speed, citing precisely `one minute` to create an i2v clip on a `4070 Ti Super` GPU with default settings, providing a useful real-world performance datapoint for that hardware class.
- [**4minute AI Animated Story - Over 500 videos of Experimentation - Cost $1000+**](https://v.redd.it/9zp1jk21ag6f1) ([Score: 363, Comments: 136](https://www.reddit.com/r/aivideo/comments/1l9gnur/4minute_ai_animated_story_over_500_videos_of/)): **The OP produced a 4-minute AI-generated animation using a pipeline comprised of Midjourney for asset (backgrounds/characters) creation, Pika Scenes for animation, and Topaz for video upscaling and frame interpolation; the process involved over 500 individual videos and cost in excess of $1000. Details on workflow and methods are shared in supplemental content linked in the comments.** One commenter technically critiques the animation as a sequence of short '2-second gifs' rather than a traditionally animated story, highlighting current temporal limitations of AI animation tools; another predicts growing acceptance for fully AI-generated animation prior to live-action adoption, while aesthetic appreciation is also noted.
    - One commenter critiques the technical aspects of the animation, noting that while the workflow produces a consistent visual output and a coherent narrative, the shot selection and editing are described as 'jarring.' They emphasize that mastering AI tools does not replace the need for an understanding of traditional directorial principles, such as eyeline continuity and effective narrative editing, to improve storytelling quality.
    - Another commenter points out that the animation style is essentially a sequence of short, 2-second gifs rather than traditional fluid animation, which may impact the perception of continuity and engagement for viewers accustomed to conventional animated shorts.
- [**Trailer for the AI show Seraphys**](https://v.redd.it/4fbu8tf4zj6f1) ([Score: 193, Comments: 63](https://www.reddit.com/r/aivideo/comments/1l9wgr7/trailer_for_the_ai_show_seraphys/)): **The creator produced a spec trailer for an AI-driven series titled 'Seraphys' by integrating multiple AI and traditional tools: script written manually, visual assets generated with Midjourney v7 and processed in Photoshop, image-to-video conversion via Kling 2.1, voices/SFX with Eleven Labs, facial animation using HeyGen Avatar 4, and music with Udio, all edited, color graded, and VFX-applied in DaVinci Resolve. Some additional non-AI sound/music assets were sourced from Uppbeat, demonstrating a hybrid workflow. This illustrates an end-to-end content pipeline leveraging state-of-the-art AI generative tools across media types, showcasing their interoperability and supplementing manual curation.** Commentary emphasizes widespread amazement at the trailer's quality and rapid AI progress, though technical critique or debate is not present, reflecting overwhelmingly positive sentiment on the current advancement of AI tools in creative workflows.
    - creuter highlights the technical achievement by noting the trailer's duration is 54 seconds out of an intended 10-hour runtime, prompting speculation about content generation scalability and completion over such an ambitious timespan. This raises questions about both production workflow and AI model throughput for long-form video generation.
- [**Yeti takes on YETI in this spec ad**](https://v.redd.it/msayw097tj6f1) ([Score: 102, Comments: 28](https://www.reddit.com/r/aivideo/comments/1l9vnla/yeti_takes_on_yeti_in_this_spec_ad/)): **The post discusses the use of Google's Veo 3 video generation model to create a spec ad for YETI as part of a Curious Refuge assignment, referencing the format of popular vlogs. Veo 3 is notable for its capacity to generate coherent video sequences from text prompts, with emphasis on realism and style transfer according to user intention (see [Google Veo paper](https://blog.google/technology/ai/google-veo/)).** Commenters focus on the effectiveness and appeal of Veo 3-generated content, expressing surprise at the quality and entertainment value, despite initial skepticism towards AI-generated media.
    - One commenter notes that the sound effects throughout the ad were of notably high quality, enhancing the overall impact and engagement. They specifically point out the 'sledding on top' segment as a highlight, suggesting that both the audio and editing in this sequence were well-executed. There is also a suggestion to further tighten the video for a more engaging flow, implying attention to pacing and post-production refinement could enhance the technical polish.
- [**If Jerusalem had street interviews A.D.**](https://v.redd.it/lcc0paubkj6f1) ([Score: 281, Comments: 26](https://www.reddit.com/r/ChatGPT/comments/1l9ucko/if_jerusalem_had_street_interviews_ad/)): **The Reddit post presents a comedic video that anachronistically imagines "street interviews" occurring in ancient Jerusalem (A.D.), with user discussions referencing quotable, meme-inspired dialogues presumably featured in the VEO3 video. There is no mention of technical content such as algorithms, benchmarks, implementation, or any software frameworks within the post or top comments.** Users react positively, citing the video as "really funny" and appreciating the satirical approach; some note a typically negative stance toward meme formats but express that this execution is above average in comedic value.
    - No comments in this thread provide substantive technical content, detailed model discussions, benchmarks, or engineering insights. The discussion consists solely of humor, wordplay, and generic praise.

### 3. Seminal AI Research, Industry Debates, and Global AI Impact

- [**Happy 8th Birthday to the Paper That Set All This Off**](https://i.redd.it/ka788zoani6f1.jpeg) ([Score: 1336, Comments: 92](https://www.reddit.com/r/singularity/comments/1l9ple2/happy_8th_birthday_to_the_paper_that_set_all_this/)): **The image is a screenshot from arXiv showing the paper 'Attention Is All You Need,' submitted on June 12, 2017, with revisions through August 2, 2023. This seminal work by Vaswani et al. introduced the transformer architecture, replacing recurrence with self-attention and sparking foundational advances in generative AI. The screenshot includes author names and categorization under Computer Science > Computation and Language.** Comments note the rapid progress since the paper, referencing the 7-year milestone since GPT-1, and praise the paper's title for its concise encapsulation of the paradigm shift.
    - A user notes that it has been 7 years since the release of GPT-1, highlighting the rapid development and evolution in large language models from the original Transformer paper to current state-of-the-art architectures. This underscores how quickly foundational research in attention mechanisms led to practical and scalable implementations like those in OpenAI's GPT series.
    - Another commenter alludes to the lasting relevance of Vaswani et al.'s "Attention Is All You Need" paper, emphasizing that its introduction of the attention mechanism fundamentally altered the trajectory of AI research, and suggesting its influence will be examined for decades as the paper that "changed humanity" by enabling dramatic advancements in NLP and beyond.
- [**Google DeepMind just changed hurricane forecasting forever with new AI model**](https://venturebeat.com/ai/google-deepmind-just-changed-hurricane-forecasting-forever-with-new-ai-model/) ([Score: 988, Comments: 57](https://www.reddit.com/r/singularity/comments/1l9or4z/google_deepmind_just_changed_hurricane/)): **Google DeepMind introduced a new AI hurricane forecasting system [Weather Lab](https://weather.deepmind.com/), tailored for predicting both storm track and intensity, using ensemble techniques for up to 15 days ahead. Internal evaluations per National Hurricane Center (NHC) protocols show DeepMind's model achieved an average five-day track prediction error** `140 km` **lower than ECMWF's ENS, surpassing both global low-res (track) and regional high-res (intensity) models and marking the first experimental AI integration into NHC's operational workflow. The AI leverages deep learning to simultaneously model large-scale atmospheric dynamics and fine-scale storm intensity, aiming to resolve the trade-offs of current physics-based approaches, which have historically separated track and intensity modeling due to resolution constraints.** Commenters note DeepMind's track record of impactful scientific AI (e.g., AlphaFold), the novelty of closed-vs-open models, and hopes for similar advances in tornado prediction. Technical debate focuses on transparency, model accessibility, and the real-world impact of specialized deep learning in weather forecasting.
    - The DeepMind model's key innovation is its ability to forecast both track and intensity of cyclones simultaneously. Traditional models either have global, low-resolution approaches focusing on path prediction or regional, high-resolution ones focusing on intensity—DeepMind claims to bridge this gap, marking a significant advance over past techniques.
    - In internal benchmarks using National Hurricane Center protocols, DeepMind reports that its AI's 5-day hurricane path forecasts are, on average, 140 kilometers closer to actual storm outcomes than the ENS (the leading European physics-based ensemble model), demonstrating substantial practical improvements relevant to operational forecasting.
    - For the first time, the US National Hurricane Center will integrate experimental AI predictions into its operational workflow, accelerating direct adoption of machine learning models into critical infrastructure and setting a precedent for future AI collaboration in meteorology.
- [**Nvidia’s Jensen Huang says he disagrees with almost everything Anthropic CEO Dario Amodei says**](https://fortune.com/2025/06/11/nvidia-jensen-huang-disagress-anthropic-ceo-dario-amodei-ai-jobs/) ([Score: 532, Comments: 153](https://www.reddit.com/r/singularity/comments/1l9o8m9/nvidias_jensen_huang_says_he_disagrees_with/)): **Nvidia CEO Jensen Huang publicly disagreed with Anthropic CEO Dario Amodei's technical assessments regarding the trajectory and risks of AI: Huang specifically challenged Amodei's prediction that 50% of entry-level office jobs may be automated by frontier AI within five years and rejected implications that only select firms (like Anthropic) are safe stewards of AI. Huang argued for broad, open AI development and maintained that technological advances historically lead to labor transformation, not mass job destruction. Additionally, Huang discussed Nvidia's quantum/classical hybrid compute roadmap with CUDA-Q and plans for >20 European "AI factories."** Commenter debate centers on Amodei's true intent—whether he's advocating for regulatory capture or merely cautioning against AI risks—while pointing out possible misrepresentation by Huang. Notably, Yann LeCun sided with Huang, asserting Amodei's positions tend toward exclusivity and alarmism regarding general AI risk, high cost, and mass job loss.
    - There is a technical debate about the accuracy of Dario Amodei's job loss projections due to AI, with Jensen Huang acknowledging potential job losses but questioning the scale and depth predicted by Amodei. Commenters note that historical productivity gains from automation have often led to job transformation rather than absolute loss, challenging dismissive attitudes by some industry leaders.
    - Yann LeCun's linked commentary summarizes three main critiques against Amodei: 1) Overstatement of AI's risks warranting only select companies to build it, 2) excessive focus on cost barriers to entry in AI development, and 3) hyperbolic claims about AI's economic impact, particularly with respect to workforce displacement. The claim about prohibitive costs is seen as ironic given Nvidia's hardware pricing and market dominance.
    - There's technical skepticism about whether Amodei actually advocates that only Anthropic or a few entities should build AI, with counter-arguments referencing his consistent advocacy for broad, multi-stakeholder collaboration in AI governance. Accusations of regulatory capture are discussed but found unsubstantiated in public statements or policy proposals.
- [**Apple’s ‘AI Can’t Reason’ Claim Seen By 13M+, What You Need to Know**](https://youtu.be/wPBD6wTap7g) ([Score: 150, Comments: 92](https://www.reddit.com/r/singularity/comments/1l9snr4/apples_ai_cant_reason_claim_seen_by_13m_what_you/)): **Apple's recent paper claims that current large language models (LLMs) lack genuine reasoning abilities, instead displaying advanced pattern-matching and often failing at complex puzzles—especially as task complexity rises—unless augmented with external tools or code interpreters. Critics note that the paper does not account for tool use, tests some problems outside token limits, and restates known limitations (e.g., LLM hallucinations and token-bound output), suggesting the findings merely confirm what the community already understands about LLMs absent architectural improvements or systematic tool integration. For source material, see the [AI Explained video breakdown](https://youtu.be/wPBD6wTap7g).** Top comments highlight technical criticism of the Apple paper's experimental design (bias, ignoring tool-use capabilities) and reiterate that serious researchers consider these LLM limitations common knowledge; the debate centers on whether the paper's conclusions provide meaningful new insights or simply recycle well-established concerns about reasoning and output limits.
    - The technical critique of the Apple paper centers on its methodology: it assessed LLMs on complex puzzles and observed performance degradation as complexity rose, but overlooked that LLMs are not deterministic solvers and are known to be limited in unaided reasoning. Notably, the paper is faulted for ignoring LLMs' capability to use external tools (such as code interpreters), which can dramatically improve problem-solving performance on complex tasks outside the LLM's direct reasoning capacity.
    - Another methodological flaw noted: the paper tested LLMs on tasks that exceeded their maximum output (token) limitations, thus invalidating some results by pushing models beyond their design specification. Critics argue this, along with a perceived bias against LLM reasoning in the initial framing, means the paper adds little to what is already widely known in the AI research community about LLM weaknesses.
    - A key insight highlighted is that breakthroughs in practical AI applications relying on LLMs often result from integrating these language models with external systems and tools that can supplement reasoning or fact-checking, rather than depending solely on the native modeling capacity of the language model itself.
- [**If GPT 4.5 came out recently and is barely usable because of its power consumption, what is GPT 5 supposed to be? (Sam said everyone could use it, even free accounts.)**](https://www.reddit.com/r/OpenAI/comments/1l9k1en/if_gpt_45_came_out_recently_and_is_barely_usable/) ([Score: 225, Comments: 109](https://www.reddit.com/r/OpenAI/comments/1l9k1en/if_gpt_45_came_out_recently_and_is_barely_usable/)): **Discussion centers on why OpenAI is hyping GPT-5 despite GPT-4.5 (aka 4-turbo) reportedly being deprecated due to performance (especially power consumption and cost) issues. The transition roadmap indicates GPT-4.5 will be replaced by GPT-4.1. Some comments suggest GPT-5 may integrate multiple model families (like 4o, o3, o4, o5), dynamically selecting which system to use per request, potentially improving efficiency and scalability for broader accessibility—including free users. Commenters note current models such as o3 are significantly cheaper ('20x') and more capable than 4.5, making widespread deployment feasible. Versioning and naming across models is described as inconsistent and confusing (see shared [ChatGPT discussion](https://chatgpt.com/share/684ac50b-60c8-8012-8978-aa0dddd75fa3)).** There is skepticism about OpenAI's versioning and naming conventions and their roadmap, with multiple commenters noting the confusion and inconsistency in public communications about model upgrades and replacements.
    - Discussion centers on model progression and deployment efficiency: commenters note that GPT-4.5 was primarily a brute-force scaled model, but was inefficient in energy use and is being deprecated in favor of lighter successors like 4.1 and 4-turbo, both of which are more cost-effective (`o3 is now like 20x cheaper than 4.5 while being much more capable`).
    - There is debate about versioning and architecture: several users stress that OpenAI's version numbers (4.5, 5, etc.) do not directly correspond to model size, but rather functionality and deployment practicality. For example, one comment notes that OpenAI has been optimizing to make GPT-4-class models 'smarter, except for the poorly named GPT-4.5,' while actually reducing cost and size.
    - Speculation arises that future versions like GPT-5 may combine the best components/techniques from multiple prior models (e.g., 4o, o3), and incorporate lessons from 4.5’s inefficient architecture to create a more balanced, scalable LLM, supporting both free and paid users.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: AI Model Performance & Capabilities Unleashed (and Compared)**

- **Trio of Titans: Opus, O3 Pro, Gemini 2.5 Pro Tag-Team Tasks!**: Engineers in the Perplexity AI community combine **Opus** for conceptual learning, **O3 Pro** as an analyst, and **Gemini 2.5 Pro** for mathematics, noting task-dependent performance. Cursor Community users also find **Opus 4** outshines **Sonnet 4** in coding by looping less, while praising **Gemini 2.5 Pro** for critical thinking despite its issues configuring Postgres.
- **ChatGPT Gets Juiced! OpenAI Spills Beans on GPT-4o's $15M Price Tag!**: OpenAI announced **ChatGPT Projects** now support deep research, voice mode, and improved memory for **Plus, Pro, and Team users**, while **Canvas** gains PDF, docx, and markdown export capabilities. Meanwhile, users estimate **GPT-4o** training cost around **$15 million**, sparking debates on Plus subscription profitability following a discussion prompted by a [YouTube video on Veo 3](https://www.youtube.com/watch?v=QGwWJl7AZ6c).
- **Video AI Slugfest: Seedance 1.0 KOs Veo3 While Veo 3 KOs Wallets!**: Discussion in Perplexity AI highlighted that [Seedance 1.0 is currently outperforming Google's VEO3](https://video.twimg.com/amplify_video/1933194931566243840/vid/avc1/1920x1080/HEEIOuxi8TLuVj8y.mp4) in text/image-to-video generation. However, Manus.im users raised concerns about **Veo 3's** video generation costs, with one user reporting a charge fluctuating between **300 to 600 credits for just 8 seconds** of footage.

**Theme 2: When Clouds Cry: Infrastructure Woes and Platform Stability Saga**

- **Internet Armageddon! Cloudflare & GCP Outage Triggers AI Platform Panic!**: A widespread internet outage involving **Cloudflare** and **Google Cloud Platform (GCP)** caused significant disruptions across multiple AI platforms including OpenRouter, Cursor, LlamaIndex, and Cohere. OpenRouter confirmed impact via its [status page](https://status.openrouter.ai/), users referenced [Downdetector](https://downdetector.com/) reports, and Cohere acknowledged issues due to the [GCP incident reported by Google](https://status.cloud.google.com/incidents/mKVakfB1qM3Hvb9cUpqv).
- **Manus Grounded by AWS Outage, LMArena Weeps Over Lost Chats!**: Manus.im users experienced problems with file uploads and task execution due to a broad **AWS outage** that also affected services like YouTube and Twitch. Separately, LMArena faced a **cloud provider outage** leading to potential loss of chat history data, prompting the team to apologize and work on preventative measures.
- **Firebase Falls, OpenRouter Tumbles in Domino Effect of Doom!**: Users on LlamaIndex and OpenRouter reported **Firebase** being down, impacting authentication services as highlighted in [Greg Hunkins' X.com post on the Firebase outage](https://x.com/greghunkins/status/1933223568394846703?s=46). This had a knock-on effect, causing [OpenRouter to also go down, as discussed on Hacker News](https://news.ycombinator.com/item?id=44260810).

**Theme 3: Squeezing AI Brains: Fine-Tuning, Quantization, and Optimization Frontiers**

- **ABBA Says "Gimme! Gimme! Gimme!" More Performance Than LoRA!**: The new **ABBA** architecture for **Parameter-Efficient Fine-Tuning (PEFT)**, detailed in [its arXiv paper](https://arxiv.org/abs/2505.14238) with [code available on Github](https://github.com/CERT-Lab/abba), significantly outperforms **LoRA** by modeling updates as a Hadamard product of two low-rank matrices. Unsloth AI and Eleuther members discussed its consistent wins over SoTA LoRA variants on models like **Mistral-7B, Gemma-2 9B, and LLaMA-3.2 1B/3B**.
- **DeepSeek R1 Quantizes Like a Champ, AMD GPUs Flex 35x Inference Muscles!**: Unsloth AI members found **DeepSeek R1** quantizes remarkably well compared to **Qwen3**, possibly due to its **bf16** training, though fine-tuning `unsloth/DeepSeek-R1-0528-Qwen3-8B` triggered "Unrecognized keys" warnings. Excitement also brewed over **AMD's MI350X and MI355X AI GPUs**, which [Tom's Hardware reports](https://www.tomshardware.com/pc-components/gpus/amd-announces-mi350x-and-mi355x-ai-gpus-claims-up-to-4x-generational-gain-up-to-35x-faster-inference-performance) claim up to **35x** faster inference performance.
- **Torchtune Solves Memory Mysteries and Supercharges MoE Models!**: Torchtune developers investigated a memory consumption anomaly where `(bs, seqlen*8)` inputs use more memory than `(bs*8, seqlen)` with **flex attention** and **FSDP**, as shown in this [memory usage chart](https://cdn.discordapp.com/attachments/1216353675744641096/1382573185916211200/image.png?ex=684c4dde&is=684afc5e&hm=0d498abf01433cd6a078a17e983947e7ac7e0590281a6728dec8a13a5fba2776). They also found that using `_grouped_mm` substantially boosts **finegrained MoE** speed, making **Qwen3-30B-A3B** performance nearly match **8B** models.

**Theme 4: Dev Tooling & API Adventures: From Rate Limits to WASM Dreams**

- **OpenRouter Users Dodge Rate Limits & Decipher DeepSeek's Chinese Whispers!**: OpenRouter users hit **10,000 RPM** on the Fireworks provider for **Qwen3 30B** structured outputs, learning the displayed `rate_limit` object is inaccurate and will be deprecated. Others reported **DeepSeek models** intermittently switching to Chinese during responses, with suggestions to try alternative providers like GMICloud or the `r1 0528` version.
- **Mojo Unleashes String Speed Demon, Gets Cozy on LeetGPU!**: Modular's **Mojo** now offers **40% faster string operations** in its nightly build compared to Python and has gained support on [LeetGPU](https://leetgpu.com/), increasing accessibility for developers. Engineers also demonstrated its borrowing iterators for `FastxReader` and discussed workarounds for the current absence of dynamic dispatch using the [Variant library by josiahls](https://github.com/josiahls/firehose/tree/master/firehose).
- **MCP Servers Flirt with WASM & Service Workers, FastFS Joins the Party!**: Developers in the MCP (Glama) community explored running **MCP servers** using **service workers** directly in the browser, potentially compiling to **WASM**. Hyper-MCP advocated for WASM on the host, though concerns about SDK access loss were noted, while another user shared their [fastfs-mcp project on GitHub](https://github.com/aj-geddes/fastfs-mcp) as an example of *having some fun with this*.

**Theme 5: Research Ripples: New Papers and Projects Making Waves**

- **Factorio AI Aims for "AlphaFactorio" Glory, One Docker Image at a Time!**: The GPU MODE community discussed improving the **Factorio Learning Environment (FLE)** to help superintelligent systems understand complex real-world systems, as outlined in their [FLE position paper on arXiv](https://arxiv.org/pdf/2502.01492). A key goal is an **AlphaFactorio** project, with a member sharing a proof-of-concept [FLE docker image and mod project on GitHub](https://github.com/MortenTobiasNielsen/fle_suggestion).
- **World Models Are In, Schmidhuber Says: Agents, Get Your Predictive Hats On!**: A [new paper from Schmidhuber's lab on "Agents that Learn to Model the World"](https://arxiv.org/abs/2506.01622), discussed in the Yannick Kilcher server, argues that general agents must learn a **predictive model of their environment** for multi-step goal-directed tasks. The author's [accompanying blog post](https://richardcsuwandi.github.io/blog/2025/agents-world-models/) provides further context on this fundamental requirement for improving agent performance.
- **EleutherAI Hits Reset Button as Meta's V-JEPA 2 Sees the Light of Day!**: EleutherAI announced a research focus shift after its local volume estimator, detailed in [their "Neural Redshift" paper on arXiv](https://arxiv.org/abs/2501.18812), failed to accurately track learning behaviors across activation functions. Concurrently, Meta unveiled [V-JEPA 2, its self-supervised video model publication](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/), with code and data release imminent.



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI Models Combine for Peak Performance**: One member detailed their AI combo for maximizing performance: **Opus** for conceptual learning, **O3 Pro** as an analyst, and **Gemini 2.5 Pro** for mathematics.
   - The member noted that performance varies depending on the task, emphasizing the importance of using each AI in its area of expertise.
- **Discord Deploys New Automated Bot Detector**: Discord users noticed that Discord has released a new bot detector that flags spam automatically.
   - Members noted that the bot detector works automatically without the need to download any additional discord mod.
- **Perplexity Tasks Rollout Triggers Comet Buzz**: **Perplexity Tasks** are rolling out to Pro and Enterprise accounts to generate news on specific topics, with **Deepsearch** also planned to be available in Tasks.
   - One user said *This will get really wild on Comet*, referring to their [own take on the same technology](https://video.twimg.com/amplify_video/1933215329154404352/vid/avc1/1920x1080/pbQOGV7Jenwgr2_c.mp4).
- **Deepsearch Faces Delays, Sparks Compute Concerns**: Members are reporting that **Deepsearch** has been delayed despite PPLX saying otherwise, and now they are back to *room sized computers* to get the compute.
   - Users in the discord channel are still hoping for the best after [the recent announcement](https://x.com/OpenAI/status/1933208575968752092).
- **Veo3 Outshines Gemini in AI Video Arena**: [Seedance 1.0 is beating VEO3](https://video.twimg.com/amplify_video/1933194931566243840/vid/avc1/1920x1080/HEEIOuxi8TLuVj8y.mp4) in the AI text and image to video spaces, but it is unknown what is being used in perplexity.
   - The space is moving so fast that a lead today can become a laggard tomorrow.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **AI Bot is Abused!**: A member asked the bot to *make a program to find G(n,k)*, and other members believed they *officially abused it*.
   - A member responded that this was more for *testing it rather than solving*.
- **Kingfall Image Leaks!**: A user shared an [alleged Kingfall image](https://cdn.discordapp.com/attachments/1340554757827461211/1382530309098049566/image.png?ex=684cceaf&is=684b7d2f&hm=7fbc452b8b5b5969993ab2493a3ba78f2558bc390095a9aef1e1c5c11742b2bd&) described as *an allegory about impotence*.
   - It was later reported that it was *patched*, and redirected to an older version, with one member saying *you can't use it anymore*.
- **O3 Pro Pricing Confuses Users!**: Members debated the value and pricing of **O3 Pro** versus **Gemini 2.5 Pro**, citing various experiences, benchmarks, and cost considerations.
   - Some believe **O3 Pro** is priced higher due to its superior capabilities, while others found **Gemini 2.5 Pro** to be more cost-effective, or even superior in certain tasks like math.
- **LMArena Experiences Cloud Outage!**: A **cloud provider outage** caused issues with the site, potentially resulting in the **loss of chat history data**.
   - The team [apologized](link.to.apology) for the inconvenience and is working on preventative solutions. The development team is actively implementing **preventative measures**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Free Model Limits Fall**: Members discussed the rate limits for free models, clarifying that it's **50 requests/day** if you have less than $10 total topped up, otherwise, it's **1000/day** shared across all free models.
   - The limit applies regardless of the number of tokens in/out, and even failed requests count toward the limit.
- **Paid Model Rate Limits in Flux**: A user encountered **429 errors** despite paying for the service, and inquired about the rate limits for paid models, while attempting to run a bunch of requests concurrently to label data.
   - A staff member said the displayed *rate_limit* object is inaccurate and will be deprecated, stating that there aren't really rate limits for paid models, but identified the user was hitting **10,000 RPM** on the only structured outputs provider for **Qwen3 30B**, which is Fireworks.
- **OpenRouter Plunges During Planetary Problems**: OpenRouter experienced a **global outage** due to a widespread internet issue impacting services like **Cloudflare** and **Google Cloud**, causing widespread service disruptions and user frustration.
   - Staff confirmed they were impacted but the outage was not their fault, linking to a [status page](https://status.openrouter.ai/) and [Downdetector](https://downdetector.com/) for updates, while users humorously speculated on the cause and impact, with some mentioning Gemini Website was luckily working.
- **DeepSeek Delivers Dubious Dialogue**: A user reported **DeepSeek models** were intermittently switching to **Chinese** during responses, with others confirming the issue.
   - Recommendations included adjusting settings like *temperature*, *top_p*, and *top_k*, and monitoring which providers are serving broken responses, with suggestions to try *r1 0528* and providers such as GMICloud and Inference.net.
- **Requesty Rides to the Rescue**: Users briefly mentioned **Requesty** as an alternative to OpenRouter, with one user describing it as more of an *enterprise-grade infra solution* focused on reliability and performance.
   - It was noted that Requesty users were experiencing uptime while OpenRouter was struggling due to the global outage and it was touted as a solution for production workloads needing stability.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Opus 4 Outshines Sonnet 4 in Code Looping**: Members debated the merits of **Sonnet 4** and **Opus 4** for coding, with some noting that [**Opus** loops less than **Sonnet**](https://www.cursor.com/docs/models/understanding-models#claude-3).
   - However, **Gemini 2.5 Pro** was lauded for critical thinking and refusing bad suggestions, unlike **Sonnet** and **Opus**, which *always obey no matter what*.
- **Gemini 2.5 Pro Nuke Postgres Configuration**: Users reported that **Gemini 2.5 Pro** performs poorly when configuring Postgres, sometimes nuking databases and needing **Opus 4** or **O3** to fix the configurations.
   - Despite the shortcomings, it was praised for critical thinking and the ability to refuse bad user suggestions.
- **Cloudflare Incident Causes Cursor Slowdowns**: A **Cloudflare** and **GCP** incident caused a widespread internet outage, leading to significant slowdowns and login issues for **Cursor** users, according to [Cloudflare status](https://www.cloudflarestatus.com/).
   - Despite the downtime, some users reported that **O3** was still functioning and praised **Cursor** for its prompt response.
- **Cursor Mobile App Anticipation Surges**: Community members are excited about a potential **Cursor mobile app**, drawing parallels with **Replit** for on-the-go coding.
   - There was discussion about the efficiency of **Cursor Tab** completions, and comparisons to [**Copilot**](https://github.com/features/copilot) and its overall effectiveness.
- **Background Agents Error on Windows**: Users reported getting a `Connection Failed` error when running background agents on Windows and a Cursor dev is tracking [Windows bugs](https://discord.com/channels/1074847526655643750/1380811765218283660) and hopes to have them fixed in the next release.
   - Background agents must install dependencies and have access to all extensions **in a remote environment**, meaning agents require code storage.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 Quantizes Nicely**: Members find that **DeepSeek R1** quantizes very well compared to **Qwen3** leading to suspicion that it is due to **DeepSeek R1** being trained in **bf16**.
   - Members reported an *Unrecognized keys* warning while fine-tuning the new **DeepSeek-R1** model ([unsloth/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B)).
- **AMD Promises Performance Gains**: A [Tom's Hardware article](https://www.tomshardware.com/pc-components/gpus/amd-announces-mi350x-and-mi355x-ai-gpus-claims-up-to-4x-generational-gain-up-to-35x-faster-inference-performance) detailed that **AMD's MI350X** and **MI355X AI GPUs** claim up to **4x** generational gain and up to **35x** faster inference performance.
   - The community encouraged the Unsloth team to prioritize support for **AMD** hardware.
- **Unsloth to Feature Multi-GPU Support**: The Unsloth team is working on official **multi-GPU** support and there are already around **5** different repos for **multi-GPU** support.
   - Members linked to a [Reddit thread discussing multi-GPU support](https://www.reddit.com/r/unsloth/comments/1l8mxkq/multigpu_support_how_to_make_your_unsloth/).
- **ABBA Architecture Outperforms LoRA**: A new architecture called **ABBA** for **Parameter-Efficient Fine-Tuning (PEFT)** significantly outperforms **LoRA** and its major variants under the same parameter budget as detailed in this [paper](https://arxiv.org/abs/2505.14238).
   - The code is available [on Github](https://github.com/CERT-Lab/abba) and consistently beats **SoTA LoRA** variants on commonsense and arithmetic reasoning across **4** open-source LLMs (**Mistral-7B, Gemma-2 9B, LLaMA-3.2 1B/3B**).
- **Fetch Image encounters NoneType Error**: A user reported an `AttributeError: 'NoneType' object has no attribute 'startswith'` during training with **Unsloth**, stemming from the `fetch_image` function encountering `None` values in the images field of the JSON dataset.
   - A member suggested ensuring that each batch contains either all images and text or only text, or using a batch size of 1, or passing a custom collator.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI Eyes Exit from Estimator Endeavors**: EleutherAI is pivoting its research focus after its local volume estimator, detailed in [this paper](https://arxiv.org/abs/2501.18812), failed to accurately track learning behaviors across activation functions.
   - The shift follows concerns that prior work on simplicity at initialization might be brittle, particularly for networks with high weight magnitudes, as outlined in [EleutherAI's blog](https://blog.eleuther.ai/inductive-bias/) and [accompanying code](https://github.com/EleutherAI/tyche).
- **India Inspires AI Institute**: AI Safety India [aisafetyindia.com](https://aisafetyindia.com/about) emerged this year, aiming to be a hub for AI safety research and discussion with at least one advisor on Discord.
   - Its sudden appearance surprised some, given the existence of other AI safety institutes and the member's location, sparking hopes that it's more than just *"a dead website."*
- **Meta Manifests V-JEPA 2, Validating Vision**: Meta unveiled [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/), a self-supervised video model, planning to release code and data imminently.
   - While one member dubbed **JEPA's premise nuts**, others defended **Yann's** long-standing vision of creating useful world representations in an unsupervised manner.
- **ABBA Annihilates Alternatives, Acclaimed as Apex of Adaptability**: A new architecture for **Parameter-Efficient Fine-Tuning (PEFT)**, named **ABBA**, outperformed **LoRA** by modeling updates as a **Hadamard product of two independently learned low-rank matrices** [paper](https://arxiv.org/abs/2505.14238) and [code](https://github.com/CERT-Lab/abba).
   - Members discussed the balance between expressivity and rank, recognizing **ABBA's** achievement in both for enhanced performance.
- **Epoch Engineering Elevates LLM Excellence**: A member discovered that training small LLMs for **2 epochs** with **warm-up and linear decay** in the first and **cosine decay** in the second enhances performance in classification tasks, outlined in [this paper](https://arxiv.org/pdf/2404.06395).
   - These improvements are notably significant for smaller LLMs when applying this specific training method.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Projects Gain New Powers**: Projects in **ChatGPT** are gaining new capabilities including **deep research support**, **voice mode support**, and **improved memory**, rolling out to **Plus**, **Pro**, and **Team users**.
   - Improved memory is exclusive to **Plus** and **Pro** users, while **Canvas** now supports downloads to **PDF**, **docx**, or **markdown** formats, and mobile users now have access to the model selector.
- **Apple Called Mediocre in Flying Car Discussion**: A user shared a [YouTube interview](https://youtu.be/NTLk53h7u_k?si=VF8-zJZLQziFhpD_So) where a woman told **Apple** how mediocre they are nowadays, amid discussions of **flying cars** and **flying taxis**.
   - Users debated if **O3 Pro**'s generation time is artificially inflated to disincentivize usage and cut down on compute, but some agree that *O3 Pro is better than O3*.
- **LLMs Fail Simple Reasoning Tasks**: A paper showed that **LLMs failed when images are changed artificially**, leading to a discussion on whether LLMs can truly reason or are simply biased toward training data.
   - One user argued that LLMs are just *mimicking intelligence* and compared LLMs to **System 1** in the dual process theory of psychology, suggesting that achieving **AGI** will require more than just LLMs.
- **Explicit Forbidden Tokens Increase Leakage Risk**: A member warned that [enumerating forbidden tokens](https://owasp.org) amplifies recency bias and increases the risk of **LLM leakage**, and that the absence of evidence is not evidence of absence, especially with emergent risks.
   - They also pointed out that best practices recommend externalizing enforcement of prohibited content, with OpenAI's guidelines suggesting generalized, legal descriptors.
- **GPT-4o Training Costs Revealed!**: A user estimated that **GPT-4o** cost around **$15 million** to train, sparking a discussion on the profitability of **Plus** subscriptions based on the real costs of inferencing the model, as prompted by a link to a [YouTube video](https://www.youtube.com/watch?v=QGwWJl7AZ6c) regarding **Veo 3**.
   - Another member reported experiencing potential memory bleed across custom **GPTs**, sharing some *strong evidence* suggesting the memory they are seeing is not hallucinations, but is truly memory bleed.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Chat Mode Excites Users**: Members are excited about the new **chat mode** in Manus, viewing it as a *gamechanger* for avoiding credit wastage on simple queries and enhancing user experience by eliminating app-switching.
   - While some believe Manus should focus on task completion rather than general chatting, a moderator highlighted that it would reduce complaints about credit wastage, as users can get quick answers without using agent mode.
- **Veo 3 Video Costs Prompt Concerns**: Users discussed the cost of **Veo 3 video generation**, with one member reporting an initial charge of **300 credits for 8 seconds** of footage, which later increased to **600 credits**.
   - Calculations suggest a 5-minute video could cost **$47.50**, and a 1-hour movie could cost around **$570**, excluding additional expenses for music and sound.
- **"High Effort Mode" Now Auto-Enabled**: Members noticed that the option to manually select **High Effort Mode** has been removed, with the system now automatically enabling it when deemed necessary.
   - A user expressed satisfaction that the **high effort mode** is now a natural process, removing the need for manual selection.
- **Credit Wastage and Text Handling Errors Plague Users**: Users reported issues with **text handling** that led to credit loss; one user saw **150 credits** vanish due to repeated text handling errors in the editor, while another witnessed Manus performing a task twice.
   - A member advised starting a new session to mitigate the issue, while another observed that the problem is linked to slide uploads and is more common since the introduction of chat mode.
- **AWS Outage Grounds Manus**: The Manus platform faced issues because of a widespread **AWS outage**, which impacted file uploads, task execution, and general functionality.
   - Services like YouTube, Twitch, and Discord image uploads were also affected, with members jokingly speculating about alien landings in San Francisco.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Dual GPUs Double Down in LM Studio**: Users confirm that dual GPUs improve performance in **LM Studio**, with configurations like **32+16GB** showing great results.
   - Some concerns were raised that dual GPUs aren't too taxing.
- **SSD Swapping Sparks Lifespan Scare**: Discussion about using **mmap()** swapping led to warnings about potentially reducing **SSD lifespan** due to excessive writes.
   - While SSDs are rated by terabytes written (**TBW**), the heavy write operations from swapping raised concerns.
- **Spec-Decoding Shenanigans with Unsloth's Qwen**: Users ran into issues with speculative decoding using **Unsloth's Qwen3 models**, especially when trying to run the draft model on the GPU and the main model on the CPU.
   - A [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1kftu3s/draft_model_compatible_with/) clarified that the draft model has to load into the GPU, suggesting the crashing issues aren't related to processor selection.
- **LM Studio Shuns Spontaneous Software Updates**: Users confirmed that **LM Studio** does not automatically update models when new versions are available on **Hugging Face**.
   - Model updates often involve new generations in new repositories, complicating in-place updates and lineage tracking.
- **LLMs Learn to Love Limited Length**: LLMs are trained to be concise to save computational cost and avoid boring users, which may be frustrating those looking for essay-like summaries.
   - A suggestion was made to split the task: first get the structure, then ask for content for each bullet point.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Screenplay Tools Spark Interest**: Members shared resources for creating **AI screenplay and filmmaker tools**, including links to [ACM Digital Library](https://dl.acm.org/doi/fullHtml/10.1145/3544548.3581225), [Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling), and [EQBench](https://eqbench.com/creative_writing.html).
   - One member noted a recent incident in Japan where **ChatGPT** was used to write a screenplay for a government-commissioned film, causing *some trouble*.
- **HF API 'No Inference Provider' Error Troubles Users**: Users reported a `No Inference Provider available` error for models like `nlpconnect/vit-gpt2-image-captioning` when using the Inference API.
   - A member suggested checking the available models and inference providers on the [Hugging Face settings page](https://huggingface.co/settings/inference-providers) and [chat interface](https://huggingface.co/chat/).
- **AI Avatar Project Runs into Memory Limits**: A member building an **AI avatar project** is facing crashes due to a **2GB video generation model** exceeding their **8GB RAM**, seeking local workarounds as AWS GPU isn’t viable.
   - Suggestions included exploring **model quantization, frame splitting, and running low-vram modes**, plus a link to sign up for credits on the Nvidia developer program.
- **Newbies Want `requirements.txt` to tame Agents**: Developers request a `requirements.txt` file and **Python version guidelines** for the **Agents course**, to aid local development outside of Colab.
   - One developer encountered issues with **llama-index** not being found, which they then found out the directory was named the same as the libraries.
- **Unlimited Text To Video Emerges!**: A new [Unlimited Text To Video](https://huggingface.co/spaces/NihalGazi/Unlimited-Text-To-Video) app has been released, but suffers from low resolution and slowness.
   - The upside is that the video generation is *unlimited*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Parallel Patterns Prompted for GPU Roles**: Members preparing for **GPU engineering roles** sought resources on **parallel programming patterns**, referencing the *holy book* which turned out to be **PMPP**.
   - The discussion highlighted the need for supplementary video materials to enhance understanding of these patterns.
- **Rounding Errors Kill Triton Kernel**: A user optimized **conv1d kernel performance** by **rounding up to the next power of 2**, with another user improving from the **5th to the 95th percentile** by addressing the issue.
   - Another user provided their **Triton kernel code** demonstrating input splitting into blocks and kernel tiling to optimize **conv1d** performance.
- **PTX Modifiers Tweak Cache Policies**: A member discussed using **PTX modifiers** for load instructions and **cache eviction policies** with the goal of optimizing memory layout across the **L2 cache** for **GEMMs**.
   - The member noted issues setting eviction policies independently for **L1** and **L2** caches, but was given pointers about the **Blackwell library** and its optimizations.
- **`torch.func.functional_call` Debuts**: A member observed the integration of `torch.func.functional_call`, doubting it solves the **integrated API problem** but noted `functorch` can now be accessed through `torch.func`.
   - Additionally, a new approach to loading pretrained weights into `nn.Linear` layers was proposed via `nn.Linear.from_pretrained(weight=param)` instead of the meta device approach currently used by the [VLLM project](https://github.com/vllm-project/vllm/pull/19265/commits/0a0b1a8e9a57d0f2f543e76c18b074544847cce4).
- **FLE aims for AlphaFactorio**: Members stated that improving the Factorio Learning Environment (**FLE**) will help superintelligent systems understand complex real-world systems, referencing [this position paper](https://arxiv.org/pdf/2502.01492).
   - They outlined the first priority to make the **FLE** usable and the second to create an **AlphaGo-like** project (AlphaFactorio) to optimize for performance, while a member created a POC project for a standalone **FLE docker image and mod** ([https://github.com/MortenTobiasNielsen/fle_suggestion](https://github.com/MortenTobiasNielsen/fle_suggestion)).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CS Degree Value Questioned**: A member sparked debate on *whether a CS degree is still useful*, igniting a brief discussion before being cautioned to stay on topic.
   - The discussion quickly pivoted back to technical matters.
- **SVD Test Stumbles on Sign Flip**: During **SVD testing**, a member reported a failure attributed to a **sign flip**.
   - The issue manifested as **mismatched elements**, accompanied by detailed tracebacks highlighting **max absolute** and **relative differences**.
- **eigh() Implementation Sparks Bounty**: The complexity of adding **eigh()** to tinygrad led to a suggestion to make it a separate bounty, along with attachment of [A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf](https://cdn.discordapp.com/attachments/1070745817025106080/1382505351634616381/A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf?ex=684cb771&is=684b65f1&hm=c832598efa3830d558a0f9f457a339b3b05863e26df8e9f372fdd6419a7ba60e&).
   - The community acknowledged the challenge and signaled interest in contributing.
- **LLM Chatbot for tinygrad Discord?**: Members explored integrating an **LLM chatbot** (like [getunblocked.com](https://getunblocked.com/)) with the Discord chat and codebase to provide context-aware answers to user queries.
   - The proposal involves stripping bulk and low-signal files and feeding the relevant context to the **LLM** to enhance its responsiveness and accuracy.
- **Tensor Norm Function Surfaces, Caveats Apply**: In response to an inquiry, a member shared a [linalg.py file](https://cdn.discordapp.com/attachments/1070745817025106080/1382766636364333176/linalg.py?ex=684c5948&is=684b07c8&hm=2f078256ba98c1fd605435de76fd7f16dee8dbb1ea9ca817886a44df1e9b7338&) containing a **tensor.norm()** implementation for tinygrad.
   - The author conceded that while the function *works in tinygrad 100%*, it's *not as fast as numpy or as accurate*.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **AI Audio Customization Remains Unresolved**: Users report that generating separate AI audio overviews for each topic in a source isn't possible due to the **customization option disappearing** after the initial generation.
   - The workaround involves prepping sources and custom audio instructions and then generating a new notebook and audio repeatedly.
- **Anime Edits Channel Seeks Subs**: A user promoted their **YouTube channel**, *THE OP KID*, which features anime edits, actively seeking subscriptions from the community.
   - The user gave no further details.
- **Podcast Powerhouse Shares Compilations**: A user shared a series of podcasts created using audio overviews, touching on subjects like **high school reading**, **missionaries and Bible characters**, **cognitive distortions and psychology**, **solving cold cases with AI**, and **thoughtful TV or film programs**.
   - The user shared links to each podcast on **Spotify** and mentioned that the "How-to" podcast unexpectedly hit #1 in Morocco.
- **NotebookLM Age Restrictions Spark Debate**: Discussions arose around **NotebookLM's age restrictions**, with one user stating it's integrated with **Family Link** and has a minimum age of **13**.
   - Another user suggested that the age policy might vary by region, particularly between **America** and the **EU**.
- **MathJax Rendering Extension Launched**: A user introduced **LaTeXLM**, an open-source **Chrome extension** designed for **MathJax rendering** within **NotebookLM**, shared via [GitHub](https://github.com/hachoj/LaTeXLM).
   - The extension enables users to utilize local **Chrome extensions** without the need for scripts.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Oscar-C Project Seeks Testers**: A member is seeking testers for **oscar-c**, a project focused on *cognitive architecture/xai/neurosymbolic AI* and invites interested individuals to DM them for more information.
   - The project aims to explore new frontiers in cognitive architecture.
- **Altman and Marcus Spar Over Intelligence**: Members debated a [post](https://x.com/sama/status/1932588741584957482) between **Sam Altman** and **Gary Marcus** on the definition of reasoning and intelligence.
   - One member contended that *99% of people arguing this is not 'true' reasoning / intelligence etc can't even define it in a way that includes most of the humans too*.
- **Prompt Engineers Find Humanity's Last Guide**: A member requested resources for writing **system prompts for agents** and another shared [Humanity's Last Prompt Engineering Guide](https://www.forwardfuture.ai/p/humanity-s-last-prompt-engineering-guide).
   - The guide contains useful recipes and tips to write better prompts.
- **Adaptive Resonance Theory Resonates**: A member highlighted the relevance of **Adaptive Resonance Theory (ART)** algorithms, prompting the sharing of a [survey paper](https://arxiv.org/abs/1905.11437) on the topic.
   - ART is a class of algorithms that attempts to address stability-plasticity dilemma.
- **World Models: Critical for General Agents?**: A new [paper](https://arxiv.org/abs/2506.01622) argues that **general agents** must learn a **predictive model of their environment** for multi-step goal-directed tasks, extracting it from the agent's policy and requiring increasing accuracy for improved performance.
   - The author's [blog post](https://richardcsuwandi.github.io/blog/2025/agents-world-models/) provides additional context, though the paper is the meat of the information.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Leaps onto LeetGPU!**: **Mojo** now enjoys support on services like [LeetGPU](https://leetgpu.com/), increasing accessibility for development and testing.
   - This allows developers to harness **Mojo**'s capabilities across diverse hardware configurations.
- **FastxReader Mapped with Borrowing Iterators**: An engineer demonstrated **FastxReader's** borrowing iterator with a dict-comp in **Mojo** using `rec.name[]: rec.seq[]` syntax.
   - This showcases how **Mojo** concisely maps sequence names to sequences when reading a fastq file.
- **Modular Docs Ahead of the Nightly?**: Engineers identified a mismatch between **Modular Docs** and **nightly** builds of **Mojo**, due to errors related to references.
   - A member recommended using the **nightly** build with `--index-url https://dl.modular.com/public/nightly/python/simple/` to align with the documentation.
- **Dynamic Dispatch Dodged in Mojo**: Members dissected the absence of dynamic dispatch, type lambdas, and type families in **Mojo**'s type system.
   - The discussion covered using [Variant](https://github.com/josiahls/firehose/tree/master/firehose) in lists as a workaround, although full implementation is still pending.
- **Mojo String Ops 40% Faster!**: String operation optimizations in **Mojo**'s **nightly** branch show a **40%** speed improvement over Python in a small string benchmark.
   - The engineer posting the test said that the next stable release will see *"a lot of performance improvements"* for anyone doing fast string manipulation with the code example he provided showing how to split a string.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Memory Consumption Anomaly Surfaces with Large Inputs**: A user noticed that `(bs, seqlen*8)` inputs consume more memory than `(bs*8, seqlen)` inputs with **flex attention** and **FSDP**, especially with large inputs reaching a "tipping point" where memory usage rapidly increases, as seen in [this chart](https://cdn.discordapp.com/attachments/1216353675744641096/1382573185916211200/image.png?ex=684c4dde&is=684afc5e&hm=0d498abf01433cd6a078a17e983947e7ac7e0590281a6728dec8a13a5fba2776).
   - It was hypothesized that logits might be the source, but tokens per second remain steady, shown in [another chart](https://cdn.discordapp.com/attachments/1216353675744641096/1382594289455857765/image.png?ex=684c6185&is=684b1005&hm=895b4043b33e49c5a7a89f047382cc18fc1416879e9d1c26b318c87bf345e22b).
- **_grouped_mm Accelerates Finegrained MoE**: The use of `_grouped_mm` substantially boosts the speed of **finegrained MoE**, making **Qwen3-30B-A3B** performance nearly on par with **8B**, after it was initially slower than **32B** using a for-loop approach.
   - This optimization highlights the significance of efficient matrix multiplication in optimizing the performance of large-scale models.
- **Packing Refactor Aims for Iterable Dataset Harmony**: A proposal for packing refactor to integrate better with [iterable datasets](https://github.com/pytorch/torchtune/pull/2819) is underway, with the intent to support packing for **DPO**, **GRPO**, and **multimodal** applications.
   - The projected timeline includes gathering feedback and landing iterable dataset RFC and packing RFC, targeted for completion by the *end of next week*.
- **Devs Navigate **Qwen3** Builder Snafu!**: A user reported an issue where **Qwen3** models were utilizing **Qwen2** builders in [#2809](https://github.com/pytorch/torchtune/pull/2809), leading to incorrect model construction.
   - The proposed solutions involve creating dedicated **Qwen3** component builders or enhancing the **Qwen2** builder with custom attention mechanisms, with a leaning towards the latter for leaner boilerplate.
- **Architectural Novelties may complicate **Mistral 3.1 Small**: Members discussed potential architectural novelties of **Mistral 3.1 Small** that might complicate fine-tuning implementations.
   - While **multimodal** capabilities are not new, they could add complexity to implementations such as *devstral*, which might not fully utilize multimodal features.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Service Workers Run MCP Servers**: Members discussed using **service workers** to run **MCP servers** directly in the browser, leveraging *postMessage* and dedicated threads.
   - It was noted that running a **MCP server** compiled to **wasm** in the browser is possible but might be worse than creating the **MCP server** directly in JS.
- **Zapier MCP Connectivity Plagued by 500 Errors**: A user reported trouble connecting to **Zapier MCP** over OAuth, citing issues with their OAuth metadata server and **/token** endpoint producing 500 errors.
   - This issue has been raised for the attention of server authors.
- **Spinning up Playwright MCP Servers**: A member inquired about interest in a service that spins up **Playwright MCP Server** instances in the cloud, enabling access from anywhere, like **n8n workflows**.
   - This cloud-based setup would allow the **MCP Server** endpoint to be reached from any location.
- **Hyper-MCP Champions WASM for MCP Servers**: **Hyper-MCP** is advocating for **WASM** to run **MCP servers** directly on the host.
   - The main concern is the loss of access to existing SDKs, though some believe this is less than ideal.
- **fastfs-mcp has showcase fun**: A member shared [fastfs-mcp](https://github.com/aj-geddes/fastfs-mcp) with the comment *having some fun with this*.
   - No further details were provided.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Order Completion Agent Fills Forms**: The new [Order Completion Agent with Artifact Editor example](https://t.co/oKxZxjajzZ) uses an AI assistant to fill out structured forms while talking to users.
   - This shows that an AI assistant can complete forms.
- **LlamaCloud Recovers After Hiccup**: **LlamaCloud** is back online after instability in our upstream infrastructure provider, as announced on the [LlamaIndex status page](https://t.co/IdecAksHiG).
   - Users can check the status page for the latest updates.
- **LlamaIndex Embraces MistralAI's Magistral**: **LlamaIndex** now supports @MistralAI's **Magistral** reasoning model in any agent workflow; details are [here](https://t.co/ZsUEWMrnT4) and [here](https://t.co/QFONzaZRk0).
   - This enhancement should improve the agent's reasoning capabilities.
- **Firebase feels the heat**: A member reported that **Firebase** is down, affecting authentication services, as highlighted in [this post](https://x.com/greghunkins/status/1933223568394846703?s=46).
   - Another member sarcastically noted that *a lot of things will be down if firebase is having an outage* and [OpenRouter is also down](https://news.ycombinator.com/item?id=44260810) as a knock-on effect of the **Firebase** outage.
- **GCP, Cloudflare Suffer**: A member reported that **GCP (Google Cloud Platform)** is down and **Cloudflare** is also experiencing issues.
   - Another member speculatively attributed the issues to **BGP (Border Gateway Protocol)** problems.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Skips Multi-Model Re-Ranker**: A member noted that *currently **Cohere** doesn’t have a multi model re-ranker* and suggested using **CLIP** and **openCLIP** as an alternative.
   - Another member is exploring using **GPT-4.1** with structured output and custom prompt instead for a more tailored approach.
- **Amotions AI Hunts Tech Co-Founder**: The founder of **Amotions AI** is seeking a technical co-founder with an AI background to *take Amotions AI to the next level*, particularly its [real-time AI sales coach](https://www.amotionsinc.com/).
   - The goal is to strengthen the **AI** capabilities of their sales tool.
- **Xarray-JAX Ascends**: A member is developing the **Xarray-JAX library** for **Google DeepMind** as part of **GSoC 2025**, highlighting it as *effectively the first named tensor implementation in a deep learning framework*.
   - They anticipate that this integration will significantly benefit the machine learning community and are open to discussing potential applications and improvements.
- **Cohere Services Suffer GCP Setback**: Cohere reported experiencing an outage due to a [Google Cloud Platform (GCP) incident](https://ift.tt/on1ARP0) impacting some of their services as of **June 12, 2025 at 12:02PM**.
   - The specific component affected is **Infrastructure**, with further details available on the [Cohere Status Page](https://ift.tt/Ens6bma).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 3.0 Arrives in Beta**: [DSPy 3.0](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1) has launched in **beta** and members are looking for a comprehensive overview of the changes.
   - The community seems interested in the new features and improvements, especially as it relates to replacing input fields dynamically.
- **Agent Bricks make Debut**: A member shared a screenshot and a link to a [Databricks blog post](https://www.databricks.com/blog/introducing-agent-bricks) introducing **Agent Bricks**.
   - The blog post details how **Agent Bricks** enhance agent capabilities within the Databricks environment, but no further discussion was given about specific usage or implications within the DSPy context.
- **Docstring Referencing: Jinja Replacement Needed**: A member inquired about referencing the **input field** inside a **docstring** in DSPy, specifically seeking **dynamic jinja replacement** for enhanced flexibility.
   - Although it was pointed out that *docstrings are just text*, the request highlights a desire for more dynamic capabilities within DSPy's documentation practices.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX Summit Details Emerge**: The **AgentX summit** invites finalists for the research track to a poster session or talk, which gives them an opportunity to submit papers separately via the [summit website](https://rdi.berkeley.edu/events/agentic-ai-summit).
   - Finalists will receive separate invitations to the summit and do not need to register to attend but, it is recommended to register early for guaranteed spots with potential ticket refunds.
- **Additional AgentX Summit Information**: A user inquired about specific details regarding research paper submissions and attendance for finalists at the **AgentX summit**.
   - Submitting separately increases chances for additional consideration, using the [summit website](https://rdi.berkeley.edu/events/agentic-ai-summit).



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Models too slow for members**: Members complain that *thinking models are too slow*.
   - There was discussion on why models are slow even when they are smaller, like **1GB, 2GB, or 4GB**.
- **Token Count Impacts Performance**: Model slowness may be due to the number of tokens in use.
   - Members suspect that *Too many f*cking tokens* might be contributing to slower processing speeds.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Drops Wave 10 with UI/UX Facelift**: Windsurf announced the release of **Wave 10**, featuring a fresh slate of **UI/UX upgrades**, along with new teams and enterprise offerings as covered in their [blogpost](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise).
   - The release includes new icons for `@-mentions` and file citations, codeblocks in the Cascade panel matching the IDE theme, a native terminal in the Cascade panel accepting user inputs, and a new Conversation History UI.
- **Windsurf Expands to EU with New Cluster**: Windsurf announced the launch of their **EU Cluster**, promising faster performance and catering to the rising demand from European enterprises, detailed in their [blog post](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise).
   - Details can be found in their [video on Youtube](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI) and change logs can be found at [https://windsurf.com/changelog](https://windsurf.com/changelog).



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1382449597539946697)** (1408 messages🔥🔥🔥): 

> `AI Combos, Power Grid Prompts, Text 2 Vid Arena, Deepsearch Delayed, Comet Browser Issues` 


- **AI Model Team-Up**: One member detailed their AI combo for maximizing performance: **Opus** for conceptual learning, **O3 Pro** as an analyst, and **Gemini 2.5 Pro** for mathematics.
   - The member noted that performance varies depending on the task, emphasizing the importance of using each AI in its area of expertise.
- **Discord Releases New Anti-Spam Bot Detector**: Discord users noticed that Discord has released a new bot detector that flags spam automatically.
   - Members noted that the bot detector works automatically without the need to download any additional discord mod.
- **PPLX Tasks rolling out to PRO and Enterprise**: Perplexity Tasks are rolling out to Pro and Enterprise accounts to generate news on specific topics, with Deepsearch also planned to be available in Tasks.
   - One user said *This will get really wild on Comet*, referring to their [own take on the same technology](https://video.twimg.com/amplify_video/1933215329154404352/vid/avc1/1920x1080/pbQOGV7Jenwgr2_c.mp4).
- **Deepresearch on backorder?**: Members are reporting that Deepsearch has been delayed despite PPLX saying otherwise, and now they are back to *room sized computers* to get the compute.
   - Users in the discord channel are still hoping for the best after [the recent announcement](https://x.com/OpenAI/status/1933208575968752092).
- **Veo3 wins against Gemini in video arena**: [Seedance 1.0 is beating VEO3](https://video.twimg.com/amplify_video/1933194931566243840/vid/avc1/1920x1080/HEEIOuxi8TLuVj8y.mp4) in the AI text and image to video spaces, but it is unknown what is being used in perplexity.
   - The space is moving so fast that a lead today can become a laggard tomorrow.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1382559148482232422)** (3 messages): 

> `RTX 4090, Windows Recall Security Flaws` 


- **Perplexity showcases RTX 4090, the first consumer GPU**: A user shared a [Perplexity page](https://www.perplexity.ai/page/rtx-4090-first-consumer-gpu-to-XRM0cWrDSQO5Z4PwRmQzlA) about the **RTX 4090**, calling it the first consumer GPU.
- **Perplexity uncovers Windows Recall's security flaws**: A user posted a [Perplexity page](https://www.perplexity.ai/page/windows-recall-security-flaws-Q5a7MAJWTn.KWGoDocbbmA) covering the security flaws in **Windows Recall**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1382822588576694427)** (1 messages): 

> `Sonar API Documentation, Perplexity API documentation feedback` 


- **Sonar API Documentation: Users Solicited for Feedback**: The Perplexity team is seeking feedback on their **Sonar API documentation** and has created a thread in their community forum for this purpose: [Improvements to the Sonar API Documentation](https://community.perplexity.ai/t/improvements-to-the-sonar-api-documentation/542?u=vikvang).
   - Users are encouraged to share any difficulties they've had with the documentation, such as aspects that are *unclear or hard to find*.
- **Community Thread Created for API Documentation Feedback**: A community thread has been created to gather user feedback regarding the **Sonar API documentation** to identify areas needing improvement.
   - Users are invited to contribute their experiences and suggestions in the dedicated thread.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1382434491213283408)** (999 messages🔥🔥🔥): 

> `G(n,k) program, Claude ultrathink option, O3 Pro Benchmarks, Kingfall, Titanforge` 


- **Abusing AI: Program for G(n,k) requested**: A member asked the bot to *make a program to find G(n,k)*, which another member called *officially abused it*.
   - A member responded that this was more for *testing it rather than solving*.
- **Kingfall Leaks and Patches**: A user shared an [alleged Kingfall image](https://cdn.discordapp.com/attachments/1340554757827461211/1382530309098049566/image.png?ex=684cceaf&is=684b7d2f&hm=7fbc452b8b5b5969993ab2493a3ba78f2558bc390095a9aef1e1c5c11742b2bd&), which was described as *an allegory about impotence*.
   - Later, members mentioned it being *patched*, and redirecting to an older version, with one member saying *you can't use it anymore*.
- **OpenAI's O3 Pro faces backlash for reasoning and tool use**: Members share that OpenAI's **O3 Pro** is *good at complex tasks*, especially math when using web browsing and tools.
   - Some mention its issues and limitations that impact the overall user experience, suggesting improvements like *adapt the reasoning length* or * make the models enjoy talking to*.
- **Google's cultural shift raises eyebrows**: Concerns are raised about the ethical direction of Google's AI development, with discussions about the influence of DeepMind's CEO and the need for *anti-killbots etc*.
   - A member pointed out Google may have several cultures, with one observing *over half the company has been new people so many times that the culture has reset many times*.
- **Gemini 2.5 and O3 Pro pricing is confusing users**: Members debated the value and pricing of **O3 Pro** versus **Gemini 2.5 Pro**, citing various experiences, benchmarks, and cost considerations.
   - Some argued that **O3 Pro** is priced higher due to its superior capabilities, while others found **Gemini 2.5 Pro** to be more cost-effective or even superior in certain tasks like math.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1382832731314061373)** (1 messages): 

> `Cloud Provider Outage, Data Loss Incident` 


- **LMArena Suffers Data Loss From Cloud Outage**: A **cloud provider outage** caused issues with the site, potentially resulting in the **loss of chat history data**.
   - The team [apologized](link.to.apology) for the inconvenience and is working on solutions to prevent future occurrences.
- **Preventative Measures Underway**: The development team is actively implementing **preventative measures** to safeguard against similar cloud provider-related data loss incidents in the future.
   - Details on specific strategies and infrastructure improvements will be shared as they become available.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1382784485938036928)** (7 messages): 

> `Cloudflare downtime, Google Cloud outage, Internet outage` 


- ****Internet-wide Outage** Strikes!**: An **internet-wide outage** was reported, impacting Cloudflare and Google Cloud, as detailed on [Downdetector](https://downdetector.com/).
   - The Cloudflare [status page](https://www.cloudflarestatus.com/) and [Google Cloud status page](https://status.cloud.google.com/) provided ongoing updates, with recovery seen later in the day, according to [this tweet](https://x.com/OpenRouterAI/status/1933263905385500853).
- ****Cloudflare & Google Cloud** Suffer Downtime!**: **Cloudflare** and **Google Cloud** experienced downtime due to a broader internet outage, prompting investigations and status updates.
   - Updates were actively shared on the Cloudflare [status page](https://www.cloudflarestatus.com/) and Google's [status page](https://status.cloud.google.com/), where users could monitor the situation.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

memgrafter: I will test it tomorrow, send it over
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1382438884180234340)** (971 messages🔥🔥🔥): 

> `Free Model Rate Limits, Paid Model Rate Limits, OpenRouter Global Outage, DeepSeek models and Chinese, Requesty as an alternative to OpenRouter` 


- **Free Model Limits Capped at 50 or 1000**: Members discussed the rate limits for free models, with one noting it's **50 requests/day** if you have less than $10 total topped up, otherwise, it's **1000/day** shared across all free models.
   - It was clarified that this limit applies to the number of requests, regardless of the number of tokens in/out, and that failed requests also count.
- **Rate Limits for Paid Models in Flux**: A user encountered **429 errors** despite paying for the service, and inquired about the rate limits for paid models, trying to run a bunch of requests concurrently to label data, but was being rate limited.
   - A staff member noted that the displayed *rate_limit* object is inaccurate and will be deprecated, stating that there aren't really rate limits for paid models, but identified the user was hitting **10,000 RPM** on the only structured outputs provider for **Qwen3 30B**, which is Fireworks.
- **OpenRouter Paralyzed by Global Internet Meltdown**: OpenRouter experienced a **global outage** due to a widespread internet issue impacting services like **Cloudflare** and **Google Cloud**, causing widespread service disruptions and user frustration.
   - Staff confirmed they were impacted but the outage was not their fault, linking to a [status page](https://status.openrouter.ai/) and [Downdetector](https://downdetector.com/) for updates, while users humorously speculated on the cause and impact, with some mentioning Gemini Website was luckily working.
- **DeepSeek V3 Model Garbles Gibberish**: A user reported **DeepSeek models** were intermittently switching to **Chinese** during responses, with others confirming the issue and suggesting potential causes and solutions.
   - Recommendations included adjusting settings like *temperature*, *top_p*, and *top_k*, and monitoring which providers are serving broken responses, with suggestions to try *r1 0528* and providers such as GMICloud and Inference.net.
- **Requesty Replaces Router During Rough Ride**: Users briefly mentioned **Requesty** as a reliable alternative to OpenRouter, with one user describing it as more of an *enterprise-grade infra solution* focused on reliability and performance, while OpenRouter focuses on trying new models.
   - It was noted that Requesty users were experiencing uptime while OpenRouter was struggling due to the global outage and it was touted as a solution for production workloads needing stability.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1382435189518499890)** (619 messages🔥🔥🔥): 

> `Opus vs Sonnet, Gemini 2.5 Pro fails, MCP servers, Cloudflare outage, Cursor Mobile App` 


- **Opus 4 Trumps Sonnet 4 in Code Loops**: Members debated whether **Sonnet 4** or **Opus 4** is better for coding, with one user stating that in their experience, [**Opus** loops less than **Sonnet**](https://www.cursor.com/docs/models/understanding-models#claude-3).
   - Another member pointed out that **Gemini 2.5 Pro** is even better when processing over 120k tokens, but that **Cursor** has hidden the tool call error messages.
- **Gemini 2.5 Pro Flounders with Postgres Configuration**: Users reported that **Gemini 2.5 Pro** is performing badly when configuring Postgres, sometimes resulting in nuked databases and requiring **Opus 4** or **O3** to fix its configurations.
   - Despite **Gemini**'s poor performance, it was lauded for its critical thinking and ability to refuse bad user suggestions, in contrast to **Sonnet** and **Opus** which *always obey no matter what*.
- **Cloudflare Meltdown and Intermittent Cursor Outage**: A widespread internet outage, primarily caused by a [**Cloudflare** and **GCP** incident](https://www.cloudflarestatus.com/), led to **Cursor** experiencing significant slowdowns and login issues.
   - While most services were down, some users noted that **O3** was still functioning and praised **Cursor** for its prompt response to the outage, despite others comically announcing *I'm fired*.
- **Cursor Mobile App Anticipation Builds**: Members are buzzing with excitement over the potential of a **Cursor mobile app**, drawing parallels to **Replit** and speculating on its ability to provide on-the-go coding capabilities.
   - Others touted the efficiency of **Cursor Tab** completions, with the efficiency of [**Copilot**](https://github.com/features/copilot) itself under question.
- **Rules-Based AI**: Community members shared a tip on using rules in **Cursor**, where they can be set up at either the global level or at the project level.
   - Members shared tips for their MCP setup, and the importance of [rules](https://docs.cursor.com/context/rules) in achieving maximum effectiveness when coding with AI.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1382527015445336215)** (44 messages🔥): 

> `Background Agents, Code Storage, Privacy Mode, Windows Bugs, Non-Github Repositories` 


- **Background Agents Error on Windows**: Users report getting a `Connection Failed` error when running background agents on Windows.
   - A Cursor dev is tracking [Windows bugs](https://discord.com/channels/1074847526655643750/1380811765218283660) and hopes to have them fixed in the next release.
- **Agents Require Code Storage**: Background agents, in their current form, fundamentally require code storage to execute and iterate code in a remote environment.
   - A repo-level control is sadly not supported right now.
- **Background Agents and Privacy Mode**: Background agents are not supported in privacy mode, though a fix is rolling out soon.
   - There is an account-level privacy mode [that can be enabled](https://www.cursor.com/slack-connected) even if machine-level privacy mode is disabled.
- **Background Agents LSP Errors**: Background agents should have all the **LSP errors** and access to all extensions, but dependencies must be installed in the agent environment.
   - Background agents **run in a remote environment**.
- **Deeper PR Integration in the Works**: Cursor is considering a [deeper PR integration](https://github.com/langfuse/langfuse-docs/blob/main/.cursor/environment.json) for background agents.
   - The background agent **amends commits** per user direction.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1382436868896985279)** (215 messages🔥🔥): 

> `DeepSeek R1 fine-tuning issues, Safetensors to AWQ conversion, DeepSeek R1 8Q model fine-tuning, Aider Polygot benchmark trustworthiness, QwenLong-32B model release` 


- **DeepSeek-R1 Fine-Tuning Warning Appears**: A user reported an *Unrecognized keys* warning while fine-tuning the new **DeepSeek-R1** model ([unsloth/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B)).
   - It was recommended to follow the [Unsloth documentation](https://docs.unsloth.ai/get-started/installing-+-updating/updating#to-use-an-old-version-of-unsloth) to resolve the issue.
- **DeepSeek R1's Quantization Capability**: Members observed that **DeepSeek R1** quantizes very well compared to **Qwen3** on a benchmark, leading to speculation about **Qwen's** training and quantization process.
   - The member suspects it is due to DeepSeek R1 being trained in **bf16**.
- **Unsloth's AMD Event Showdown!**: Members shared a [link to a Tom's Hardware article](https://www.tomshardware.com/pc-components/gpus/amd-announces-mi350x-and-mi355x-ai-gpus-claims-up-to-4x-generational-gain-up-to-35x-faster-inference-performance) about **AMD's MI350X** and **MI355X AI GPUs** which claim up to **4x** generational gain and up to **35x** faster inference performance.
   - The community is actively encouraging the Unsloth team to prioritize support for **AMD** hardware.
- **RL Agents become the new prompting agent?**: A member suggested that prompting agents are the 'old style' and the 'new style' is **RL agents**!
   - No one disagreed.
- **Multi-GPU Support Hitting Unsloth Soon!**: Unsloth team is working on official **multi-GPU** support and indicated that there are already around **5** different repos for **multi-GPU** support.
   - Members linked to a [Reddit thread discussing multi-GPU support](https://www.reddit.com/r/unsloth/comments/1l8mxkq/multigpu_support_how_to_make_your_unsloth/).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1382521253025419336)** (19 messages🔥): 

> `Hyperbolic Pricing, Synthetic Datasets, VRAM vs RAM, Typo in Advertisements` 


- **Hyperbolic Pricing Attracts Scrutiny**: A user questioned the reality and uptime of a product, commenting that *hyperbolic is cheap* in reference to [an attached screenshot](https://cdn.discordapp.com/attachments/1179039861576056922/1382521252853584004/Screenshot_20250611-204515.png?ex=684cc640&is=684b74c0&hm=dae1d40020262d2817ede7e21a27bee71172e3708db11a4633555dbcd2884054&).
- **Synthetic Datasets from Large LLMs are very cool**: One user indicated that they tried a product and will use it next time they need a **synthetic dataset from a larger LLM**.
   - They also added that *good hardware is fairly cheap*.
- **VRAM vs RAM causes confusion**: Users debated whether an advertisement listed **512GB of RAM** or **VRAM**, with one initially posting a [duck-themed GIF](https://tenor.com/view/duck-eco-gif-22827059) expressing disbelief.
   - Another user noted that *in the PC it says GPU RAM: 80GB*.
- **Typo suspected in Advertisement**: One user speculated that the **advertisement contained a typo** and should have specified **80GB of VRAM** instead of RAM, alongside **2TB of RAM**.
   - Another user countered that the advertiser was just *lazy and thought people will understand its VRAM* due to advertising a GPU.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1382439952544960712)** (103 messages🔥🔥): 

> `Unsloth version requirements, bias training issues, Granite biases, Qwen2.5-VL-7B-Instruct fine-tuning, Overfitting with LoRA` 


- ****Unsloth Version Compatibility: A Pip-tastic Predicament****: A user inquired about the requirements for older versions of **Unsloth**, questioning if `pip install unsloth==2025.2.15` is sufficient, and seeking guidance on handling compatibility issues with other packages like **transformers** and **PEFT**.
   - They also asked about the status of bias training toggle in older Unsloth versions, referencing [an open issue](https://github.com/unslothai/unsloth/issues/2343) related to LoRA training, suggesting potential tweaks to the utils.
- ****Fetch Image Fails: A NoneType Nightmare****: A user reported an `AttributeError: 'NoneType' object has no attribute 'startswith'` during training with **Unsloth**, stemming from the `fetch_image` function encountering `None` values in the images field of the JSON dataset.
   - A member suggested ensuring that each batch contains either all images and text or only text, or using a batch size of 1, or passing a custom collator.
- ****Qwen2.5-VL-7B-Instruct Fails: A Template Tango****: A user encountered a `RuntimeError` when deploying a fine-tuned `unsloth/Qwen2.5-VL-7B-Instruct` model on vLLM v0.8.5, related to missing or incorrect tokens for multi-modal inputs.
   - The issue was traced to a missing `chat_template.jinja` file in the merged model, with potential fixes involving upgrading **unsloth-zoo** and **unsloth** via `pip install --force-reinstall --no-deps git+https://github.com/unslothai/unsloth-zoo.git` and `pip install --force-reinstall --no-deps git+https://github.com/unslothai/unsloth.git`.
- ****LoRA Loss: Overfitting Outbreak!****: A user shared charts showing a model overfitting after around 700 epochs during **LoRA** fine-tuning with **Unsloth**, indicated by a significant drop in training loss and an increase in eval loss.
   - It was suggested to monitor the eval loss and stop training when it starts climbing, focusing on generalization rather than maximizing training epochs.
- ****Llama 3.2 Tool Time: Calling for Guidance****: A user sought advice on fine-tuning **Llama 3.2 (3B)** for tool calling with 6 custom tools, planning to use **GPT-4** to generate synthetic example conversations.
   - They asked for guidance on the approach and the recommended number of examples, with a follow-up question about whether zero-shot or multi-shot/long conversations are expected.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

not_easy_to: I’m fine-tuning Qwen 2.5 7B (using Unsloth) and need a small French math dataset
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1382488877838635189)** (5 messages): 

> `ABBA architecture, LoRA alternatives, Parameter-Efficient Fine-Tuning` 


- **ABBA crushes LoRA in Parameter-Efficient Fine-Tuning**: A new architecture called **ABBA** for **Parameter-Efficient Fine-Tuning (PEFT)** significantly outperforms **LoRA** and its major variants, under the same parameter budget as detailed in this [paper](https://arxiv.org/abs/2505.14238).
   - Unlike **LoRA**, which adds a low-rank delta to frozen weights, **ABBA** models the update as a Hadamard product of two independently learned low-rank matrices.
- **ABBA beats SoTA LoRA variants**: **ABBA** consistently beats **SoTA LoRA** variants on commonsense and arithmetic reasoning across **4** open-source LLMs (**Mistral-7B, Gemma-2 9B, LLaMA-3.2 1B/3B**).
   - In some cases, it even outperforms full fine-tuning, with code available [on Github](https://github.com/CERT-Lab/abba).


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1382574996077023363)** (1 messages): 

> `Volume Estimator, Neural Redshift, Generalization Heuristic, AI Alignment, Inductive Bias` 


- **Local Volume Estimator Fails to Track Learning Behavior**: A research update indicated that their local volume estimator from [this paper](https://arxiv.org/abs/2501.18812) failed to track differences in learning behavior across activation functions.
   - The estimator also did not confirm that **ReLU networks** are simpler than **tanh networks**, leading to pessimism about its utility as a heuristic for generalization.
- **EleutherAI Plans Shift in Research Focus**: After one more research update on local volume work focusing on applications to **AI alignment**, EleutherAI plans to shift its research focus elsewhere.
   - This decision follows findings that prior work on simplicity at initialization may be brittle, particularly concerning networks with high weight magnitudes; see [EleutherAI's blog post](https://blog.eleuther.ai/inductive-bias/) and [accompanying code](https://github.com/EleutherAI/tyche).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1382469402087395431)** (86 messages🔥🔥): 

> `AI Model Comparison Platforms, Open Science at CVPR, AI Safety India, GPT models behavior, Symbolica AI startup` 


- **AI Model Comparison Platform Hunt Kicks Off**: A member inquired about platforms that compare **AI models** across tasks like programming, translation, documentation, and email writing; [livebench.ai](https://livebench.ai/#/hello) was suggested as a resource.
   - They mentioned that they are new to the field and were curious about which models excel in different scenarios.
- **Open Science Gathering at CVPR**: A member extended an invitation to discuss **open science** at **CVPR** for anyone attending; [lu.ma link](https://lu.ma/z1o7ncnt) provided.
   - Discussion also covered the observation that research labs are predominantly in computer science, leveraging Discord and Slack for communication due to funding and the ease of *"doing things"* such as hackathons.
- **AI Safety India is Launched**: A member shared a link for **AI Safety India**; [aisafetyindia.com](https://aisafetyindia.com/about) was created this year, with one advisor on Discord.
   - Another member, surprised at not having heard of it despite knowing other AI safety institutes and being from India, expressed interest, hoping it wasn't a dead website.
- **GPT Models Provoke Emotional Responses**: A member noted that GPT-3 and GPT-4o sometimes give plain responses, but other times, suddenly go deep into **emotional and structural interpretation**; [arxiv.org paper](https://arxiv.org/pdf/2406.20052) linked.
   - This was deemed a known *"issue"*, explained as models picking the next word via randomness, influencing subsequent word predictions.
- **Symbolica AI Aims for Symbolic AI**: A member inquired about **Symbolica AI**, a London startup with ambitious goals, which led to discussion about its purpose and potential.
   - Another member suggested contacting an ex-employee ([Bruno Gavranovic](https://www.brunogavranovic.com/)) for insights and another shared that some reviews mention that the "boundaries of the work" aren't clear and the goals keep changing.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1382449128465895635)** (210 messages🔥🔥): 

> `Small LLM Training Epochs, Meta's V-JEPA 2 Self-Supervised Video Models, Building an AI Expert Agent for Google Ads, Parameter-Efficient Fine-Tuning (PEFT) with ABBA, CommonPile Data and the Role of Synthetic Data` 


- **Epoch Engineering Elevates LLM Excellence**: A member shared their experience that training small LLMs for **2 epochs**, with a **warm-up and linear decay** in the first and **cosine decay** in the second, yields better results for classification tasks, referencing [this paper](https://arxiv.org/pdf/2404.06395).
   - The results for smaller LLM can be significantly improved when using that type of training technique.
- **Meta Reveals V-JEPA 2, Visionary Video Validator**: **Meta** released [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/), a self-supervised video model, with code and data slated for release soon.
   - One member called **JEPA's premise nuts**, but the work under the label *pretty cool*, while another defended **Yann's vision** as consistent since 2022, aiming for useful world representations in an unsupervised manner.
- **ABBA Achieves Apex-level Adaptability, Annihilating Alternatives**: A new architecture for **Parameter-Efficient Fine-Tuning (PEFT)**, called **ABBA**, was released, significantly outperforming **LoRA** and its variants, modeling updates as a **Hadamard product of two independently learned low-rank matrices** [paper](https://arxiv.org/abs/2505.14238) and [code](https://github.com/CERT-Lab/abba).
   - Members discussed the value of expressivity versus rank, with **ABBA** achieving both for performance boosts.
- **Commons Conundrum: Commercial or Community?**: Members discussed the release of the **Institutional Books 1.0** dataset under a noncommercial license, despite its aim to create a healthier foundation for model development across the commercial, academic, and public spheres, available on [HuggingFace](https://huggingface.co/datasets/institutional/institutional-books-1.0).
   - Concerns were raised about the restrictive nature of the license and the potential for companies to free-ride without contributing back to the commons.
- **Quantization Quest: Can LLMs Learn with Less?**: There's hype around a stealth startup making progress in extreme quantization, allegedly quantizing a model to **1-2 bits** with minimal loss in accuracy.
   - The fact that quantization works so well shows how little information is truly absorbed, leading to discussion about external stores and pointers for long-tail recall.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1382471938407928038)** (4 messages): 

> `Knockoffs, Predictor-corrector methods, Realistic null distribution` 


- ****Knockoffs** Reminder Strikes Back**: A member mentioned that they haven't seen **knockoffs** mentioned in like 4 years, thanking another for the reminder that they should learn about them.
   - The member then elaborated a human analogy, stating *"The way I was thinking about this is similar to humans benefitting from coaches even though we can self-reflect"*.
- ****Predictor-corrector methods** pop up**: A member suggested considering **predictor-corrector methods** in context of the recent discussion.
   - The conversation alluded to the possibility of specialized lightweight models intervening only when needed, suggesting computational efficiency.
- **Realistic Null Distribution Notion Surfaces**: A member mentioned about **realistic null distribution** per feature, without using a surgical "leave a feature out" approach.
   - The member stated that *"I haven't seen it done for more complex features AFAIK, but it certainly gets at the notion of realistic null distribution per feature without using a surgical "leave a feature out" approach.*"


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1382735844657336432)** (14 messages🔥): 

> `EvalEval coalition, Standardized LM evaluation methods, Inspect standard, lm_eval multi-gpu progress bar` 


- **EvalEval Coalition Assembles Eval Infra People**: A collaborative effort is looking for eval infra people to join the **EvalEval coalition**, aiming to unify evaluations outputs and training, sharing them all into the same hub, and extracting evaluation data easily across experiments, and [a form was linked](https://forms.gle/6fEmrqJkxidyKv9BA).
- **Standardized LM Evaluation Methods Under Consideration**: There are plans on attempts at creating **new standardized LM evaluation methods** given the rise of reasoning models.
   - One member submitted the form, hoping to help in some capacity.
- **Inspect Standard Questioned for LM Evaluation**: A member questioned why the **Inspect standard** wasn't being used, linking to [inspect.aisi.org.uk](https://inspect.aisi.org.uk/).
- **lm_eval Multi-GPU Progress Bar Plagues Performance**: A member asked how the progress bar works in **lm_eval** under a multi gpu setting, as it seems to only track one GPU's progress.
   - They asked if they could see the progress of all GPUs.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1382774942533288029)** (2 messages): 

> `ChatGPT Projects, Canvas Updates, Model selector on mobile` 


- **ChatGPT Projects Gain New Powers**: Projects in **ChatGPT** are gaining new capabilities including **deep research support**, **voice mode support**, and **improved memory**.
   - These updates are rolling out to **Plus**, **Pro**, and **Team users** starting today, with improved memory exclusive to **Plus** and **Pro** users.
- **Canvas Gets Download Feature**: **Canvas** now supports downloads, allowing users to export documents as **PDF**, **docx**, or **markdown**.
   - When using **Canvas** to write code, it will export directly to the appropriate file type (e.g. **.py**, **.js**, **.sql**).
- **Model Selection Arrives on Mobile**: Users on mobile can now upload files and access the model selector within ChatGPT projects, enhancing the mobile experience.
   - This update aligns the mobile platform more closely with the desktop version, providing feature parity for on-the-go productivity.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1382450341051629701)** (182 messages🔥🔥): 

> `O3 Pro performance versus O3, Limits of LLMs, Google Ads expert AI agent, Discord activity drop, GPT-4o cost to train` 


- **O3 Pro vs. O3: A Generation Showdown**: Users are debating whether **O3 Pro**'s generation time is artificially inflated to disincentivize usage and cut down on compute.
   - Despite this, some agree that *O3 Pro is better than O3*, although some users are experiencing failures using **O3 Pro mode** in projects and are not getting document answers or chain of thoughts.
- **Apple Gets an Earful: Mediocrity Criticized in Interview**: A user shared a [YouTube interview](https://youtu.be/NTLk53h7u_k?si=VF8-zJZLQziFhpD_So) where a woman told **Apple** how mediocre they are nowadays.
   - The link comes amid discussions of **flying cars** and **flying taxis**.
- **LLMs Fail Simple Reasoning Tasks**: A paper showed that **LLMs failed when images are changed artificially**, leading to a discussion on whether LLMs can truly reason or are simply biased toward training data.
   - One user argued that LLMs are just *mimicking intelligence* and that achieving **AGI** will require more than just LLMs, and compared LLMs to **System 1** in the dual process theory of psychology, which is mostly reflex-based.
- **New OpenAI Updates**: Users are discussing a [new OpenAI update](https://help.openai.com/en/articles/6825453-chatgpt-release-notes) that includes **deep research and voice mode support in projects**.
   - However, there's disappointment as **Teams subscribers** seem to be excluded from certain features like improved project memory.
- **GPT-4o Training Costs Revealed**: A user estimated that **GPT-4o** cost around **$15 million** to train, sparking a discussion on the profitability of **Plus** subscriptions based on the real costs of inferencing the model.
   - This was prompted by a link to a [YouTube video](https://www.youtube.com/watch?v=QGwWJl7AZ6c) regarding **Veo 3**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1382489448335278141)** (22 messages🔥): 

> `GPT Quantization, ChatGPT for Language Learning, Free GPT Credits for Training, GPT Memory Across Custom GPTs` 


- **GPT Quantization Debated**: While one user suspected that a **GPT model was quantized**, citing its slowness compared to **o1 pro**, an employee stated that this was not the case, though this claim remains unverified.
   - The user found the new model *significantly slower than o1 pro*.
- **ChatGPT Aids Language Learning**: A member shared their success using **ChatGPT** to recognize, input, and output localized, dialect-specific language, including **idioms** and **cultural references**.
   - They emphasized the need for careful guidance and human verification to ensure correct translation of intent, noting that different models offer varying degrees of quality.
- **Hunting Free GPT Credits**: Users discussed obtaining free **GPT credits for training**, with one noting the discontinuation of **OpenAI's academy program**.
   - A member suggested exploring **Gemini's free API models** as a starting point, while another mentioned a now-defunct **Microsoft for Startups** program.
- **GPTs Memory: Hallucination or Bleed?**: A member reported experiencing potential memory bleed across custom **GPTs**, despite the understanding that memory isn't shared.
   - They found some *strong evidence* suggesting that the memory they are seeing is not hallucinations, they are seeing memory bleed across **custom GPTs**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1382437769317716102)** (26 messages🔥): 

> `Prompt Security, LLM leakage, Forbidden tokens, Recency bias, Adversarial prompt injection` 


- **Explicit Forbidden Tokens Amplify LLM Leakage Risk**: A member warned that [enumerating forbidden tokens](https://owasp.org) amplifies recency bias and increases the risk of **LLM leakage**, while best practices recommend externalizing enforcement of prohibited content.
   - Internalizing explicit restricted terms within the prompt increases the chance of unintended model leakage, particularly due to recency effects.
- **Math Config's Security Debated**: A member defended their 500-line math config against claims of a catastrophic leakage risk due to forbidden tokens, arguing that their testing showed no issues, even with extensive attempts to bypass the system.
   - Another member countered that a single forbidden token can create a catastrophic risk, and that the absence of evidence is not evidence of absence, especially with emergent risks.
- **Generalizing Descriptors Improves Safety**: A member suggested replacing explicit illegal tokens with generalized, legal descriptors such as *'illegal content,'* *'restricted material,'* or *'prohibited subjects'*, which aligns with OpenAI’s published guidelines and reduces the risk of emergent leakage.
   - This approach is safer and consistent with current industry standards, avoiding the inclusion of problematic words while still ensuring compliance.
- **Testing AI Ethics Requires Tests at Scale**: When asked about testing an A.I. to see if the claim of self governing moral values is true, it was noted that *you need tests at scale because of the diversity of human inputs*.
   - Streamlit is a good solution for scaling up.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1382437769317716102)** (26 messages🔥): 

> `Forbidden tokens, LLM Leakage, Prompt Security, AI moral values` 


- **Explicit forbidden tokens increases LLM Leakage**: Members discussed how enumerating forbidden tokens at the end of a config amplifies **recency bias** and increases the risk of **LLM leakage**.
   - They pointed out that best practices in prompt security recommend externalizing enforcement of prohibited content and using generalized references, rather than embedding explicit tokens in any user-visible or model-accessible memory.
- **Safety Engineering Minimizes Catastrophic Failure**: A member said that in prompt security, risk is multiplicative, not additive; a single critical flaw dominates any amount of complexity and the standard in safety engineering is not “*has it failed for me,*” but “*could it catastrophically fail under adversarial or unforeseen conditions?*”
   - They pointed to [OpenAI’s prompt-engineering guidelines](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api) (Tip #7) to recommend positive alternatives.
- **Testing AI for Moral Values Requires Scale**: A member asked about testing an AI to see if the claim of **self-governing moral values** is true.
   - Another member said you need tests at scale because of the diversity of human inputs.
- **Shotgun Spammers are back**: Members mentioned that the shotgun spammers are back.
   - No additional details were given.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1382449871583182950)** (220 messages🔥🔥): 

> `Manus Chat Mode, Veo 3 Video Generation, High Effort Mode Removal, Context Limits, Credit Usage and Pricing` 


- **Chat Mode Sparks Excitement and Debate**: Members are excited about the new **chat mode** in Manus, which is seen as a *gamechanger* to avoid wasting credits on simple questions and improve user experience by eliminating the need to switch between apps.
   - A moderator noted it will help reduce complaints about credit wastage, as users can get quick answers without using agent mode, while some argue that Manus is primarily for task completion, not general chatting.
- **Veo 3 Video Costs Spark Price Concerns**: Users discussed the cost of **Veo 3 video generation**, with one member reporting a charge of **300 credits for 8 seconds** of footage this morning then suddenly seeing **600 credits** as the price.
   - Another user calculated that a 5-minute video could cost **$47.50** and a 1-hour movie around **$570**, factoring in additional costs for music and sound from other providers.
- **"High Effort Mode" Auto-Enables, Confuses Users**: Members noticed the disappearance of the "High Effort Mode" selection, with one user stating, *I have never understood why high effort mode has to be manually selected in the first place, good to see it's become a natural process*.
   - It was clarified that **high effort mode** is now automatically enabled when the system deems it necessary, eliminating the manual switch.
- **Users Report Credit Wastage and Text Handling Errors**: Several users reported issues with **text handling** during tasks, leading to significant credit loss; one user saw **150 credits** disappear due to repeated text handling errors in the editor, and another saw Manus perform a task twice.
   - A member suggested starting a new session to mitigate this, while another observed that the issue seems related to slide uploads and is more prevalent since the introduction of chat mode.
- **AWS Outage Grounds Manus**: The Manus platform experienced issues due to a widespread **AWS outage**, affecting file uploads, task execution, and general functionality.
   - Other services like YouTube, Twitch, and even Discord image uploads were also affected by the outage, with members humorously pondering if aliens had landed in San Francisco.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1382462424652845088)** (85 messages🔥🔥): 

> `Dual GPUs in LM Studio, SSD lifespan concerns with swapping, Speculative decoding with Qwen models, Model updates in LM Studio, LLM conciseness training` 


- **Dual GPUs work wonders**: A user confirmed that dual GPUs work great in LM Studio, with one user running models on a **32+16GB** setup.
   - Concerns were raised that *experts don't seem too taxing*.
- **Swapping data kills SSDs**: A user intended to use **mmap()** swapping, leading to warnings about potentially killing an SSD due to excessive writes.
   - Another user countered that **SSDs** are rated in terabytes written, not read, but concerns about write-heavy swapping persisted.
- **Troubleshooting spec-decoding with Unsloth's Qwen**: Users discussed issues with speculative decoding using **Unsloth's Qwen3 models**, specifically trying to run the draft model on the GPU and the main model on the CPU.
   - It was noted that the draft model has to load into the GPU, and the crashing issues might not be related to processor selection, with a link to a relevant [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1kftu3s/draft_model_compatible_with/) being shared.
- **LM Studio doesn't auto-update**: A user asked if LM Studio automatically updates models when updates are available on Hugging Face, but it was confirmed that **LM Studio does not download models automatically**.
   - Model updates typically involve new generations in new repositories, making in-place updates rare and hindering lineage tracking.
- **LLMs have conciseness training**: A user looking for thorough, essay-like summaries was informed that **LLMs are trained to be concise** to save computational cost and avoid boring the user.
   - A suggestion was made to split the task: first get the structure, then ask for content for each bullet point.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1382476846188200118)** (81 messages🔥🔥): 

> `CPU vs GPU Setups, EPYC for LLMs, Strix Halo Memory, DeepSeek R1 on CPU, Unified Memory Comparison` 


- **CPUs Get the Cold Shoulder for LLMs**: Members discussed the viability of CPU-only setups for running LLMs, citing a [YouTube video](https://youtu.be/qV2bgTYLSX4?si=kPp3xmv23N2Y8Rcd) demonstrating poor performance with **Gemma 27B** achieving only **8 t/s**.
   - The consensus leaned towards CPUs being less efficient than GPUs due to memory bandwidth limitations, especially when considering the cost relative to equivalent VRAM in workstation GPUs.
- **EPYC CPUs Enter the Ring**: One member suggested exploring **EPYC CPUs** with high memory bandwidth (around **200GB/s**), providing a [link to an AliExpress listing](https://www.aliexpress.com/item/1005008461588060.html?spm=a2g0n.productlist.0.0.3f91e9dfvz9NZg) for a complete set priced at **€4300**.
   - The member speculated that an older **EPYC** or **Threadripper** could serve as a capable Proxmox server, acknowledging that workstation GPUs remain the inevitable choice for optimal LLM performance.
- **Strix Halo's Memory Bandwidth**: Members discussed the memory configuration of the **Strix Halo**, clarifying that it features **8 soldered channels** of LPDDR5x memory, offering higher bandwidth than standard desktop setups.
   - Despite the improved memory bandwidth, comparisons were made to Apple's RAM system as being the closest one can get to real HBM (High Bandwidth Memory).
- **DeepSeek R1 CPU Server Proves Pointless**: A member shared a [YouTube video](https://www.youtube.com/watch?v=v4810MVGhog) of a server build for running **DeepSeek**, but noted that it became unusable after a few prompts.
   - Others agreed, stating that the **CPU alone** in the build cost around **AUD$5k**, deeming it a *pointless exercise* for practical LLM applications.
- **Unified Memory comparison is "terrible"**: A member shared a [YouTube video comparing unified memory](https://youtu.be/Cn_nKxl8KE4?si=0-iQclmGi2UcWVNx), but was immediately declared *terrible*.
   - The commenter called out how the presenter was confusing soldered ram for slotted ram and went off-topic. They also explained that video failed to mention Strix Halo and its memory performance.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1382435964395454494)** (84 messages🔥🔥): 

> `Screenplay and Filmmaker AI Tools, No Inference Provider Error, Image-to-Text Models, Hugging Face Spaces Runtime Errors, LLM Distillation with Qwen` 


- **Crafting Screenplays with AI Tools**: A member inquired about creating **screenplay and film maker AI tools**, with another member pointing to several resources including [ACM Digital Library](https://dl.acm.org/doi/fullHtml/10.1145/3544548.3581225), [Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling), and [EQBench](https://eqbench.com/creative_writing.html).
   - They also noted a recent incident in Japan where **ChatGPT** was used to write a screenplay for a government-commissioned film, causing *some trouble*.
- **Inference API Troubles with Missing Providers**: A user reported getting a `No Inference Provider available` error for every model, including `nlpconnect/vit-gpt2-image-captioning`, while using the Inference API.
   - Another member suggested checking the available models and inference providers on the [Hugging Face settings page](https://huggingface.co/settings/inference-providers) and [chat interface](https://huggingface.co/chat/).
- **Navigating Image-to-Text Models on HF**: A member sought **image-to-text models** available in inference, and was advised that the text generation client doesn't support it, instead suggesting the use of inference providers directly or a Gradio client with Hugging Face Spaces.
   - Examples were given like the [FLUX.1-dev space](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev) and *Moondream by Vikhyat*.
- **Reported Runtime Error within HF Spaces**: A user inquired about reporting runtime errors in a Hugging Face Space, with another providing a [link to the model-card-regulatory-check discussions](https://huggingface.co/spaces/society-ethics/model-card-regulatory-check/discussions).
   - Additional contact methods were given such as directly reaching out to Hugging Face via email (`website@huggingface.co`) or [GitHub issues](https://github.com/huggingface/hub-docs/issues).
- **Challenges with AI Avatar Project due to Limited RAM**: A member is building an **AI avatar project** but faces crashes due to a **2GB video generation model** exceeding their **8GB RAM**, and is looking for local workarounds since *AWS GPU isn’t an option*. 
   - Suggestions included exploring **model quantization, frame splitting, and running low-vram modes**, plus a link to sign up as an *unregistered american company* for credits on the Nvidia developer program.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1382614870905851905)** (17 messages🔥): 

> `MCP servers study, AI avatar project, Deep3DFaceReconstruction and Face-vid2vid models, AI Agent course` 


- **Newbie Asks: MCP Servers Study and Course Order**: A new learner is seeking resources to master **MCP servers** and is unsure whether to take the **LLM course** or the **AI Agent course** first, being new to the field.
   - Another member suggests that it should be fine to directly start with the **AI Agent course**.
- **AI Avatar Project Struggles on Limited RAM**: A member is building an **AI avatar** project that allows real-time voice interaction but is facing crashes due to the **2GB video generation model** exceeding their **8GB RAM** limit.
   - They are seeking local workarounds like **model quantization**, **frame splitting**, or **low-vram modes** as AWS GPU isn't an option due to cost. Here is an example of their work so far: [AI Avatar Demo](https://cdn.discordapp.com/attachments/898619964095860757/1382642504796737546/2025_06_11_01.12.45.mp4?ex=684c8e6d&is=684b3ced&hm=8eaf0cacb8697427b52556211ada52bb54b99f3bb5a552671b35043e5166c323)
- **Deep3DFaceReconstruction and Face-vid2vid models cause frustration**: A member shared their frustration for failing to run **Deep3DFaceReconstruction**, **Face-vid2vid** and other **heavy models** without significant GPU resources, they mentioned that they also failed to generate images using **stable diffusion**.
   - Another member suggested using **Colab with H100 access** for around $10 to experiment, but the original poster aims to build something different and is looking for free ways to test before investing in GPU resources.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1382586342856200202)** (20 messages🔥): 

> `Unlimited Text To Video, LLM exploring its own awareness, Structural Interactions in the input, Hy-Bio Agent vs ChatGPT, Building an AI avatar voice-by-voice` 


- **Unlimited Text To Video Appears!**: A new [Unlimited Text To Video](https://huggingface.co/spaces/NihalGazi/Unlimited-Text-To-Video) app has been released; however, the resolution is low, and it's a bit slow.
   - The main upside is that the video generation is *unlimited*.
- **AERIS LLM Explores Self-Awareness!**: The **AERIS** project, which explores an LLM's self-awareness, has a chatbox available at [aeris-project.github.io/aeris-chatbox/](https://aeris-project.github.io/aeris-chatbox/) and an associated paper accepted to **ACL Main 2025** at [arxiv.org/abs/2403.13106](https://arxiv.org/abs/2403.13106).
- **Hy-Bio Agent Packs Punch over ChatGPT!**: The `Hy-Bio Agent` output is more focused on real working natural cure , powered by DATABASE contains **plants** ☘️ data.
- **Smolagents Power Claude Code Clone**: A member created a **Claude Code clone** using smolagents with a similar interface and most used tools, showcased on [X.com](https://x.com/niemerg/status/1932919266946203989).
   - Further work went into updating their [LLM OS agent](https://github.com/starsnatched/llmOS-Agent) which now utilizes the Linux VM more efficiently, with multimodal support coming soon.
- **Digital Twin AI Chatbots Emerge!**: **CloneMe** is an advanced AI platform that builds your digital twin—an AI that chats like you, remembers details, and supports multiple platforms, with code on [github](https://github.com/vibheksoni/cloneme) and available on [huggingface](https://huggingface.co/MasaFoundation).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1382752132221898875)** (2 messages): 

> `model explainability, heatmap visualization, Kaggle datasets` 


- **Heatmap Hopes Heat Up**: Members are seeking insights into **model explainability** and **heatmap visualization** for vision models.
   - Another suggested exploring [Kaggle datasets](https://www.kaggle.com/datasets) to find relevant resources for this task.
- **Kaggle: Data Goldmine**: One member suggested that [Kaggle](https://www.kaggle.com/) is the *best bet* to find datasets for vision models.
   - The suggestion was made in response to a question about experience with model explainability and heatmap visualization on Vision models.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

ut_nkezins: ive sent you friends request, maybe i could help you out
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1382446425048092736)** (21 messages🔥): 

> `requirements.txt, llama-index issues, certification path deadline, course sign up link broken, Tool Calling agents error` 


- **Devs Want `requirements.txt` and Python Version Guidelines**: Developers are requesting a `requirements.txt` file and guidelines on which **Python versions** to use for the course, especially when running the code locally instead of in **Colab**.
   - One dev ran into issues with **llama-index** module not being found, later discovering they named files in their directory the same as the libraries.
- **Deadline for Certification Path Looms**: New students starting the Agents course are asking whether it's still possible to enter the **certification path** with the deadline of **July 1st** approaching.
   - Experienced students suggest focusing on the core units and skipping optional content to meet the deadline, estimating the course takes around **20 hours**.
- **Sign-Up Link No Longer Functional**: Users are reporting that the sign-up link for the course, `https://bit.ly/hf-learn-agents`, is broken, preventing new registrations.
   - The link redirects to a broken page.
- **Tool Calling Agents: Debugging `FinalAnswerTool.forward()`**: One student is facing a `TypeError: FinalAnswerTool.forward() missing 1 required positional argument: 'answer'` when working with **Tool Calling agents**.
   - No immediate solutions were provided in the chat.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1382504016117891092)** (4 messages): 

> `GPU Engineering Role Preparation, Parallel Programming Patterns, PMPP` 


- **Parallel Patterns Prep Prompted**: A member is preparing for a **GPU engineering role** and is looking for video resources to brush up on **parallel programming patterns** beyond the *holy book*.
   - Another member inquired about the *holy book*, and was told it's **PMPP**.
- **GPU Role Seeker Asks for Parallel Programming Resources**: A member is studying for a **GPU engineering role**, seeking video resources for **parallel programming patterns** to supplement the *holy book*.
   - Another member asked about the *holy book*, with the original poster clarifying it refers to **PMPP**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1382670126574211172)** (2 messages): 

> `Conv1d performance optimization, Triton kernel optimization, LeetGPU challenge` 


- **Power of 2 Strikes Conv1d Kernel Performance**: A user reported poor performance (5th percentile) on a **conv1d** challenge, and another user suggested that **rounding up to the next power of 2** introduced unnecessary work, and was able to achieve **95th percentile** after addressing this issue.
- **Triton Kernel Snippet Highlights Input Blocking and Kernel Tiling**: A user provided a snippet of their **Triton kernel code** that shows they are splitting the input into blocks and tiling the kernel, with the goal of optimizing **conv1d** performance, loading blocks of `BLOCK_SIZE` using masks and accumulation loops.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1382763219495555344)** (5 messages): 

> `PTX modifiers, cache eviction policies, Blackwell library` 


- **PTX Modifiers for Load Instructions**: A member inquired about using **PTX modifiers** for load instructions, specifically regarding [cache eviction policies](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy).
   - The member noted they couldn't set eviction policies independently for **L1** and **L2** caches, resulting in compilation errors.
- **Cache eviction policies for GEMMs**: A member mentioned that **PTX modifiers** are useful for **GEMMs** with small batch sizes, where activations are forced to use the *evict-last* policy via **Triton**.
   - It was clarified that using `ld.global.L1::evict_last` isn't always enforced, depending on the data layout.
- **Blackwell library optimizes memory layout across L2 Cache**: One of the members was asked about the **Blackwell library** they created to optimize the memory layout across the **L2 cache**.
   - This was later clarified that it was a different member who created the library.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1382439477980692531)** (9 messages🔥): 

> `torch.func.functional_call, nn.Linear.from_pretrained, torch.compile and RL training, Mojo + PyTorch, torch.compile speedup` 


- **`torch.func.functional_call` makes its debut**: A member noted the integration of `torch.func.functional_call`, but doesn't think it solves the **integrated API problem**.
   - It was suggested that `functorch` can now be accessed through `torch.func`.
- **`nn.Linear.from_pretrained` proposal surfaced**: A member proposed a cleaner way to load pretrained weights into `nn.Linear` layers with `nn.Linear.from_pretrained(weight=param)` instead of the usual **3-4 lines of code using meta devices**.
   - The [VLLM project](https://github.com/vllm-project/vllm/pull/19265/commits/0a0b1a8e9a57d0f2f543e76c18b074544847cce4) currently uses the meta device approach.
- **`torch.compile` enters RL training arena**: A member is exploring using `torch.compile` to speed up an **RL training framework** with a common pattern of classes with state tensors stored as self attributes that are mutated by methods.
   - Another member suggested looking at the **trace to see if you see cuda graph launch**.
- **Mojo Kernel ideas sought for PyTorch**: A member is looking for ideas to make something _useful_ for the upcoming **Modular Hack Weekend** and considering writing some kernels for **Torch** using **Mojo**.
   - They asked for suggestions on what missing/wanted functionality implemented in **Mojo** would be nice to see in the **PyTorch kernel ecosystem**.
- **`torch.compile` gives speed boost for free**: A member reported that operations generated from `torch.compile` tend to run faster even when **nothing is being fused**.
   - For a convolution kernel, `torch.compile` ran in **0.0020 ms** compared to **1.7489 ms** for **native PyTorch**, and the compiled version seemed to be calling **extern_kernels.convolution** instead of **aten.convolution**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1382724344660103199)** (1 messages): 

> `Image analysis, Running AI on anything` 


- **AI Runs on Everything, Even Images!**: A user jokingly remarked that *AI runs on anything these days*, attaching an [image](https://cdn.discordapp.com/attachments/1215328286503075953/1382724344190337075/rn_image_picker_lib_temp_f68671eb-727c-448e-b9ff-e218ad0e04ef.jpg?ex=684cdaa5&is=684b8925&hm=3370f6c313acbf9a40221d8f5d46353ddead7c98ee96af78dbcd953bff69dd75&) to emphasize the point.
- **Another Topic Example**: This is a placeholder to satisfy the requirement of at least two topics.
   - Further details could be added here if available.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1382549860628041899)** (3 messages): 

> `OSDI 2025, AMD Advancing AI day` 


- **OSDI 2025 Attendance**: A member inquired about attendance at **OSDI 2025**.
- **AMD Advancing AI Day Meetup**: A member proposed a meetup at the **AMD Advancing AI day** with the speaker <@325883680419610631>.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1382440997438623825)** (3 messages): 

> `ROCm 6.4.1, MI50s, gfx906, rocprofiler-sdk, aqlprofile` 


- **ROCm 6.4.1 has issues with MI50s**: A user reported that **ROCm 6.4.1** didn't work with **MI50s**, throwing an error related to **HSA device gfx906** supporting only a single ISA.
   - They resolved this by reverting to **ROCm 6.3.3** and building *rocprofiler-sdk* and *aqlprofile* from source, plus downloading the [rocprof-trace-decoder](https://github.com/ROCm/rocprof-trace-decoder/).
- **User makes ROCm Work with Triton**: After working through the ROCm issues, a user stated that everything looks like it works with **Triton**.
   - They added that now they just have to learn how to use it, attaching a screenshot as proof.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1382546018922397727)** (5 messages): 

> `Efficient Attention Varieties, MLA Implementation, GQA with GLA Benchmarks, Distillation Loss Function` 


- **Efficient Attention Varieties Prompt Implementation**: A member inquired about plans for implementing efficient attention varieties like **MLA** and **GQA** in [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).
   - Another member suggested that **MLA decode** will be implemented through grouped latent attention and that **GQA with GLA** could be nice for benchmarks, but may require extra code.
- **MLA Decode Implementation via Grouped Lattent Attention**: A member confirmed that **MLA decode** will be implemented through grouped lattent attention, where the number of groups can be set to 1.
   - They also mentioned that **GQA with GLA** implemented could be nice for benchmarks, though it might require extra code.
- **Distillation Loss Function Implementation Proposed**: One member asked about the implementation of a distillation loss function like cosine_similarity and offered to work on it if it's not already being done.
   - The same member agreed with the prior opinion and linked to [issue 371](https://github.com/linkedin/Liger-Kernel/issues/371) and committed to working on it.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1382715955070636102)** (4 messages): 

> `cuBLASDx 0.4.0 Release, Ozaki Scheme for FP64, cuBLASDx Python Bindings, MathDx Package, cuBLASDx and CuTe DSL Integration` 


- **cuBLASDx 0.4.0 launches with Early Access**: NVIDIA has released [cuBLASDx 0.4.0](https://docs.nvidia.com/cuda/cublasdx/) as an Early Access library, callable from CUDA kernels, providing building blocks for GEMMs and data movement using **CuTe Algebra and Tensors** as the default datatype.
   - This release optimizes MMA/LDSM instructions, generates shared memory swizzles, chooses vectorized async copy instructions, and provides thread-local to global index mappings, aiming for peak performance on inference GPUs, with **UTCMMA/TMA** support coming soon.
- **Ozaki Scheme boosts FP64 Performance**: NVIDIA added an [Ozaki scheme explainer example](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/MathDx/cuBLASDx/16_dgemm_emulation), demonstrating how to boost **FP64 performance** 5-6x with **IMMA emulation** without precision loss.
   - One user inquired whether the **FP64 scheme** could be applied to other operations, particularly sparse LU solves.
- **cuBLASDx Offers Python Bindings**: The new version of cuBLASDx offers [Python bindings](https://docs.nvidia.com/cuda/cublasdx/python_bindings.html) in either **Numba** or **NVIDIA Warp** (tile_matmul).
- **MathDx expands set of Packages**: cuBLASDx is part of the MathDx package, which also includes **cuSolverDx** (dense matrix factorization), **nvCOMPDx** (data compression), **cuFFTDx** (Fast Fourier Transforms), and **cuRANDDx** (Random Number Generation), all with Python bindings and invokable from inside CUDA kernels.
   - All these libraries follow an *it just works* philosophy.
- **Users inquire on CuTe DSL Integration**: Users are asking for plans to support **cuBLASDx integration** with the new **CuTe DSL**.
   - Additionally, one user asked what it would take to bring the header-only, CUTLASS-based libraries to Julia.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1382445555216814091)** (3 messages): 

> `AMD GPU Support, Triton evals, Backward prop, Roadmap, Undergrad collaboration` 


- **Developers Adding AMD GPU Support**: Developers are actively working on adding newer features and more benchmarks including **AMD GPU support**, **Triton evals**, and **backward prop**.
- **Seeking Collab with Undergrads**: Developers expressed interest in collaborating with undergrads working on the project and inquired about a detailed roadmap.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1382545737736519721)** (5 messages): 

> `conv2d leaderboard, H100 results` 


- **Fifth Place Finishes Flood conv2d Leaderboard**: A member secured **5th place** on the `conv2d` leaderboard with two submissions, one at **338 ms** and another at **294 ms** on an **H100**.
   - Later submissions with IDs `32028` and `32029` were successful on the **H100**, clocking in at **25118 ms** and **25124 ms**, respectively.
- **H100 Honcho Hits Fourth**: A final submission achieved **4th place** on the `conv2d` leaderboard with a time of **47.8 ms** on an **H100**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1382843564450385921)** (1 messages): 

> `CUDA 12.9 Update 1, CC 10.3, B300` 


- **CC 10.3 equals B300 confirmed**: Members confirmed that **CC 10.3** is **B300** based on the [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cufft-release-12-9-update-1).
- **NVIDIA CUDA Toolkit Release Notes Highlighted**: The [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cufft-release-12-9-update-1) are essential for confirming hardware specifications.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1382462959145582623)** (43 messages🔥): 

> `Factorio capabilities/performance, FLE usability obstacles, Visual inputs usefulness, FLE interface alignment, FLE Docker image and mod` 


- **Factorio Boosts Superintelligence Journey**: Maximizing **capabilities/performance in Factorio** will assist in giving superintelligent systems responsibility of complex real-world systems, according to [this position paper](https://arxiv.org/pdf/2502.01492).
- **Community-Driven Projects Facilitated by FLE**: The first priority is to make the **Factorio Learning Environment (FLE)** good and usable enough to facilitate organic and diverse community-driven projects.
   - The second priority is to start or co-lead a flagship AlphaGo-like project (**AlphaFactorio**) focused on maximizing performance and extracting transferable algorithmic and/or engineering insights.
- **Tackling FLE Usability Obstacles**: Usability obstacles in **FLE**, such as **containers**, **pip installation/setup**, and the **environment interface**, hinder experiments in reasoning, planning, memory, multimodality, and multi-agent coordination.
   - The member noted that visual observations don't seem to help much in practice, making understanding this issue a valid scientific question for an Option 1 project.
- **Aligning FLE Interface with Human Developer Standards**: The objective is to align the **FLE interface** to resemble what a normal human developer would consider nice to work with, rather than optimizing for specific models.
   - The member also said he doesn't think LLMs can understand an interface which a normal developer wouldn't be able to understand and that what is considered impactful or meaningful is if it is rendered obsolete, invalid or redundant.
- **New FLE Docker Image and Mod POC Project Emerges**: A member created a POC project for a standalone **FLE docker image and mod** ([https://github.com/MortenTobiasNielsen/fle_suggestion](https://github.com/MortenTobiasNielsen/fle_suggestion)), inviting feedback.
   - He had issues integrating it into the main FLE codebase and it is therefore a standalone project for now.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1382750823519617226)** (44 messages🔥): 

> `AMD Conference meetup, AMD Advancing AI sign, Workshop 202, Fireside chat, Official Photo Link` 


- **Discordians Deploy to AMD Conference**: Several members, including Mark, Seb, GnSight, and az, attended the **AMD conference** and coordinated meetups at various locations, including in front of the *“AMD advancing AI sign”* and during breaks.
   - They also planned to meet at **Workshop 202 (Room 212C-D)** and the lunch area, attempting to sync despite interview schedules and crowded locations.
- **A Fireside Fail, A Photo Finish**: One member searched for others at the **fireside chat**, only to find *"no one here😭"*, later clarifying they were at the back of the room.
   - Attendees also sought the **official photo link** from the event after the award ceremony.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1382443885959057548)** (4 messages): 

> `CUTLASS Matmul Optimizations, EVT API Epilogues, Fused LoRA Layers` 


- **CUTLASS for Optimized Matmul**: A member inquired about using **CUTLASS** to optimize a matmul operation, aiming to chain several operations together like performing **Chebyshev** after the matmul and a partial max before writing the matrix to global memory ([issue #2393](https://github.com/NVIDIA/cutlass/issues/2393)).
   - CUTLASS samples accept **m, n, k** as arguments and benchmark the kernel to report the achieved flop/sec.
- **Exploring EVT API Epilogues**: A member suggested using the **EVT API** for expressing certain epilogues, linking to a [Colfax International research page](https://research.colfax-intl.com/epilogue_visitor_tree/).
   - However, it was noted that **EVT** requires expressing epilogues using a limited set of predefined operations, restricting the fusion of arbitrary CUDA code.
- **Fusing LoRA Layers to FP4 Matmul**: A member raised a question about the best approach for writing a fused epilogue that can't be expressed as an **EVT**, specifically wanting to fuse a **LoRA layer** to the end of an **FP4 matmul**.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1382482782814470314)** (2 messages): 

> `j4orz.ai, picograd, picoc, CUDA C extension` 


- **Zero to Hero Progresses with Math and Compilers**: The "Zero to Hero" project at [j4orz.ai](https://j4orz.ai/zero-to-hero/) has completed a significant portion of **Appendix A**, covering various mathematical concepts.
   - Work is now commencing on **Appendix B**, focused on implementing a **C compiler**.
- **CUDA C Extension Envisioned for Deep Learning**: The creation of a basic **CUDA C extension** is planned, drawing inspiration from the [picograd](https://github.com/j4orz/picograd) and [picoc](https://github.com/j4orz/picoc) codebases.
   - This extension aims to bridge the gap between traditional optimizing compilers and deep learning compilers.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1382447660916805633)** (6 messages): 

> `Usefulness of CS Degree, SVD Test Failure` 


- **Is CS Degree Still Good?**: A member inquired *if a CS degree is still useful?*
   - Another member responded that *it's not if you have to ask* and cautioned about going off-topic.
- **SVD Test Fails Due to Sign Flip**: A member reported that while running **SVD tests**, it fails due to a **sign flip**.
   - They shared a detailed traceback showing **mismatched elements** and **max absolute/relative differences**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1382459037635051570)** (55 messages🔥🔥): 

> `eigh() bounty, Tensor.norm(), LLM Discord Chatbot, tinygrad vs numpy accuracy, QR algorithm discrepancies` 


- **Ask and ye shall receive, eigh() needs bounty**: After discussion of adding **eigh()** to tinygrad, a member suggested it should be its *own bounty* due to complexity, and attached [A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf](https://cdn.discordapp.com/attachments/1070745817025106080/1382505351634616381/A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf?ex=684cb771&is=684b65f1&hm=c832598efa3830d558a0f9f457a339b3b05863e26df8e9f372fdd6419a7ba60e&).
- **LLM answers tinygrad questions!**: Members discussed integrating an **LLM chatbot** (like [getunblocked.com](https://getunblocked.com/)) with the Discord chat and codebase to answer questions by linking to specific conversations.
   - One member suggested stripping bulk and low-signal files to feed the rest as **input context to the LLM**.
- **Tensor Norm implementation surfaces**: One member asked about the existence of **tensor.norm()** in tinygrad and attached a [linalg.py file](https://cdn.discordapp.com/attachments/1070745817025106080/1382766636364333176/linalg.py?ex=684c5948&is=684b07c8&hm=2f078256ba98c1fd605435de76fd7f16dee8dbb1ea9ca817886a44df1e9b7338&) with a norm function.
   - The author admitted it *works in tinygrad 100%, just not as fast as numpy or as accurate*.
- **Numpy vs Tinygrad Accuracy differences**: A member highlighted **accuracy discrepancies** in float matmuls between **numpy** and **tinygrad**, specifically noting differences in the bottom left corner value of the output matrix.
   - Members attributed this to the use of different algorithms and the complexities of **floating-point operations**, where compilers and machines handle edge cases and optimizations differently.
- **QR Variance**: A member shared his struggles implementing the **QR algorithm**, noting variance issues with both the **Gram-Schmidt process** and **Householder Reflections**, compared to numpy's LAPACK package.
   - He stated that he's *going to skip doing the jacobian-method for symmetric matrices for eigh that got brought-up earlier in the conversation.*


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1382438964291698840)** (16 messages🔥): 

> `AI Audio Overview Customization, YouTube Channel Promotion, Podcast Compilation` 


- ****AI Audio Customization Remains Elusive****: A user inquired how to generate separate AI audio overviews for each topic in a source, but another user clarified that the **customization option disappears after the initial generation**.
   - The user suggested that the best approach is to prep sources and custom audio instructions, then generate a new notebook and audio, repeating the process until the desired result is achieved.
- ****Anime Edits Channel Courts Subs****: A user promoted their **YouTube channel** called *THE OP KID*, which features anime edits, seeking subscriptions.
- ****Podcast Powerhouse Shares Showcases****: A user shared a series of podcasts created using audio overviews, including one focused on **high school reading**, another on **missionaries and Bible characters**, a third on **cognitive distortions and psychology**, a fourth on **solving cold cases with AI**, and a final one diving into **thoughtful TV or film programs**.
   - The user also shared links to each podcast on Spotify and invited collaboration, noting that the "How-to" podcast surprisingly hit #1 in Morocco.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1382435670940717157)** (43 messages🔥): 

> `NotebookLM Age Restrictions, NotebookLM Feature Requests, Audio Overview Issues, Image as sources` 


- ****NotebookLM Age Restrictions Debated****: Users discussed whether **NotebookLM** has age restrictions, with one stating it's integrated with **Family Link** and has a minimum age of **13**.
   - Another user mentioned that the age policy might vary by region, particularly between **America** and the **EU**.
- ****Refresh All Sources Missing in NotebookLM****: A user inquired about a way to refresh all sources in a notebook but was informed that this is unavailable.
   - *Right now, you have to refresh every source manually*.
- ****MathJax Rendering Extension Launched****: A user created an open-source **Chrome extension** named **LaTeXLM** for **MathJax rendering** on **NotebookLM** and shared the [GitHub link](https://github.com/hachoj/LaTeXLM).
   - The extension allows users to enable local **Chrome extensions** without needing scripts.
- ****Audio Overview Struggles with Equations****: Several users reported that **NotebookLM's audio overview** struggles with reading equations.
   - It was also asked *what this uses to generate audio*.
- ****Support for Excel Files and Google Sheets Questioned****: A user asked if **NotebookLM** plans to support **Excel files** or **Google Sheets**, as they didn't see it supported or on the roadmap.
   - A user pointed out that there is *no public road map* and suggested to chime in on the feature request channel.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1382440797882024086)** (28 messages🔥): 

> `oscar-c project, Sam Altman vs Gary Marcus, system prompts for agents, Adaptive Resonance Theory (ART)` 


- ****Oscar-C** Project Seeks Testers**: A member has been trying to get people to check out their project called **'oscar-c'** which involves *cognitive architecture/xai/neurosymbolic AI*.
   - They invite those interested in a cool project to DM them for more information.
- **Altman and Marcus Duke It Out**: Members discussed a [post](https://x.com/sama/status/1932588741584957482) between **Sam Altman** and **Gary Marcus**.
   - One member stated that *99% of people arguing this is not "true" reasoning / intelligence etc can't even define it in a way that includes most of the humans too*.
- **Prompt Engineering Guide Surfaces**: A member asked for *resources they found really useful for* writing **system prompts for agents**.
   - Another member shared a link to [Humanity's Last Prompt Engineering Guide](https://www.forwardfuture.ai/p/humanity-s-last-prompt-engineering-guide).
- **Members Discuss Adaptive Resonance Theory**: A member mentioned the relevance of **Adaptive Resonance Theory (ART)** class of algorithms.
   - Another member shared a [survey paper](https://arxiv.org/abs/1905.11437) on the topic.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1382438772813201489)** (26 messages🔥): 

> `World Models, Energy Based Models, Active Inference, Predictive Coding` 


- **World Models Necessary for General Agents?**: A new [paper](https://arxiv.org/abs/2506.01622) argues that **general agents** capable of multi-step goal-directed tasks must learn a **predictive model of their environment**, which can be extracted from the agent's policy and requires increasing accuracy for improved performance.
   - The author's [blog post](https://richardcsuwandi.github.io/blog/2025/agents-world-models/) and the paper itself may not be the best introduction to the math/computation side of the author's research program.
- **Energy Based Models: LeCun's Pet Project Explored?**: Interest was expressed in understanding the hype around **energy-based models**, particularly due to **LeCun's** frequent mentions, with a request for a good introductory resource.
   - A suggestion was made that the interest lies more in the community and associated concepts like **Active Inference** and **Predictive Coding** than in the models themselves.
- **Predictive Coding as Gradient Descent Alternative**: Predictive coding is described as a localized **gradient descent** operation, deriving gradients from the *'pressure'* between upstream predictions and downstream errors, offering an alternative to backpropagation.
   - [This paper](https://arxiv.org/abs/2407.04117) was suggested as the best introduction, readable, survey of the field, and an awesome read.
- **Active Inference talk incoming!**: Members discussed the potential of a talk on **Active Inference**, highlighting its interesting formulation and connections to **Predictive Coding**.
   - A member volunteered to present a talk on Active Inference the following week.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1382442340744040449)** (4 messages): 

> `Mistral Compute, New video model` 


- **Mistral Enters the Compute Arena**: Mistral AI announced **Mistral Compute**, aiming to democratize AI infrastructure and provide tools and environments for everyone, moving beyond just building open models, as outlined in their [blog post](https://mistral.ai/news/mistral-compute).
- **Potential Veo3 Competitor Emerges**: A new video model was teased, potentially rivaling **Veo3**, with an [image](https://cdn.discordapp.com/attachments/853983317044756510/1382713167863611503/20250612_142749.jpg?ex=684cd03c&is=684b7ebc&hm=cdc33b41196f26909cf60ef8d205b9c583e1d5578b96bacdd65379f4a638059a&) suggesting a *huge upgrade*, though lacking sound.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1382449356749279305)** (36 messages🔥): 

> `Mojo on LeetGPU, FastxReader in Mojo, Modular Docs issues with nightly, Dynamic Dispatch/Type Lambdas in Mojo, String Performance Improvements in Mojo` 


- ****Mojo** Now Available on LeetGPU!**: **Mojo** is now supported on services like [LeetGPU](https://leetgpu.com/).
   - This makes **Mojo** more accessible for development and testing on various hardware configurations.
- ****FastxReader** iterator usage in Mojo**: A member shared an example of using a borrowing iterator over the input file into a dict-comp with **FastxReader** in **Mojo** using `rec.name[]: rec.seq[]` syntax.
   - The example code uses **Mojo** to map sequence names to sequences when reading a fastq file, highlighting the concise syntax enabled by the language.
- **Modular Docs Outdated for Nightly Builds**: A member encountered an error related to references, indicating a potential mismatch between the documentation and the **nightly** build of **Mojo**.
   - The docs seem to be ahead of the nightly changes, and another member suggested using the **nightly** build to align with the documentation, using `--index-url https://dl.modular.com/public/nightly/python/simple/`.
- **Dynamic Dispatch and Type System Limitations in Mojo**: Members discussed the current limitations of **Mojo**'s type system, noting the absence of dynamic dispatch, type lambdas, and type families.
   - One member suggested using [Variant](https://github.com/josiahls/firehose/tree/master/firehose) in lists as a workaround to handle this, but a full implementation of these features is not yet available.
- **Mojo String Operations Gain 40% Speed Boost**: Optimizations in the **nightly** branch have resulted in a **40%** performance improvement in **Mojo** compared to Python for a small string benchmark.
   - The code example shows how to split a string and the poster notes that the next stable release will see *"a lot of performance improvements"* for anyone doing fast string manipulation.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1382523524274454559)** (14 messages🔥): 

> `Memory Usage, Flex Attention, FSDP, TP, Loss Parallel` 


- **Memory Anomaly with Batch and Sequence Length**: A user observed that a `(bs, seqlen*8)` input (e.g. `(1, 64k)`) uses more memory than a `(bs*8, seqlen)` input (e.g. `(8, 8k)`) specifically with **flex attention** and **FSDP**, even though linear layers should be equivalent.
- **Large Inputs Trigger Memory Jump**: The increased memory usage seems to occur only with **very large inputs**, where peak memory jumps rapidly after a "tipping point", possibly due to memory allocation in blocks.
- **Loss Parallel Exposes Memory Savings**: The **loss parallel memory savings** appear to expose the issue, as without it, there is constant memory usage likely due to another bottleneck; the user attached a [chart](https://cdn.discordapp.com/attachments/1216353675744641096/1382573185916211200/image.png?ex=684c4dde&is=684afc5e&hm=0d498abf01433cd6a078a17e983947e7ac7e0590281a6728dec8a13a5fba2776) showing this jump.
   - It was hypothesized that the logits might be the source of a large allocation, but tokens per second remain stable, as shown in another [chart](https://cdn.discordapp.com/attachments/1216353675744641096/1382594289455857765/image.png?ex=684c6185&is=684b1005&hm=895b4043b33e49c5a7a89f047382cc18fc1416879e9d1c26b318c87bf345e22b).
- **_grouped_mm Speeds Up Finegrained MoE**: The use of `_grouped_mm` significantly improves the speed of **finegrained MoE**, with **Qwen3-30B-A3B** becoming almost as fast as **8B**, after being slower than **32B** with a for-loop implementation.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1382519865033625620)** (8 messages🔥): 

> `packing refactor, iterable datasets, contributing to torchtune, qwen3 and qwen2 builders` 


- ****Packing Refactor** Proposed for Iterable Datasets**: A member shared a proposal on packing refactor to work with [iterable datasets](https://github.com/pytorch/torchtune/pull/2819), aiming to support packing for **DPO**, **GRPO**, and **multimodal** applications.
   - The timeline includes gathering feedback, landing an iterable dataset RFC, and then landing the packing RFC, with an estimated completion by the *end of next week*.
- **Guidance on **Contributing** to TorchTune**: A member expressed interest in contributing to the repo, so they were directed to issues tagged with *"Community help wanted"* for clear action items and instructions.
   - The suggestion was made due to the user stating that *they have been working on an old forked repo for a while*.
- ****Qwen3** Mishap Uses **Qwen2** Builders!**: A member reported a problem where **Qwen3** uses **Qwen2** builders in [#2809](https://github.com/pytorch/torchtune/pull/2809).
   - To address this, they proposed either creating separate **Qwen3** component builders or adding the ability to pass custom attention in the **Qwen2** builder, with a preference for the latter to avoid extra boilerplate.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1382925426325852281)** (6 messages): 

> `Mistral 3.1 Small, Architectural Novelties, Multimodal Support, Devstral` 


- **Novel Architectural Nuances in Mistral 3.1?**: A member inquired about potential architectural novelties in **Mistral 3.1 Small** that might complicate its implementation for fine-tuning.
   - Another member suggested that **multimodal** capabilities, while not a novelty, could introduce complexity, especially considering *devstral* which may not heavily rely on multimodal features.
- **MultiModal implications for Devstral**: It was implied that multimodal capabilities are not a new thing, but it depends if the implementation has to support multimodal.
   - *Devstral* use case may not need the multimodal component.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1382434998908485703)** (18 messages🔥): 

> `Service Workers, MCP and Zapier, Playwright MCP Server, Hyper-MCP WASM` 


- ****Service Workers** Discussion**: Members discussed using **service workers** to run **MCP servers** directly in the browser, leveraging *postMessage* and dedicated threads.
   - It was noted that while feasible, running a **MCP server** compiled to **wasm** in the browser and serving it as a streamable HTTP via a service worker is possible but might be worse than creating the **MCP server** directly in JS.
- **Zapier MCP connection woes**: A user reported trouble connecting to **Zapier MCP** over OAuth, citing issues with their OAuth metadata server and **/token** endpoint producing 500 errors.
   - This issue has been raised for the attention of server authors.
- **Spinning up Playwright MCP Servers in the Cloud**: A member inquired about interest in a service that spins up **Playwright MCP Server** instances in the cloud, enabling access from anywhere, like **n8n workflows**.
   - This cloud-based setup would allow the **MCP Server** endpoint to be reached from any location.
- **Hyper-MCP pushing WASM for MCP servers**: **Hyper-MCP** is advocating for **WASM** to run **MCP servers** directly on the host, although some believe this is less than ideal.
   - The main concern is the loss of access to existing SDKs.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/)** (1 messages): 

whoateit: having some fun with this.
https://github.com/aj-geddes/fastfs-mcp
  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1382732461091590265)** (1 messages): 

> `Office Hour Reminder` 


- **Office Hours are Coming Soon!**: Reminder that the Office Hour will start in 15 minutes: [discord.com/events](https://discord.com/events/1059199217496772688/1379510205687140412).
   - Don't miss out on this **valuable opportunity**!
- **Office Hours Reminder #2**: Another reminder that the office hours will start soon and questions will be answered.
   - Come prepared to ask any questions and get involved.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1382751870141071402)** (3 messages): 

> `Order Completion Agent, LlamaCloud Stability, MistralAI Magistral` 


- **Order Completion Agent Fills Forms**: An AI assistant fills out structured forms as it talks to users using the new [Order Completion Agent with Artifact Editor example](https://t.co/oKxZxjajzZ).
- **LlamaCloud Bounces Back After Infrastructure Hiccup**: **LlamaCloud** is back up after some instability in our upstream infrastructure provider.
   - Check [LlamaIndex status page](https://t.co/IdecAksHiG) for the latest updates.
- **LlamaIndex Embraces MistralAI's Magistral**: **LlamaIndex** now supports @MistralAI's **Magistral** reasoning model in any agent workflow.
   - Dive into the details [here](https://t.co/ZsUEWMrnT4) and [here](https://t.co/QFONzaZRk0) to see how **Magistral** enhances your agent's reasoning capabilities.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1382784159533105294)** (14 messages🔥): 

> `Firebase outage, OpenRouter Down, Cloudflare Issues, GCP is down, BGP problems` 


- **Firebase goes offline!**: A member reported that **Firebase** is down, affecting authentication services, and noted that *Twitter is faster than the firebase status page* as evidenced by [this post](https://x.com/greghunkins/status/1933223568394846703?s=46).
   - Another member humorously expressed their frustration with *sadge*.
- **OpenRouter feels the Firebase effect!**: As a knock-on effect of the **Firebase** outage, a member reported that [OpenRouter is also down](https://news.ycombinator.com/item?id=44260810).
   - One member sarcastically noted that *a lot of things will be down if firebase is having an outage*.
- **GCP and Cloudflare suffer!**: A member reported that **GCP (Google Cloud Platform)** is down and **Cloudflare** is also experiencing issues.
   - Another member speculatively attributed the issues to **BGP (Border Gateway Protocol)** problems.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1382437048702603379)** (10 messages🔥): 

> `Multi-Model Re-Ranker, Amotions AI, Xarray-JAX library` 


- **Cohere Lacks Multi-Model Re-Ranker**: A member stated that *currently **Cohere** doesn’t have a multi model re-ranker* and suggested using **CLIP** and **openCLIP** as an alternative.
   - Another member is thinking of using **GPT-4.1** with structured output and custom prompt instead.
- **Amotions AI Seeks Technical Co-Founder**: The founder of **Amotions AI** is seeking a technical co-founder with an AI background to *take Amotions AI to the next level*, particularly its [real-time AI sales coach](https://www.amotionsinc.com/).
- **Xarray-JAX Library Development**: A member is building the **Xarray-JAX library** for **Google DeepMind** as part of **GSoC 2025**, noting that it's *effectively the first named tensor implementation in a deep learning framework*.
   - They believe this integration will be *really useful to the machine learning community* and welcome discussion about it.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1382815899920109780)** (1 messages): 

> `Reranking profiles` 


- **Reranking Profile Specs Shared**: A member requested information about reranking profiles, including the **number of docs, tokens per doc, and query tokens** used.
   - Another member responded and shared that their profile included **dozens of documents**, around **100 tokens per document** and around **20 query tokens** per query.
- **Clarification on Reranking Profile Specs**: The member who asked the question about the reranking profiles confirmed the specs shared were very helpful.
   - This validates the information shared and confirms it's useful for others seeking guidance on reranking configurations.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1382924505923719269)** (1 messages): 

> `Introductions, Company/Industry/University, Tech/Tools, Community Goals` 


- **Discord Greets Cohere Community**: The Cohere Community Discord server welcomes new members and encourages them to introduce themselves.
   - New members are prompted to share their **Company/Industry/University affiliation**, current projects, favorite **tech/tools**, and goals for joining the community.
- **Share Your Background with Community Members**: Each new member is encouraged to share what company, industry, or university they are coming from.
   - This is to help facilitate easier connection between members with similar backgrounds and interests.


  

---


### **Cohere ▷ #[🧭-status-feed](https://discord.com/channels/954421988141711382/1346652044181897307/1382824254009245916)** (1 messages): 

> `GCP Outage, Infrastructure Degradation` 


- **GCP Plunge Impacts Cohere**: Cohere reported experiencing an outage due to a [Google Cloud Platform (GCP) incident](https://ift.tt/on1ARP0) that may impact some of their services.
   - The team is actively monitoring the situation as of **June 12, 2025 at 12:02PM**.
- **Infrastructure Suffers Setback**: The specific affected component is identified as **Infrastructure**, which is experiencing degraded performance.
   - Further details can be found on the [Cohere Status Page](https://ift.tt/Ens6bma).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1382452337561567455)** (9 messages🔥): 

> `DSPy 3.0 Release, Referencing Input Fields in Docstrings, Agent Bricks Introduction` 


- **DSPy 3.0 Launches in Beta**: [DSPy 3.0](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1) has launched in **beta** and members are looking for a comprehensive overview of the changes.
   - One member inquired if **DSPy 3.0** is still in **beta**.
- **Docstring Dilemma: Input Field Referencing**: A member asked about the possibility of referencing the **input field** inside a **docstring** in DSPy, noting their relative newness to the framework.
   - Another member pointed out that *docstrings are just text*, but the original poster clarified their need for **dynamic jinja replacement**.
- **Agent Bricks Debut**: A member shared a screenshot and a link to a [Databricks blog post](https://www.databricks.com/blog/introducing-agent-bricks) introducing **Agent Bricks**.
   - No further discussion was given.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1382511149341212753)** (3 messages): 

> `AgentX summit, Research Paper Submission, Summit Attendance` 


- ****AgentX Summit Clarifications****: A user inquired about the **AgentX summit**, specifically regarding research paper submissions and attendance for finalists.
- ****Research Track Paper Submissions****: Finalists will be invited to the Summit poster session or give a talk at the Summit.
   - Submitting separately increases chances for additional consideration, using the [summit website](https://rdi.berkeley.edu/events/agentic-ai-summit).
- ****Attendance for Finalists Explained****: Finalists will be **invited separately** to the summit and do not need to register to attend.
   - It was recommended to register early to guarantee a spot, with potential ticket refunds for finalists.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1382437389514838108)** (3 messages): 

> `Model Speed, Token Count` 


- **Model Speed Concerns**: Members are finding that "thinking models are too slow".
   - Others inquired *why models are too slow* even when they are smaller, like **1GB, 2GB, or 4GB**.
- **Token Troubles**: The reason for model slowness is *Too many f*cking tokens*.
   - It was implied that a large number of tokens may be contributing to slower processing speeds.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1382853343185207366)** (1 messages): 

> `Windsurf Wave 10, UI/UX Upgrades, EU Cluster, Enterprise Offerings` 


- **Windsurf Drops Wave 10 with UI/UX Facelift**: Windsurf announced the release of **Wave 10**, featuring a fresh slate of **UI/UX upgrades**, along with new teams and enterprise offerings as covered in their [blogpost](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise).
   - The release includes new icons for `@-mentions` and file citations, codeblocks in the Cascade panel matching the IDE theme, a native terminal in the Cascade panel accepting user inputs, and a new Conversation History UI.
- **Windsurf Expands to EU with New Cluster**: Windsurf announced the launch of their **EU Cluster**, promising faster performance and catering to the rising demand from European enterprises, detailed in their [blog post](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise).
   - Details can be found in their [video on Youtube](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI) and change logs can be found at [https://windsurf.com/changelog](https://windsurf.com/changelog).


  