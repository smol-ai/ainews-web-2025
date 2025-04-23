---
id: 2c5b54c8-2b22-4462-8e1d-fc3c428eecab
title: '[AINews] Gemma 2: The Open Model for Everyone'
date: '2024-06-28T06:21:39.390033Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-gemma-2-the-open-model-for-everyone
---

<!-- buttondown-editor-mode: plaintext -->**Knowledge Distillation is all you need to solve the token crisis?**

> AI News for 6/26/2024-6/27/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**416** channels, and **2698** messages) for you. 
Estimated reading time saved (at 200wpm): **317 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

[Gemma 2 is out!](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) Previewed at I/O ([our report](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/)), it's out now, with the 27B model they talked about, but curiously sans 2B model. Anyway, it's good, of course, for its size - does lower in evals than Phi-3, but better in ratings on LMSys, just behind yi-large (which also [launched at the World's Fair Hackathon on Monday](https://x.com/01AI_Yi/status/1805431304999022812)):

 ![image.png](https://assets.buttondown.email/images/b7f2a5de-6997-4f87-8529-fa486a002b6d.png?w=960&fit=max) 

We have some small hints as to what the drivers might be:

- 1:1 alternation between local and global attention (similar to [Shazeer et al 2024](https://buttondown.email/ainews/archive/ainews-shazeer-et-al-2024/))
- Logit soft-capping per Gemini 1.5 and Grok
- GQA, Post/pre rmsnorm

But of course, data is the elephant in the room; and here the story has been KD:

> In particular, we focus our efforts on knowledge
distillation (Hinton et al., 2015), which **replaces
the one-hot vector seen at each token with the
distribution of potential next tokens computed
from a large model**. 
>
> This approach is often used
to reduce the training time of smaller models by
giving them richer gradients. In this work, we
instead train for large quantities of tokens with
distillation in order to simulate training beyond
the number of available tokens. Concretely, we
use a large language model as a teacher to train
small models, namely 9B and 2.6B models, on
**a quantity of tokens that is more than 50× the
compute-optimal quantity predicted by the theory (Hoffmann et al., 2022)**. Along with the models trained with distillation, we also release a 27B
model trained from scratch for this work.

At her [World's Fair talk on Gemma 2](https://youtubetranscript.com/?v=JVSKlEmUr0k&t=12381), Gemma researcher Kathleen Kenealy also highlighted the Gemini/Gemma tokenizer:

> "while Gemma is
trained on primarily English data the
Gemini models are multimodal they're
multilingual so this means the Gemma
models are super easily adaptable to
different languages. One of my
favorite projects we saw it was also
highlighted in I/O was a team of
researchers in India fine-tuned Gemma to
achieve state-of-the-art performance on
over 200 variants of indic languages
which had never been achieved before."

Fellow World's Fair speaker Daniel Han also called out the [attention-scaling](https://x.com/danielhanchen/status/1806372357684220308) that was only discoverable in the code:

 ![image.png](https://assets.buttondown.email/images/e54c2aaf-b43c-4a8a-858b-575fd5ccf8c1.png?w=960&fit=max) 

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI Models and Architectures**

- **New Open LLM Leaderboard released**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1805989925080219927) noted the new Open LLM Leaderboard evaluates **all major open LLMs**, with **Qwen 72B as the top model**. Previous evaluations have become too easy for recent models, indicating AI builders may have focused too much on main evaluations at the expense of model performance on others.
- **Alibaba's Qwen models dominate Open LLM Leaderboard**: [@clefourrier](https://twitter.com/clefourrier/status/1806016524496322950) highlighted that **Alibaba's Qwen models are taking 4 of the top 10 spots**, with the **best instruct and base models**. Mistral AI's Mixtral-8x22B-Instruct is in 4th place.
- **Anthropic releases Claude 3.5 Sonnet**: [@dl_weekly](https://twitter.com/dl_weekly/status/1806094847901659256) reported that Anthropic released **Claude 3.5 Sonnet, raising the bar for intelligence at the speed and cost of their mid-tier model**.
- **Eliminating matrix multiplication in LLMs**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1806108390231331260) shared a paper on **'Scalable MatMul-free Language Modeling'** which eliminates expensive matrix multiplications while maintaining strong performance at billion-parameter scales. Memory consumption can be reduced by more than 10× compared to unoptimized models.
- **NV-Embed: Improved techniques for training LLMs as generalist embedding models**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1806034882855875029) highlighted **NVIDIA's NV-Embed model**, which introduces new designs like having the LLM attend to latent vectors for better pooled embedding output and a two-stage instruction tuning method to enhance accuracy on retrieval and non-retrieval tasks.

**Tools, Frameworks and Platforms**

- **LangChain releases self-improving evaluators in LangSmith**: [@hwchase17](https://twitter.com/hwchase17/status/1806016844266197439) introduced a new LangSmith feature for **self-improving LLM evaluators that learn from human feedback**, inspired by @sh_reya's work. As users review and adjust AI judgments, the system stores these as few-shot examples to automatically improve future evaluations.
- **Anthropic launches Build with Claude contest**: [@alexalbert__](https://twitter.com/alexalbert__/status/1806040271672766756) announced a **$30K contest for building apps with Claude through the Anthropic API**. Submissions will be judged on creativity, impact, usefulness, and implementation.
- **Mozilla releases new AI offerings**: [@swyx](https://twitter.com/swyx/status/1806008516597146098) noted that **Mozilla is making a strong comeback with new AI offerings**, suggesting they could become an "AI OS" after the browser.
- **Meta opens applications for Llama Impact Innovation Awards**: [@AIatMeta](https://twitter.com/AIatMeta/status/1806048204452159848) announced the opening of applications for the **Meta Llama Impact Innovation Awards to recognize organizations using Llama for social impact** in various regions.
- **Hugging Face Tasksource-DPO-pairs dataset released**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1806100283853779166) shared the release of the **Tasksource-DPO-pairs dataset on Hugging Face, containing 6M human-labelled or human-validated DPO pairs** across many datasets not in previous collections.

**Memes and Humor**

- [@svpino](https://twitter.com/svpino/status/1806024708410015761) joked about **things they look forward to AI replacing**, including Jira, Scrum, software estimates, the "Velocity" atrocity, non-technical software managers, Stack Overflow, and "10 insane AI demos you don't want to miss".
- [@nearcyan](https://twitter.com/nearcyan/status/1806106875764801623) made a humorous comment about **McDonald's Japan's "potato novel"** (ポテト小説。。。😋).
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1806060587107164349) shared a meme about **"Perplexity at Figma config 2024 presented by Head of Design, @henrymodis"**.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Progress and Capabilities**

- **Low-energy LLMs**: Researchers have developed a high-performing large language model that can run on the [**energy needed to power a lightbulb**](https://news.ucsc.edu/2024/06/matmul-free-llm.html). This was achieved by [eliminating matrix multiplication in LLMs](https://arstechnica.com/information-technology/2024/06/researchers-upend-ai-status-quo-by-eliminating-matrix-multiplication-in-llms/), upending the AI status quo.

- **AI self-awareness debate**: Claude 3.5 has [passed the mirror test](https://www.reddit.com/gallery/1dpj0fw), a classic test for self-awareness in animals, sparking debate on whether this truly demonstrates self-awareness in AI. Another [post](https://www.reddit.com/gallery/1dpj4a2) on the same topic had commenters skeptical that it represents true self-awareness.

- **AI outperforming humans**: In a real-world "Turing test" case study, [**AI outperformed college students 83.4% of the time**](https://i.redd.it/eux68vd2b19d1.png), with 94% of AI submissions going undetected as non-human. However, humans still outperform LLMs on the MuSR benchmark according to [normalized Hugging Face scores](https://i.redd.it/f9q9a4h2819d1.png).

- **Rapid model progress**: A [timeline of the LLaMA model family](https://i.redd.it/35e3at3wr19d1.png) over the past 16 months demonstrates the rapid progress being made. Testing of the [Gemma V2 model in the Lmsys arena](https://www.reddit.com/r/LocalLLaMA/comments/1dovvbd/gemma_v2_in_the_lmsys_arena/) suggests an impending release based on past patterns. Continued [improvements to llama.cpp bitnet](https://github.com/ggerganov/llama.cpp/pull/8103) are also being made.

- **Skepticism of current LLM intelligence**: Despite progress, Google AI researcher Francois Chollet [argued current LLMs are barely intelligent](https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/) and an "offramp" on the path to AGI in a recent podcast appearance. An image of "The Myth of Artificial Intelligence" book [prompted discussion](https://i.redd.it/vv9by2jrpw8d1.jpeg) on the current state of AI.

**Memes and Humor**

- **AI struggles and quirks**: Memes poked fun at AI quirks, like an AI model [struggling to generate a coherent image](https://v.redd.it/5fohm4ft5w8d1) of a girl lying in grass, and [verbose AI outputs](https://i.redd.it/gj5c7gpsfv8d1.png). One meme [joked](https://i.redd.it/2lcebugjvx8d1.jpeg) about having the most meaningful conversations with AI.

- **Poking fun at companies, people and trends**: Memes took humorous jabs at [Anthropic](https://i.redd.it/4m9vvdeuix8d1.png), a [specific subreddit](https://i.redd.it/9oqilymzjw8d1.png), and people's [cautious optimism](https://i.redd.it/3haw02qs0x8d1.png) about AI. A [poem humorously praised](https://i.redd.it/ctuno8r9mz8d1.jpeg) the "Machine God".

**Other AI and Tech News**

- **AI copyright issues**: Major music labels [Sony, Universal and Warner are suing AI music startups](https://nypost.com/2024/06/24/business/sony-universal-warner-sue-ai-startups-suno-udio-for-infringement/) Suno and Udio for copyright infringement.

- **New AI capabilities**: OpenAI has [confirmed a voice mode](https://www.reddit.com/r/OpenAI/comments/1dp9rkl/openai_confirms_voice_mode_will_roll_out_starting/) for its models will start rolling out at the end of July. A Redditor briefly [demonstrated access](https://v.redd.it/n61ymct8qx8d1) to a GPT-4o real-time voice mode. 

- **Advances in image generation**: A new open-source super-resolution upscaler called [AuraSR based on GigaGAN](https://blog.fal.ai/introducing-aurasr-an-open-reproduction-of-the-gigagan-upscaler-2/) was introduced. The [ResMaster method](https://i.redd.it/x45h2la2ty8d1.png) allows diffusion models to generate high-res images beyond their trained resolution limits.

- **Biotechnology breakthroughs**: Two [Nature papers on "bridge editing"](https://i.redd.it/vaj4yhrmty8d1.jpeg), a new genome engineering technology, generated excitement. A [new mechanism enabling programmable genome design](https://x.com/pdhsu/status/1805981296276955571) was also announced.

- **Hardware developments**: A developer impressively [designed their own tiny ASIC](https://github.com/rejunity/tiny-asic-1_58bit-matrix-mul) for BitNet LLMs as a solo effort.

---

# AI Discord Recap

> A summary of Summaries of Summaries

## Claude 3.5 Sonnet

1. **Google's Gemma 2 Makes Waves**:

   - **Gemma 2 Debuts**: Google released [Gemma 2 on Kaggle](https://www.kaggle.com/models/google/gemma-2) in 9B and 27B sizes, featuring sliding window attention and soft-capping logits. The 27B version reportedly [approaches Llama 3 70B performance](https://x.com/reach_vb/status/1806343018640781675).

   - **Mixed Reception**: While the 9B model impressed in [initial tests](https://youtu.be/6SLuneidHYw), the 27B version [disappointed some users](https://youtu.be/vIKNRiVxWeo), highlighting the variability in model performance.

2. **Meta's LLM Compiler Announcement**:

   - **New Models for Code Tasks**: Meta [introduced LLM Compiler models](https://x.com/aiatmeta/status/1806361623831171318) built on Meta Code Llama, focusing on code optimization and compiler capabilities. These models are available under a permissive license for research and commercial use.

3. **Benchmarking and Leaderboard Discussions**:

   - **Unexpected Rankings**: The [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) saw surprising high rankings for lesser-known models like Yi, sparking discussions about benchmark saturation and evaluation metrics across multiple Discord communities.

4. **AI Development Frameworks and Tools**:

   - **LlamaIndex's Multi-Agent Framework**: LlamaIndex [announced llama-agents](https://twitter.com/llama_index/status/1806116419995844947), a new framework for deploying multi-agent AI systems in production with distributed architecture and HTTP API communication.

   - **Figma AI Free Trial**: [Figma AI is offering a free year](https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), allowing users to explore AI-powered design tools without immediate cost.

5. **Hardware Debates for AI Development**:

   - **GPU Comparisons**: Discussions across Discord servers compared the merits of NVIDIA A6000 GPUs with 48GB VRAM against setups using multiple RTX 3090s, considering factors like NVLink connectivity and price-performance ratios.

   - **Cooling Challenges**: Users in multiple communities shared experiences with cooling high-powered GPU setups, reporting thermal issues even with extensive cooling solutions.

6. **Ethical and Legal Considerations**:

   - **AI-Generated Content Concerns**: An [article about Perplexity AI](https://www.forbes.com/sites/rashishrivastava/2024/06/26/search-startup-perplexity-increasingly-cites-ai-generated-sources/) citing AI-generated sources sparked discussions about information reliability and attribution across different Discord servers.

   - **Data Exclusion Ethics**: Multiple communities debated the ethics of excluding certain data types (e.g., child-related) from AI training to prevent misuse, balanced against the need for model diversity and capability.

## Claude 3 Opus

**1. Advancements in LLM Performance and Capabilities**

- **Google's Gemma 2** models (9B and 27B) have been released, showcasing strong performance compared to larger models like [Meta's Llama 3 70B](https://x.com/_philschmid/status/1806343336292229234?s=46). The models feature sliding window attention and logit soft-capping.

- **Meta's LLM Compiler** models, built on Meta Code Llama, focus on code optimization and compiler tasks. These models are available under a permissive license for both [research and commercial use](https://x.com/aiatmeta/status/1806361623831171318?s=46).

- **Stheno 8B**, a creative writing and roleplay model from Sao10k, is now available on [OpenRouter](https://openrouter.ai/models/sao10k/l3-stheno-8b) with a 32K context window.

**2. Open-Source AI Frameworks and Community Efforts**

- **LlamaIndex** introduces [llama-agents](https://twitter.com/llama_index/status/1806116419995844947), a new framework for deploying multi-agent AI systems in production, and opens a waitlist for LlamaCloud, its fully-managed ingestion service.

- The **Axolotl** project encounters issues with [Transformers code affecting Gemma 2's sample packing](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718), prompting a pull request and discussions about typical Hugging Face bugs.

- **Rig**, a [Rust library for building LLM-powered applications](https://discord.com/channels/954421988141711382/1218409701339828245/1255618654142202156), is released along with an incentivized feedback program for developers.

**3. Optimizing LLM Training and Inference**

- Engineers discuss the potential of **infinigram ensemble** techniques to improve LLM out-of-distribution (OOD) detection, referencing a [paper on neural networks learning low-order moments](https://arxiv.org/abs/2402.04362).

- The **SPARSEK Attention** mechanism is introduced in a [new paper](https://arxiv.org/abs/2406.16747), aiming to overcome computational and memory limitations in autoregressive Transformers using a sparse selection mechanism.

- **Adam-mini**, an optimizer claiming to perform as well as AdamW with significantly less memory usage, is compared to NovoGrad in a [detailed discussion](https://x.com/ericzhang0410/status/1805814432595165567).

**4. Multimodal AI and Generative Modeling Innovations**

- **Character.AI** launches [Character Calls](https://blog.character.ai/introducing-character-calls/), allowing users to have voice conversations with AI characters, although the feature receives mixed reviews on its performance and fluidity.

- **Stable Artisan**, a Discord bot by Stability AI, integrates models like Stable Diffusion 3, Stable Video Diffusion, and Stable Image Core for [media generation and editing directly within Discord](https://discord.com/channels/1002292111942635562/1002292112739549196/1255599689118777468).

- The **Phi 3** model, mentioned in a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/), brings powerful AI chatbots to browsers via WebGPU.

## GPT4O (gpt-4o-2024-05-13)

1. **LLM Deployment and Training Optimization**:

   - **Hurdles in AI Deployment Leave Engineers Frustrated**: Engineers shared challenges in deploying custom models efficiently, with discussions focused on avoiding weights errors and optimizing parameters for hardware like the RTX 4090 using tools like [Koboldcpp](https://github.com/LostRuins/koboldcpp).

   - **Diving Into Flash Attention**: Members requested tutorials on [Flash Attention](https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b), an efficient technique for memory management in models, highlighting the need for better understanding of this optimization.

2. **Benchmarking and Performance Evaluation**:

   - **Yi Takes LLM Leaderboard by Storm**: The [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) sparked interest as models like Yi surprisingly rose to top ranks, challenging engineers to reassess their models' performances.

   - **Gemma 2's Mixed Reactions**: Excitement and skepticism surrounded **Gemma 2**—while some praised its innovations, others were unsure if it marked a significant leap. Comparisons with existing models were fueled by [benchmark analyses](https://x.com/danielhanchen/status/1806372357684220308).

3. **Open-Source AI Frameworks and Tools**:

   - **LlamaIndex Introduces llama-agents**: [LlamaIndex](https://twitter.com/llama_index/status/1806116419995844947) announced **llama-agents**, a multi-agent AI framework aiming to streamline production deployments; it includes distributed architecture and HTTP API communication.

   - **LangChain AI Discusses Endpoint Building**: Engineers shared examples of building LangChain endpoints with [documentation](https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events) showing proper use of `load_qa_chain()` and handling high-volume requests.

4. **AI Licensing and Ethical Considerations**:

   - **AI Training Ethics Stir Heated Debate**: Engineers in **LAION** deliberated over ethical training practices, debating whether to exclude child-related data to prevent misuse, while balancing the impact on model diversity and normal scene generation.

   - **Skepticism Towards AI Licensing Models**: Legal and practical concerns arose around the exclusive **Command-R** model via [OpenRouter](https://openrouter.ai/models/cohere/command-r/status), examining potential licensing misuse and enforcing compliance.

5. **Cutting-Edge AI Models and Innovations**:

   - **Meta Unveils LLM Compiler Models**: [Meta](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/) introduced the **Meta LLM Compiler** focusing on code optimization, with models built on extensive token corpuses for advanced compiler tasks.

   - **Innovative SPARSEK Attention Mechanism**: The **SPARSEK Attention** mechanism promises efficient long-sequence processing with linear complexity, as detailed in a new [paper](https://arxiv.org/abs/2406.16747), aiming to overcome typical self-attention limitations.

6. **Misc**

   - **Mojo Compiles and Executes Models with Ease**: Community members discussed [Mojo language challenges](https://www.modular.com/blog/mojo-vs-rust-is-mojo-faster-than-rust), highlighting object identity and self-referential type issues and the need for thorough GitHub documentation.

   - **Storage Requirements for Large Models Revealed**: Insights shared in **Nous Research AI** discussed the necessary hardware for running models like [DeepCoder V2](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF), indicating that substantial RAM and VRAM are required for efficient performance.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Yi Tops LLM Leaderboard**: [New benchmarks](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) have placed lesser-known models like Yi at surprising high ranks in the LLM leaderboard, intriguing the AI community.

**Rollout of Gemma 2 Stirs Excitement and Skepticism**: The release of **Gemma 2** has sparked enthusiasm and curiosity, particularly around its similarities with Grok. Notably, a [tweet dissecting Gemma 2's innovations](https://x.com/danielhanchen/status/1806372357684220308) became a focal point, despite some users questioning if the advancements mark a significant leap from previous models.

**Hurdles in AI Deployment and Training**: Discussions pointed to challenges and solutions in deploying custom models, with an emphasis on avoiding weights errors. AI engineers shared insights about saving and serving models using **Ollama** and suggested parameters adjustments for optimization on hardware like the RTX 4090, citing specific tools like [Koboldcpp](https://github.com/LostRuins/koboldcpp).

**Bugs and Support Discussed Ahead of the AI World's Fair**: The Unsloth AI team is gearing up for the AI World's Fair, planning to discuss open-source model issues and the new inclusion of **@ollama** support, as announced in [this tweet](https://x.com/danielhanchen/status/1806051465544536535).

**The Heat on ChatGPT**: ChatGPT became a contentious topic, with some community members calling it "literally fucking garbage" while others acknowledged its role in paving AI's path, despite **ChatGPT 3.5**'s accuracy issues. Problems with AI hardware overheating were also humorously lamented.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Multimodal RAG on the Horizon**: Excited chatter surrounded the development of a multimodal RAG article with anticipation for a groundbreaking outcome; however, the specifics such as models or results were not discussed.

**Entity Extraction Tools Evaluated**: Technical discussion identified shortcomings of BERT for NER, with members suggesting alternatives like GLiNER and NuExtract, which are touted for their flexibility in extracting non-predefined entities, pointing to community resources like [ZeroGPU Spaces](https://huggingface.co/spaces/enzostvs/zero-gpu-spaces).

**Skeptical Reception for Sohu AI Chip**: The community shared cautious skepticism regarding the claimed performance of Sohu's new AI chip, with members considering experimentation on Sohu's advertised service, despite no direct experience shared.

**Efficient Dynamic Diffusion Delivery**: Strategies for enhancing the performance of stable diffusion models were enthusiastically exchanged, notably including "torch compile" and leveraging libraries such as Accelerate and [stable-fast](https://github.com/chengzeyi/stable-fast) for improved inference times.

**AI Leaderboard Reflections**: The [Open LLM Leaderboard blog](https://huggingfile.co/spaces/open-llm-leaderboard/blog) spurred concerns about saturation in AI benchmarks, reflecting a sentiment for the community's drive for continuous improvement and new benchmarks.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**GPT Sibling Rivalry:** **CriticGPT** emerges to squash bugs in GPT-4’s code, boasting integration into OpenAI's RLHF pipeline for enhanced AI supervision, [official announcement details](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/).

**Claude vs. GPT-4o - The Context Window Showdown:** **Claude 3.5 Sonnet** is lauded for its coding prowess and expansive context window, overshadowing **GPT-4o**, which some claim lacks real omnimodal capabilities and faces slow response times.

**Beyond Traditional Text Chats:** Innovators employ **3.5 Sonnet API** and **ElevenLabs API** to drive real-time conversation, challenging the necessity of **ChatGPT** in certain contexts.

**Prompt Engineering Puzzles and Pitfalls:** Users exchange methods for few-shot prompting and prompt compression, with an eye on structuring prompts in YAML/XML for precision, and experimenting with "Unicode semiotics" for token-efficient prompts.

**Navigating the API Labyrinth:** Discussions focused on calculating prompt costs, seeking examples of knowledge bases for model training, gif creation challenges with GPT, deprecated plugin replacements, and the API's knack for struggling with certain word puzzles.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tensor Cores Lean Towards Transformers**: Engineers noted that while **tensor cores** on GPUs are generic, there is a tendency for them to be more "dedicated to transformers." Members concurred, discussing the wide applicability of tensor cores beyond specific architectures.

- **Diving Into Flash Attention**: A tutorial was sought on [Flash Attention](https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b), a technique for fast and memory-efficient attention within models. An article was shared to help members better understand this optimization.

- **Power Functions in Triton**: Discussions on **Triton** language centered around implementing *pow functions*, eventually using `libdevice.pow()` as a workaround. It was advised to check that **Triton** generates the optimal PTX code for pow implementations to ensure performance efficiency.

- **PyTorch Optimizations Unpacked**: The new [TorchAO 0.3.0 release](https://github.com/pytorch/ao/releases/tag/v0.3.0) captured attention with its quantize API and FP6 dtype, intending to provide better optimization options for PyTorch users. Meanwhile, the `choose_qparams_affine()` function's behavior was clarified, and community contributions were encouraged to strengthen the platform.

- **Sparsity Delivers Training Speed**: The integration of 2:4 sparsity in projects using [xFormers](https://pytorch.org/blog/accelerating-neural-network-training/) has led to a 10% speedup in inference and a 1.3x speedup in training, demonstrated on NVIDIA A100 for models like [DINOv2 ViT-L](https://github.com/facebookresearch/dinov2).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Infinigram and the Low-Order Momentum**: Discussions highlighted the potential of using **infinigram ensemble** techniques to boost LLMs' out-of-distribution (OOD) detection, referencing the "Neural Networks Learn Statistics of Increasing Complexity" [paper](https://arxiv.org/abs/2402.04362) and considering the integration of **n-gram** or **bag of words** in neural LM training.

- **Attention Efficiency Revolutionized**: A new **SPARSEK Attention** mechanism was presented, promising leaner computational requirements with linear time complexity, detailed in [this paper](https://arxiv.org/abs/2406.16747), while **Adam-mini** was touted as a memory-efficient alternative to AdamW, as per another [recent study](https://arxiv.org/abs/2406.16793).

- **Papers, Optimizers, and Transformers**: Researchers debated the best layer ordering for Transformers, referencing various arxiv papers, and shared insights on manifold hypothesis testing, though no specific code resources were mentioned for the latter.

- **Mozilla's Local AI Initiative**: There was an update on Mozilla's call for grants in **local AI**, and the issue of an expired Discord invite was resolved through a quick online search.

- **Reframing Neurons’ Dance**: The potential efficiency gains from training directly on neuron permutation distributions, using Zipf's Law and Monte Carlo methods, was a point of interest, suggesting a fresh way to look at neuron weight ordering.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GPU Face-off: A6000 vs. 3090s**: Engineers compared **NVIDIA A6000 GPUs with 48GB VRAM** against **quad 3090 setups**, citing NVLink for A6000 can facilitate 96GB of combined VRAM, while some preferred 3090s for their price and power in multi-GPU configurations.

- **Cost-Effective GPU Picks**: There was a discussion on budget GPUs with suggestions like **certified refurbished P40s and K80s** as viable options for handling large models, indicating significant cost savings over premium GPUs like the 3090s.

- **Specialized AI Chip Limitations**: The specialized **Sohu chip by Etched** was criticized for its narrow focus on transformers, leading to concerns about its adaptability, while Nvidia's forthcoming transformer cores were highlighted as potential competition.

- **AI Training Ethics & Data Scope**: There was a spirited debate regarding whether to exclude child-related data in AI training to prevent misuse, with some expressing concerns that such exclusions could diminish model diversity and impede the ability to generate non-NSFW content like family scenes.

- **NSFW Data's Role in Foundational AI**: The necessity of NSFW data for foundational AI models was questioned, leading to the conclusion that it's not crucial for pre-training, and post-training can adapt models to specific tasks, though there were varying opinions on how to ethically manage the data. 

- **AIW+ Problem's Complexity Unpacked**: The challenges of solving the AIW+ problem were explored in comparison to the common sense AIW, with the complexities of calculating family relationships like cousins and the nuanced possibilities leading to the conclusion that ambiguity persists in this matter.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Predictive Memory Formula Sought for AI Models**: Engineers are searching for a reliable method to predict memory usage of models based on the **context window size**, considering factors like *gguf metadata* and model-specific differences in attention mechanisms. Empirical testing was proposed for accurate measurement, while skepticism persists about the inclusivity of current formulas.

- **Chat GPT API Frontends Showcased**: The community shared new frontends for GPT APIs, including [Teknium’s Prompt-Engineering-Toolkit](https://github.com/teknium1/Prompt-Engineering-Toolkit) and [FreeAIChatbot.org](https://freeaichatbot.org), while expressing security concerns about using platforms like Big-AGI. The use of alternative solutions such as **librechat** and **huggingchat** was also debated.

- **Meta and JinaAI Elevate LLM Capabilities**: Meta's newly introduced models that optimize for code size in compiler optimizations and JinaAI's [PE-Rank](https://x.com/JinaAI_/status/1806331488247402524) reducing reranking latency, indicate rapid advancements, with some models now available under permissive licenses for hands-on research and development.

- **Boolean Mix-Up in AI Models**: JSON formatting issues were highlighted where **Hermes Pro** returned `True` instead of `true`, stirring a debate on dataset integrity and the potential impact of training merges on boolean validity across different AI models.

- **RAG Dataset Expansion**: The release of **Glaive-RAG-v1 dataset** signals a move toward fine-tuning models on specific use cases, as users discuss format adaptability with Hermes RAG and consider new domains for data generation to enhance dataset diversity while aiming for an ideal size of 5-20k samples.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **MacBook Air: AI Workhorse or Not?**: Members are debating the suitability of a MacBook Air with 6 or 8GB of RAM for AI tasks, noting a lack of consensus about Apple hardware's performance in such applications.
- **LoRA Training Techniques Under Scrutiny**: For better LoRA model performance, varying batch sizes and epochs is key; one member cites specifics like a combination of 16 batch size and 200 epochs to achieve good shape with less detail.
- **Stable Diffusion Licensing Woes**: Licensing dilemmas persist with SD3 and civitai models; members discuss the prohibition of such models under the current SD3 license, especially in commercial ventures like civit.
- **Kaggle: A GPU Haven for Researchers**: Kaggle is giving away two T4 GPUs with 32GB VRAM, beneficial for model training; a useful [Stable Diffusion Web UI Notebook](https://github.com/DEX-1101/sd-webui-notebook) on GitHub has been shared.
- **Save Our Channels: A Plea for the Past**: AI community members express desire to restore archived channels filled with generative AI discussions, valuing the depth of specialized conversations and camaraderie they offered.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Navigating Charting Library Trade-offs**: Engineers engaged in a lively debate over optimal charting libraries, considering static versus interactive charts, native versus browser rendering, and data input formats; the discourse centered on identifying the primary needs of a charting library.

- **Creative Containerization with Docker**: Docker containers for Mojo nightly builds sparked conversations, with community members exchanging tips and corrections, such as using `modular install nightly/mojo` for installation. There was also a promotion for the upcoming Mojo Community meeting, including video conference links.

- **Insights on Mojo Language Challenges**: Topics in Mojo discussions highlighted the necessity of reporting issues on GitHub, addressed questions about object identity from a Mojo vs Rust blog comparison, and observed unexpected network activity during Mojo runs, prompting a suggestion to open a GitHub issue for further investigation.

- **Tensor Turmoil and Changelog Clarifications**: The Mojo compiler's nightly build `2024.6.2705` introduced significant changes, like relocating the `tensor` module, initiating discussions on the implications for code dependencies. Participants called for more explicit changelogs, leading to promises of improved documentation.

- **Philosophical Musings on Mind and Machine**: A solo message in the AI channel offered a duality concept of the human mind, categorizing it as "magic" for the creative part and "cognition" for the neural network aspect, proposing that intelligence drives behavior, which is routed through cognitive processes before real-world interaction.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity API: Troubleshoot or Flustered?**: Users are encountering **5xx and 401 errors** when interfacing with the Perplexity AI's API, prompting discussions about the need for a *status page* and authentication troubleshooting.

**Feature Wish List for Perplexity**: Enthusiasts dissect Perplexity AI's current features such as image interpretation and suggest enhancements like **artifact implementation** for better management of files.

**Comparing AI's Elite**: The community analyzed and contrasted various AI models, notably **GPT-4 Turbo, GPT-4o,** and **Claude 3.5 Sonnet**; preferences were aired but no consensus emerged.

**Perplexity's Search for Relevance**: Shared **Perplexity AI pages** indicated interest in diverse topics ranging from mental health to the latest in operating systems, such as the performance boosts in **Android 14**.

**AI in Journalism Ethics Crosshairs**: An article criticized Perplexity for increasingly citing **AI-generated content**, sparking conversations about the reliability and privacy of AI-generated sources.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Grab Figma AI while It's Hot**: *Figma AI* is currently free for one year as shared by [@AustinTByrd](https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug); details can be found in the [Config2024 thread](https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

**AI Engineer World Fair Woes**: Members mentioned technical difficulties during an event at the *AI Engineer World Fair*, ranging from audio issues to screen sharing, and strategies such as leaving the stage and rejoining were suggested to resolve problems.

**LangGraph Cloud Takes Off**: [LangChainAI announced LangGraph Cloud](http://bit.ly/langgraph-cloud-beta-1), a new service offering robust infrastructure for resilient agents, yet some engineers questioned the need for specialized infrastructure for such agents.

**Conference Content Watch**: *AI Engineer YouTube channel* is a go-to for livestreams and recaps of the *AI Engineer World Fair*, featuring key workshops and technical discussions for AI enthusiasts, while conference transcripts are available on the [Compass transcript site](https://aie.compasswearable.com).

**Bee Buzzes with Wearables Update**: Wearable tech discussions included innovative products like [Bee.computer](https://bee.computer/), which can perform tasks like recording and transcribing, and even offers an Apple Watch app, indicating the trend towards streamlined, multifunctional devices.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LM Studio Lacks Critical Feature**: LM Studio was noted to lack support for **document-based training** or **RAG capabilities**, emphasizing a common misunderstanding of the term 'train' within the community.

**Code Models Gear Up**: **Claude 3.5 Sonnet** received praise within the **Poe** and **Anthropic** frameworks for coding assistance, while there is anticipation for upcoming **Gemma 2** support in LM Studio and llama.cpp.

**Hardware Dependency Highlighted**: Users discussed running **DeepCoder V2** on high-RAM setups with good performance but noted crashes on an **M2 Ultra Mac Studio** due to memory constraints. Additionally, **server cooling** and **AVX2 processor requirements** for LM Studio were topics of hardware-related conversations.

**Memory Bottlenecks and Fixes**: Members shared their experiences with VRAM limitations when loading models in LM Studio, providing advice such as disabling GPU offload and upgrading to higher VRAM GPUs for better support.

**Emerging AI Tooling and Techniques**: There's buzz around [Meta's new LLM Compiler models](https://go.fb.me/tdd3dw) and integrating **Mamba-2** into llama.cpp, showcasing advancement in AI tooling and techniques for efficiency and optimization.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Can't Print Streams Directly in Python**: A user emphasized that you cannot print stream objects directly in Python and provided a code snippet showing the correct method: iterate over the stream and print each token's content.

**Correctly Using LangChain for Relevant User Queries**: There were discussions on improving vector relevance in user queries with LangChain, with potential solutions including keeping previous retrieval in chat history and using `query_knowledge_base("Green light printer problem")` functions.

**Integrating LangChain with FastAPI and Retrieval Enhancements**: Community members shared documentation and examples on building LangChain endpoints using `add_routes` in FastAPI, and optimizing the use of `load_qa_chain()` for server-side document provisioning.

**Cutting-Edge Features of LangChain Expression Language**: Insights into LangChain Expression Language (LCEL) were provided, highlighting async support, streaming, parallel execution, and retries, pointing to the need for comprehensive documentation for a full understanding.

**New Tools and Case Studies for LangChain**: Notable mentions include the introduction of [Merlinn](https://github.com/merlinn-co/merlinn), an AI bot for troubleshooting production incidents, an [Airtable of ML system design case studies](https://www.evidentlyai.com/ml-system-design), and the integration of security features into LangChain with [ZenGuard AI](https://python.langchain.com/v0.2/docs/integrations/tools/zenguard). A YouTube tutorial was also highlighted, showing the creation of a no-code Chrome extension chatbot using Visual LangChain.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex's New AI Warriors**: [LlamaIndex announced](https://twitter.com/llama_index/status/1806116419995844947) **llama-agents**, a new multi-agent AI framework, touting a distributed architecture and HTTP API communication. The emerging LlamaCloud service commenced its waitlist sign-ups for users seeking a fully-managed ingestion service.

- **JsonGate at LlamaIndex**: Engineers engaged in a lively debate over the exclusion of JSONReader in LlamaIndex's default Readers map, concluding with a [pull request to add it](https://github.com/run-llama/llama_index/pull/14419).

- **When AIs Imagine Too Much**: LlamaParse, noted for its superior handling of financial documents, is under scrutiny for hallucinating data, prompting requests for document submissions to debug and improve the model.

- **BM25's Re-indexing Dilemma**: User discussions pointed out the inefficiency of needing frequent re-indexing in the BM25 algorithm with new document integrations, leading to suggestions for alternative sparse embedding methods and a focus on optimization.

- **Ingestion Pipeline Slowdown**: Performance degradation was highlighted when large documents are processed in LlamaIndex's ingestion pipelines, with a promising proposal of batch node deletions to alleviate the load.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **API Revenue Surpasses Azure Sales**: OpenAI's API now generates more revenue than Microsoft's resales of it on Azure, as highlighted by Aaron P. Holmes in a significant market shift revelation. Details were shared in [Aaron's tweet](https://x.com/aaronpholmes/status/1806312654505443347?s=46).

- **Meta's New Compiler Tool**: Unveiled was the **Meta Large Language Model Compiler**, aimed at improving compiler optimization through foundation models, which processes LLVM-IR and assembly code from a substantial 546 billion token corpus. The tool's introduction and research can be explored in [Meta's publication](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/).

- **Character Calls - The AI Phone Feature**: Character.AI rolled out **Character Calls**, a new feature enabling voice interactions with AI characters. While aiming to enhance user experience, the debut attracted mixed feedback, shared in [Character.AI's blog post](https://blog.character.ai/introducing-character-calls/).

- **The Coding Interview Dilemma**: Engineers shared vexations regarding excessively challenging interview questions and unclear expectations, along with an interesting instance involving claims of access to advanced voice features with sound effects in ChatGPT, mentioned by [AndrewCurran on Twitter](https://x.com/AndrewCurran_/status/1806178276001329373).

- **Patent Discourse - Innovation or Inhibition?**: The community debated the implications of patented technologies, from a chain of thought prompting strategy to Google's non-enforced transformer architecture patent, fostering discussions on patentability and legal complexities in the tech sphere. References include [Andrew White's tweet](https://x.com/andrewwhite01/status/1806347002126446736?s=46) regarding prompting patents.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Stheno 8B Grabs the Spotlight on OpenRouter**: OpenRouter has launched **[Stheno 8B 32K](https://openrouter.ai/models/sao10k/l3-stheno-8b)** by Sao10k as its current feature, offering new capabilities for creative writing and role play with an extended 32K context window for the year 2023-2024.

- **Technical Troubles with NVIDIA Nemotron Selection**: Users experience a hit-or-miss scenario when selecting **NVIDIA Nemotron** across devices, with some reporting 'page not working' errors while others have a smooth experience.

- **API Key Compatibility Query and Uncensored AI Models Discussed**: Engineers probe the compatibility of **OpenRouter API keys** with applications expecting OpenAI keys and delve into alternatives for uncensored AI models, including **Cmd-r**, **Euryale 2.1**, and the upcoming **Magnum**.

- **Google Gemini API Empowers with 2M Token Window**: Developers welcome the news of **[Gemini 1.5 Pro](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/)** now providing a massive 2 million token context window and code execution capabilities, aimed at optimizing input cost management.

- **Seeking Anthropic's Artifacts Parallel in OpenRouter**: A user's curiosity about Anthropic’s Artifacts prompts discussion on the potential for **Sonnet-3.5** to offer a similar ability to generate code through typical prompt methods in OpenRouter.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Innovative API Strategies**: Using the **Cohere API**, OpenRouter can engage in non-commercial usage without breaching license agreements; the community confirms that the API use circumvents the non-commercial restriction.

**Command-R Model Sparks Exclusivity Buzz**: The **Command-R** model, known for its advanced prompt-following capabilities, is available exclusively through [OpenRouter for 'I'm All In' subscribers](https://openrouter.ai/models/cohere/command-r/status), sparking discussions around model accessibility and licensing.

**Licensing Pitfalls Narrowly Avoided**: Debate ensued regarding potential misuse of **Command-R's** licensing by SpicyChat, but members concluded that payments to Cohere should rectify any licensing issues.

**Technical Troubleshooting Triumph**: A troubleshooting success was shared after a member resolved a **Cohere API script error** on Colab and PyCharm by following the official [Cohere multi-step tool documentation](https://docs.cohere.com/docs/multi-step-tool-use#step-2-ask-model-for-tool-calls-and-send-back-tool-results).

**Rust Library Unveiled with Rewards Program**: **Rig**, a new Rust library aimed at building LLM-powered applications, was introduced alongside a feedback program, rewarding developers for their contributions and ideas, with a nod to compatibility with Cohere's models.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Decoding the Neural Networks**: Engineers can join a **4-week**, free study group at Block's Mission office in SF, focusing on neural networks based on Andrej Karpathy's series. Enrollment is through this [Google Form](https://forms.gle/L4u3TMfTs5TjqWpt7); more details are available on the [event page](https://lu.ma/yzzespyu).

- **Open-Source Models Attract Interpreter Enthusiasts**: The Discord community discussed the best open-source models for local deployment, specifically **GPT-4o**. Conversations included potential usage with backing by **Ollama** or **Groq** hardware.

- **GitHub Policy Compliance Dialogue**: There's a concern among members about a project potentially conflicting with GitHub's policies, highlighting the importance of open conversations before taking formal actions like DMCA notices.

- **Meta Charges Ahead with LLM Compiler**: Meta's new **LLM Compiler**, built on **Meta Code Llama**, aims at optimizing and disassembling code. Details are available in the [research paper](https://go.fb.me/85zwgy) and the corresponding [HuggingFace repository](https://go.fb.me/tdd3dw).

- **Changing Tides for O1**: The latest release of O1 no longer includes the `--local` option, and the community seeks clarity on available models and the practicality of a subscription for usage in different languages, like Spanish.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Debugging Beware: NCCL Watchdog Meets CUDA Error**: Engineers noted encountering a **CUDA error** involving **NCCL watchdog thread termination** and advised enabling `CUDA_LAUNCH_BLOCKING=1` for debugging and compiling with `TORCH_USE_CUDA_DSA` to activate device-side assertions.

- **Gemma2 Garners Goggles, Google Greatness**: The community is evaluating **Google's Gemma 2** with sizes of 9B & 27B, which implements features like sliding window attention and soft-capped logits, showing scores comparable to **Meta's Llama 3 70B**. While the **Gemma2:9B** model received positive feedback in one [early test](https://youtu.be/6SLuneidHYw), the **Gemma2:27B** displayed disappointing results in its initial testing, as discussed in [another video](https://youtu.be/vIKNRiVxWeo).

- **Meta Declares LLM Compiler**: **Meta's announcement** of their **LLM Compiler** models, based on **Meta Code Llama** and designed for code optimization and compiler tasks, sparked interest due to their permissive licensing and reported state-of-the-art results.

- **Gemma2 vs Transformers: Round 1 Fight**: Technical issues with the Transformers code affecting **Gemma 2's** sample packing came to light, with a suggested fix via a [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718) and awaiting an upstream fix from the **Hugging Face team**.

- **Repeat After Me, Mistral7B**: An operational quirk was reported with **Mistral7B** looping sentences or paragraphs during full instruction-tuning; the issue was baffling given the absence of such patterns in the training dataset.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**PyTorch's Rise Captured on Film**: An [Official PyTorch Documentary](https://www.youtube.com/watch?v=rgP_LBtaUEc) was shared, chronicling **PyTorch’s development** and the engineers behind its success, providing insight for AI enthusiasts and professionals.

**Generic FPGA Design for Transformers**: A guild member clarified their FPGA design is not brand-specific and can readily **load any Transformer model from Huggingface's library**, a notable development for those evaluating hardware options for model deployment.

**Iterative Improvement on Tinygrad**: Work on integrating **SDXL** with **tinygrad** is progressing, with a contributor planning to streamline the features and performance before opening a pull request, a point of interest for collaborators.

**Hotz Hits the Presentation Circuit**: George Hotz was scheduled for an **eight-minute presentation**, details of which were not disclosed, possibly of interest to followers of his work or potential collaborators.

**Tinygrad Call for Code Optimizers**: A $500 cash incentive was announced for enhancements to tinygrad's [matching engine's speed](https://github.com/tinygrad/tinygrad/issues/4878), an open invitation for developers to contribute and collaborate on improving the project's efficiency.

**Deep Dive into Tinygrad's Internals**: Discussions included a request for examples of porting PyTorch's MultiheadAttention to tinygrad, a strategy to estimate VRAM requirements for model training by creating a **NOOP backend**, and an explanation of **Shapetracker’s** capacity for efficient data representation with reference to [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/). These technical exchanges are essential for those seeking to understand or contribute to tinygrad's inner workings.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Anthropic Announces Build-With-Claude Contest**: A contest focusing on building applications with **Claude** was highlighted, with reference to the [contest details](https://docs.anthropic.com/en/build-with-claude-contest/overview).

- **LLM Cover Letter Creation Queries**: Members have discussed fine-tuning a **language model** for generating cover letters from resumes and job descriptions, seeking advice on using test data to measure the model’s performance effectively.

- **Social Media Style Mimicry via LLM**: An individual is creating a bot that responds to queries in their unique style, using Flask and Tweepy for Twitter API interactions, and looking for guidance on training the model with their tweets.

- **Cursor Gains Ground Among Students**: Debates and suggestions have surfaced regarding the use of **OpenAI's Cursor** versus **Copilot**, including the novel idea of integrating Copilot within Cursor, with directions provided in a [guide to install VSCode extensions in Cursor](https://www.cursor.com/how-to-install-extension).

- **Credit Allocation and Collaborative Assistance**: Users requested assistance and updates concerning **credit allocation** for accounts, implying ongoing community support dynamics without providing explicit details.




---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1255600222344839341)** (269 messages🔥🔥): 

- **LLM Leaderboard makes a splash**: The LLM leaderboard has been refreshed with [new benchmarks](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) that generated excitement in the community, including unexpected high rankings for models like Yi.
- **Unsloth team heads to the AI World's Fair**: They plan to discuss "bugs in OSS models" and showcase new @ollama support in Unsloth AI at the event. The announcement link can be found [here](https://x.com/danielhanchen/status/1806051465544536535).
- **Apple Silicon support not immediate**: While there’s demand for Mac and AMD support, theyruinedelise clarified that Mac support is coming slowly as they lack Mac devices.
- **Gemma 2 generates buzz**: Extensive discussions around Google’s [Gemma 2](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) followed its release on Kaggle. [Hugging Face 4-bit models](https://huggingface.co/unsloth/gemma-2-9b-bnb-4bit) were quickly uploaded for efficient fine-tuning.
- **Meta releases LLM Compiler**: Meta announced a new family of models offering [code optimization and compiler capabilities](https://x.com/aiatmeta/status/1806361623831171318?s=46). These models can emulate compilers and were made available on [Hugging Face](https://go.fb.me/tdd3dw).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the learning rate resources in ...</li><li><a href="https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315">Gemma 2 Release - a google Collection</a>: no description found</li><li><a href="https://ai.google.dev/gemma/?utm_source=agd&utm_medium=referral&utm_campaign=blog-june&utm_content=gemma2">no title found</a>: no description found</li><li><a href="https://ollama.com/library/gemma2">gemma2</a>: Google Gemma 2 is now available in 2 sizes, 9B and 27B.</li><li><a href="https://x.com/OfficialLoganK/status/1806342850637918288">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Gemma 2 is here ❗️  Available to test in AI Studio right now: https://aistudio.google.com/</li><li><a href="https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3">UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/gemma-2-9b-bnb-4bit">unsloth/gemma-2-9b-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today</a>: no description found</li><li><a href="https://www.kaggle.com/models/google/gemma-2">Google | Gemma 2 | Kaggle</a>: Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.</li><li><a href="https://arxiv.org/abs/2406.02528">Scalable MatMul-free Language Modeling</a>: Matrix multiplication (MatMul) typically dominates the overall computational cost of large language models (LLMs). This cost only grows as LLMs scale to larger embedding dimensions and context lengths...</li><li><a href="https://x.com/danielhanchen/status/1806051465544536535">Tweet from Daniel Han (@danielhanchen)</a>: We&#39;ll be at @aiDotEngineer World&#39;s Fair today 2:20PM SF time YBB Salon 8 talking about bugs in OSS models.  We talked about Gemma & Phi-3 in our workshop - today is Llama-3! We&#39;ll also sho...</li><li><a href="https://x.com/danielhanchen/status/1806410668285030530">Tweet from Daniel Han (@danielhanchen)</a>: Uploaded pre-quantized 4bit bitsandbytes versions to http://huggingface.co/unsloth. Downloads are 4x faster & get &gt;1GB less VRAM fragmentation for QLoRA finetuning  Also install the dev HF pip inst...</li><li><a href="https://x.com/aiatmeta/status/1806361623831171318?s=46">Tweet from AI at Meta (@AIatMeta)</a>: Today we’re announcing Meta LLM Compiler, a family of models built on Meta Code Llama with additional code optimization and compiler capabilities. These models can emulate the compiler, predict optima...</li><li><a href="https://x.com/mindbranches/status/1806370172506091843?s=46">Tweet from MindBranches (@MindBranches)</a>: @AIatMeta Summary of the full research paper: &#34;Meta Large Language Model Compiler: Foundation Models of Compiler Optimization&#34;</li><li><a href="https://github.com/albertan017/LLM4Decompile">GitHub - albertan017/LLM4Decompile: Reverse Engineering: Decompiling Binary Code with Large Language Models</a>: Reverse Engineering: Decompiling Binary Code with Large Language Models - albertan017/LLM4Decompile</li><li><a href="https://x.com/danielhanchen/status/1806372357684220308">Tweet from Daniel Han (@danielhanchen)</a>: Just analyzed Google&#39;s new Gemma 2 release! The base and instruct for 9B & 27B is here!  1. Pre & Post Layernorms = x2 more LNs like Grok 2. Uses Grok softcapping! Attn logits truncated to (-30, 3...</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dpr487/gemma_2_is_live_on_kaggle_27b_9b/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/L">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1255634352457646154)** (17 messages🔥): 

- **ChatGPT sparks divisive opinions**: One member harshly criticized ChatGPT, describing it as *"literally fucking garbage"*. Another member pushed back, noting that while **ChatGPT 3.5** lacks accuracy, it nonetheless *"paved the way"* for modern AI advancements.
- **AI hardware and cooling woes**: A user humorously shared that their system, powered by 2x4090 GPUs pulling ~1000W, is overwhelming their AC. Another member related, mentioning that despite multiple radiators, they still experience thermal runaway.
- **Praise and confusion over Gemma 2 innovations**: A [tweet](https://x.com/danielhanchen/status/1806372357684220308) analyzing Google's **Gemma 2 release** was discussed, highlighting several innovative features borrowed from Grok. One user found the tricks like prepost ln and logit softcap fascinating but questioned if there was a major breakthrough from Grok.
- **Traditional vs. modern distillation methods**: Another user pointed out that the use of *Knowledge Distillation (KD)* in Gemma 2 seemed outdated, preferring "modern distillation" methods instead. They were impressed by the **two perplexity difference**, calling it *“😍”*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=R0X7mPagRiE">AI Engineer World’s Fair 2024 - Open Models track</a>: https://twitter.com/aidotengineer</li><li><a href="https://x.com/danielhanchen/status/1806372357684220308">Tweet from Daniel Han (@danielhanchen)</a>: Just analyzed Google&#39;s new Gemma 2 release! The base and instruct for 9B & 27B is here!  1. Pre & Post Layernorms = x2 more LNs like Grok 2. Uses Grok softcapping! Attn logits truncated to (-30, 3...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1255625481232056390)** (83 messages🔥🔥): 

- **Colab vs Ollama for Model Deployment**: Users discussed various options for deploying custom models after training, noting unresolved issues with weights errors on endpoints. One member recommended using [Ollama](https://discord.com/channels/1179035537009545276/1249414095359312087) for deployment, as demonstrated in an existing notebook.
- **Local Model Saving and Deployment Issues**: Members provided guidance on saving fine-tuned models in Google Colab, transferring them to Google Drive, and running locally using tools like [Koboldcpp](https://github.com/LostRuins/koboldcpp) for GGUF model deployment. One noted the model still present in RAM can be saved with `model.save_pretrained("lora_model")`.
- **Training Specifics for Performance Optimization**: A user sought optimal training settings for running Unsloth on an RTX 4090, with suggestions on adjusting batch sizes, learning rates, and considering higher parameter models like Qwen 32b. The use of embed tokens and lm_head in the training config was recommended for better handling languages like Swedish.
- **VRAM and Fine-Tuning Discussions**: Discussions covered the significant VRAM usage when fine-tuning the lm_head and embed tokens, making a case for its necessity in language-specific training. One user linked to a [notebook that trains Mistral for Korean](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing), highlighting the complexity and requirements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1xg3xz0J6BZPkUh8sor03_JrLHLKVGp-U?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/LostRuins/koboldcpp">GitHub - LostRuins/koboldcpp: A simple one-file way to run various GGML and GGUF models with KoboldAI&#39;s UI</a>: A simple one-file way to run various GGML and GGUF models with KoboldAI&#39;s UI - LostRuins/koboldcpp</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1255816018543120414)** (2 messages): 

- **Request for 4bit model upload**: A member requested the upload of a **4bit bnb version** of [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b/blob/main/README_en.md). They highlighted its superior performance in various benchmarks and advanced features like multi-language support and extended context length.
- **User intrigued by new model**: Another member expressed interest in the **GLM-4-9B** model, stating, *"oh what is this? looks pretty interesting."*



**Link mentioned**: <a href="https://huggingface.co/THUDM/glm-4-9b/blob/main/README_en.md">README_en.md · THUDM/glm-4-9b at main</a>: no description found

  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1255598480194408561)** (238 messages🔥🔥): 

- **Excitement for Multimodal RAG Article**: One member revealed they are working on a multimodal RAG article, eliciting excitement from others who think it's going to be "epic". *"it's gonna be epic ✨"*

- **Interest in Entity Extraction and GraphDB**: Members discussed various tools for entity extraction, with fulx69 sharing his approach using GraphDB and expressing disappointment with BERT for NER. Cursorop recommended and discussed using GLiNER and NuExtract for their flexibility in extracting non-predefined entities.

- **HuggingFace Resource for Free GPU Access**: A discussion unfolded about partnering with HuggingFace for free GPU access, where vipitis mentioned their ZeroGPU initiative and linked to [ZeroGPU Spaces](https://huggingface.co/spaces/enzostvs/zero-gpu-spaces). Fulx69 also noted that funding through platforms like Colab, Kaggle, AWS, etc., might be a necessity.

- **Skepticism Over Sohu AI Chip Performance**: Users expressed skepticism over the impressive performance claims of the new Sohu AI chip described in a detailed [Etched blogpost](https://www.etched.com/announcing-etched). Despite the skepticism, some showed interest in applying to their cloud service.

- **Tips for Speeding Up Stable Diffusion Inference**: Community members shared various strategies for improving the inference time of stable diffusion models, recommending libraries like Accelerate and stable-fast. Welltoobado suggested using the "torch compile" method, linking to [stable-fast GitHub repository](https://github.com/chengzeyi/stable-fast).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/models/google/gemma-2">Google | Gemma 2 | Kaggle</a>: Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.</li><li><a href="https://hexdocs.pm/fss/0.1.1/FSS.html">FSS — fss v0.1.1</a>: no description found</li><li><a href="https://huggingface.co/spaces/numind/NuExtract">NuExtract - a Hugging Face Space by numind</a>: no description found</li><li><a href="https://huggingface.co/spaces/urchade/gliner_multiv2.1">GLiNER-Multiv2.1 - a Hugging Face Space by urchade</a>: no description found</li><li><a href="https://tenor.com/view/cat-look-cat-look-at-camera-silly-cat-in-a-cage-gif-889392959852579879">Cat Look GIF - Cat Look Cat look at camera - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://filesystem-spec.readthedocs.io/en/latest/">fsspec: Filesystem interfaces for Python &mdash; fsspec 2024.6.0.post1+g8be9763.d20240613 documentation</a>: no description found</li><li><a href="https://huggingface.co/numind/NuExtract-large">numind/NuExtract-large · Hugging Face</a>: no description found</li><li><a href="https://www.etched.com/announcing-etched">Etched is Making the Biggest Bet in AI</a>: no description found</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>: no description found</li><li><a href="https://tenor.com/view/beach-vacation-artem-gif-26266521">Beach Vacation GIF - Beach Vacation Artem - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lu.ma/yzzespyu">AI Study Group @ Block: Andrej Karpathy&#x27;s Zero to GPT Hero · Luma</a>: ______ When signing up for this event, you will be asked via email to enrol in the study group via the following…</li><li><a href="https://tenor.com/view/joke-missed-over-your-head-gif-8604199">Joke Missed GIF - Joke Missed Over Your Head - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/thanos-memoji-gif-23490017">Thanos Memoji GIF - Thanos Memoji - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/etched/status/1805625693113663834?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from Etched (@Etched)</a>: Meet Sohu, the fastest AI chip of all time.  With over 500,000 tokens per second running Llama 70B, Sohu lets you build products that are impossible on GPUs. One 8xSohu server replaces 160 H100s.  Soh...</li><li><a href="https://huggingface.co/docs/hub/en/api#get-apidatasetsrepoidparquetconfigsplitnparquet">Hub API Endpoints</a>: no description found</li><li><a href="https://huggingface.co/urchade/gliner_base">urchade/gliner_base · Hugging Face</a>: no description found</li><li><a href="https://x.com/aiatmeta/status/1806361623831171318?s=46">Tweet from AI at Meta (@AIatMeta)</a>: Today we’re announcing Meta LLM Compiler, a family of models built on Meta Code Llama with additional code optimization and compiler capabilities. These models can emulate the compiler, predict optima...</li><li><a href="https://x.com/mindbranches/status/1806370172506091843?s=46">Tweet from MindBranches (@MindBranches)</a>: @AIatMeta Summary of the full research paper: &#34;Meta Large Language Model Compiler: Foundation Models of Compiler Optimization&#34;</li><li><a href="https://github.com/chengzeyi/stable-fast">GitHub - chengzeyi/stable-fast: Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs.</a>: Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs. - chengzeyi/stable-fast</li><li><a href="https://tenor.com/view/omg-wat-dafuq-huh-wth-gif-9101314">Omg Wat GIF - Omg WAT Dafuq - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://forms.gle/u397YGMioNFjvWXq6">Sohu Developer Cloud Application</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

neuralink: nah i didnt on a break, i just didnt post
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1255638555766821041)** (3 messages): 

- **Dive into self-similarity in ML**: A member shared an overview PDF from the [Papers We Love](https://github.com/papers-we-love/papers-we-love/blob/main/machine_learning/General-self-similarity--an-overview.pdf) repository. The document explores general self-similarity in machine learning concepts and applications.

- **Bewildering liquid neural nets (LNNs)**: Another member highlighted a [Medium article on Liquid Neural Nets (LNNs)](https://medium.com/@hession520/liquid-neural-nets-lnns-32ce1bfb045a), describing them as dynamic, adaptive neural networks useful for time series prediction. LNNs are noted for their robustness in noisy conditions and their ability to continue adapting even after initial training.

- **In-depth budget speech analysis**: A member shared a link to a [Budget Speech Essay](https://github.com/alidenewade/Publications/blob/main/Budget%20Speech%20Essay%20Final.pdf) hosted on GitHub. The essay provides a detailed analysis of a budget speech, contributing to the broader discussion of public financial management.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@hession520/liquid-neural-nets-lnns-32ce1bfb045a">Liquid Neural Nets (LNNs)</a>: A deep dive into Liquid Neural Networks, one of the most exciting recent developments in time series forecasting</li><li><a href="https://github.com/papers-we-love/papers-we-love/blob/main/machine_learning/General-self-similarity--an-overview.pdf">papers-we-love/machine_learning/General-self-similarity--an-overview.pdf at main · papers-we-love/papers-we-love</a>: Papers from the computer science community to read and discuss. - papers-we-love/papers-we-love</li><li><a href="https://github.com/alidenewade/Publications/blob/main/Budget%20Speech%20Essay%20Final.pdf">Publications/Budget Speech Essay Final.pdf at main · alidenewade/Publications</a>: Contribute to alidenewade/Publications development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1255633499558383738)** (12 messages🔥): 

- **PixUP Upscale speeds up image enhancement**: The [PixUP-Upscale](https://github.com/U-C4N/PixUP-Upscale) project features an ultra-fast, CPU-friendly design for enhancing image quality quickly. It's hosted on GitHub with a detailed description and open for contributions.

- **VoiceChat-AI enables local and cloud-based AI conversations**: The [voice-chat-ai](https://github.com/bigsk1/voice-chat-ai) project by bigsk1 allows users to speak with AI, either running locally using ollama or via cloud services like OpenAI and ElevenLabs.

- **Fast Whisper Server boosts transcription speeds**: A Space leveraging [Fast Whisper](https://github.com/SYSTRAN/faster-whisper) has been deployed, providing a faster variant of Whisper for audio transcription using the same API as OpenAI. Check out the [Faster Whisper Server](https://github.com/fedirz/faster-whisper-server) for details.

- **SimpleTuner gets a compression upgrade**: Version [v0.9.7.5](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7.5) of SimpleTuner includes updates like EMA offload improvements and disk compression for T5 embeds, enhancing efficiency and storage.

- **French Deep Learning course gets major updates**: Ricto3092 updated the French Deep Learning course with new material on YOLO, contrastive training, RNNs, GPT implementation, and more. The course is available on [GitHub](https://github.com/SimonThomine/CoursDeepLearning) and invites feedback and contributions.

- **Gemma2 model tests released on YouTube**: Volko76 shared YouTube videos testing the [Gemma2:9B](https://youtu.be/6SLuneidHYw) and [Gemma2:27B](https://youtu.be/vIKNRiVxWeo) models, showcasing their powerful performance.

- **Open-source on-device transcription app**: Hugo Duprez's [on-device transcription app](https://github.com/Hugo-Dz/on-device-transcription) using Ratchet is now open-source. Built with Svelte and Electron, it allows minimal and efficient speech-to-text conversion.

- **BLAST-SummarAIzer aids bioinformatics research**: Astrabert introduced a new space called [BLAST-SummarAIzer](https://huggingface.co/spaces/as-cle-bert/BLAST-SummarAIzer) for running BLAST on 16S rRNA bacterial sequences and summarizing results using LLMs. It aims to simplify interpreting complex BLAST search outputs for researchers.

- **Flight-Radar tracks flights in multiple languages**: Deuz_ai_80619 shared a multilingual, real-time flight tracking app called [Flight-Radar](https://github.com/U-C4N/Flight-Radar) built with Flask and JavaScript using the OpenSky Network API. Features include geolocation, adjustable search radius, and flight data download as a JPG.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/BLAST-SummarAIzer">BLAST SummarAIzer - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://youtu.be/6SLuneidHYw">Gemma2 First Test ! Incredible Results</a>: Today, we are going to install and test Gemma2 with ollama</li><li><a href="https://youtu.be/vIKNRiVxWeo">Gemma2:27B First Test ! Incredible Results</a>: Let&#39;s test the biggest version (27B) of the gemma2 release an hour ago by Google with ollama</li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7.5">Release v0.9.7.5 - compression for embed caching · bghira/SimpleTuner</a>: What&#39;s Changed  ema: offload to cpu, update every n steps by @bghira in #517 ema: move correctly by @bghira in #520 EMA: refactor to support CPU offload, step-skipping, and DiT models pixart: redu...</li><li><a href="https://github.com/U-C4N/PixUP-Upscale/">GitHub - U-C4N/PixUP-Upscale</a>: Contribute to U-C4N/PixUP-Upscale development by creating an account on GitHub.</li><li><a href="https://github.com/Hugo-Dz/on-device-transcription">GitHub - Hugo-Dz/on-device-transcription: A ready-to-use, minimal app that converts any speech into text.</a>: A ready-to-use, minimal app that converts any speech into text. - Hugo-Dz/on-device-transcription</li><li><a href="https://github.com/U-C4N/Flight-Radar">GitHub - U-C4N/Flight-Radar: A multilingual real-time flight tracking web application using the OpenSky Network API. Built with Flask and JavaScript, it allows users to view nearby flights, adjust search radius, and supports six languages. Features include geolocation, and the ability to download flight data as a JPG</a>: A multilingual real-time flight tracking web application using the OpenSky Network API. Built with Flask and JavaScript, it allows users to view nearby flights, adjust search radius, and supports s...</li><li><a href="https://github.com/bigsk1/voice-chat-ai">GitHub - bigsk1/voice-chat-ai: 🎙️ Speak with AI - Run locally using ollama or use OpenAI - XTTS or OpenAI Speech or ElevenLabs</a>: 🎙️ Speak with AI - Run locally using ollama or use OpenAI - XTTS or OpenAI Speech or ElevenLabs - bigsk1/voice-chat-ai</li><li><a href="https://huggingface.co/spaces/Iatalking/fast-whisper-server">Fastwhisper - a Hugging Face Space by Iatalking</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1255616368703701102)** (1 messages): 

- **Leaderboard worries about saturation**: The leaderboard is concerned with saturation, as highlighted in the linked blog post. The discussion referenced the [Open LLM Leaderboard blog](https://huggingface.co/spaces/open-llm-leaderboard/blog).
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1255604983022489650)** (2 messages): 

```html
<ul>
    <li><strong>Open Group Kickoff with Enthusiasm</strong>: A member started the discussion with an energetic “**Ghost!**”. This seemed to encourage further engagement in the channel.</li>
    <li><strong>Curiosity about Usage Impact</strong>: Another member, hayden_85058, asked, *"How do you feel about the effect of using it?"*. This indicates an interest in the practical outcomes or experiences from using a specific tool or method.</li>
</ul>
```
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1255931160765268049)** (1 messages): 

- **OpenAI introduces CriticGPT for bug detection**: An announcement revealed the training of a new model, **CriticGPT**, designed to catch bugs in GPT-4’s code. They are integrating these models into their RLHF alignment pipeline, aiming to assist humans in supervising AI on challenging tasks. More details can be found [here](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/).
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1255602967319416842)** (97 messages🔥🔥): 

- **Claude 3.5 sonnet shines in coding, GPT-4o lacks behind**: Members shared excitement over how well **Claude 3.5 Sonnet** performs at coding, with one saying it "gets it right first try" compared to needing multiple corrections with **GPT-4o**. Another pointed out that **Claude** offers a significantly larger context window, making it more suitable for large projects.

- **GPT-4o Omnimodal confusion sparks debate**: There was a disagreement over whether **GPT-4o** is truly omnimodal, with some members saying it is but has disabled some features, while others argue it's only omnimodal "on paper." One member noted that an **OpenAI** employee had clarified on Twitter that the model has disabled inputs/outputs.

- **Real-time conversations bypass ChatGPT**: A member claimed it's possible to achieve real-time conversations using APIs like the **3.5 Sonnet API** and **ElevenLabs API**, suggesting **ChatGPT** might not be needed anymore for some use cases.

- **OpenAI strict filters impact functionality**: Users expressed frustration over **OpenAI's** strict filters and regulations, which they argue slow down processing times and functionality. One member mentioned, "they seem to constantly shoot themselves in the foot with their strict filters and regulations".

- **Testing AI boundaries leads to bans**: There was a discussion about testing the limits of AI workarounds, with warnings that doing so could result in a ban. An official policy from [OpenAI](https://openai.com/policies/usage-policies) was cited, stating that circumventing safeguards could lead to account suspension or termination.
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1255615035405766748)** (25 messages🔥): 

- **Response speed issues with GPT-4**: Multiple users reported concerns about the slow responses of GPT-4. One user explicitly asked, *"Is there a problem with GPT-4? Because his responses are very slow!"*.

- **Calculating prompt costs accurately**: A user inquired about the industry standard for calculating prompt costs in a LLM app, mentioning they currently sum up all the token costs based on one user's input. They expressed uncertainty about whether to sample multiple users and average the costs.

- **Seeking knowledge base examples for GPTs**: A user requested examples of comprehensive knowledge bases to upload for GPT training, mentioning they wanted to create one for a client. Another user responded that every GPT uses different knowledge based on the creator's needs.

- **GIF generation with GPT**: Users discussed difficulties in generating proper animated GIFs with GPT, with one user saying they struggled to maintain the same character across frames. Another offered to share prompting ideas after further experimentation.

- **Deprecated plugins replaced by GPTs**: A query about using multiple plugin functions in a single chat was clarified with the information that plugins are deprecated and replaced by GPTs. Instead of starting a chat with multiple GPTs, users can use the `@mention` feature to call different GPTs within the same chat.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1255676134884900874)** (53 messages🔥): 

- **Seeking advice on few-shot prompting of ChatGPT with tools**: A user inquired about methods to improve few-shot prompting for ChatGPT, especially with custom tool use examples. They currently describe tool use as function calls and are looking for better approaches.

- **Dissecting Internal Workings of LLM Prompt Engineering**: A user expressed curiosity about learning the internal workings of prompt engineering and creating prompts using pure maths. They emphasized starting with numeric tokens and struggled to understand the language-based prompts.

- **Meta-prompting Guidance Given**: Another member provided a structured template for constructing prompts using YAML, XML, or JSON to manage hierarchical attention, recommending users to direct the AI to fill in these templates.

- **Frustration with Language vs. Mathematical Methods**: Extensive discussion ensued on generating prompts with numbers, with one user adamant about understanding a mathematical approach. Another user advised that effective communication with AI fundamentally requires using natural language.

- **Exploration of Prompt Compression**: A user introduced the topic of prompt compression, asking for experiences and references. The discussion highlighted "Unicode semiotics" which expands token count but uses fewer characters, useful for in-context learning despite the lack of extensive documentation.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1255676134884900874)** (53 messages🔥): 

- **Few-shot prompt strategies for ChatGPT tool usage**: A user seeks advice on improving ChatGPT performance with few-shot prompting by using function calls like `func(key1=value1, key2=value2)`. They wonder if there are better methods for this approach.
- **Understanding prompt engineering from basics**: A user requests guidance on learning prompt engineering from scratch using pure math and logic, expressing difficulties with complex structures and seeking deterministic process examples.
- **Template for effective prompting**: A user shares a detailed YAML/XMl prompt template for better hierarchy and control in prompts, emphasizing its effectiveness for ChatGPT and suggesting to let AI help build prompts.
- **Prompt compression and Unicode semiotics**: Prompt compression via Unicode semiotics was discussed, noting it uses fewer characters but more tokens, without available papers explaining this yet. The method helps in in-context learning despite higher token costs.
- **API struggles with word puzzles**: A user shares their difficulty with getting the API to solve unshuffled games like converting "edtPto lumAli" to "Potted Allium," mentioning multiple failed prompt attempts.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1255651086874972261)** (11 messages🔥): 

- **Debate on Tensor Core Specialization**: A discussion started about tensor cores' generality, with one member noting '*they probably mean "more dedicated to transformers" than that*'. Others agreed, highlighting that GPU tensor cores remain generic.
  
- **Structured Sparsity Inconsistency in PyTorch and NVidia**: A tweet shared highlighted an inconsistency between a new PyTorch post on structured sparsity and NVidia's PTX documentation. The user expressed interest in a *potential 4:8 sparsity support*.

- **Seeking Discounts for PyTorch Conference**: A member asked if anyone from Meta could provide discounts for the PyTorch conference as *$600 is a bit too much*.

- **Flash Attention Tutorial Request**: A member inquired about learning Flash Attention from scratch; another recommended a [Towards Data Science article](https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b) that delves into this power optimization technique.

- **GPU Specificity for AI and General Use**: Members discussed the versatility of GPUs, with one commenting, *"MM is generic enough to be useful elsewhere"* and another noting the complexity of GPUs doesn't always translate to increased efficacy due to transistor limitations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/trevorycai/status/1806025579222958317">Tweet from Trevor Cai (@trevorycai)</a>: There&#39;s an odd inconsistency between new PyTorch post on structured sparsity and NVidia&#39;s PTX documentation. It&#39;d be cool if 4:8 sparsity support existed!</li><li><a href="https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b">Flash attention(Fast and Memory-Efficient Exact Attention with IO-Awareness): A deep dive</a>: Flash attention is power optimization transformer attention mechanism that provides 15% efficiency
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1255625274389954710)** (46 messages🔥): 

- **Newcomer finds solution for pow implementation**: A new member shared an [issue](https://github.com/triton-lang/triton/issues/4190) about adding a pow function in Triton, later finding a workaround with `libdevice.pow(x,x+1)`. They expressed gratitude for the community's help, stating the issue is now closed.

- **Triton lacks native pow but has alternatives**: Members discussed Triton's absence of a pow function, suggesting the use of `exp` and `log` functions to simulate it. "*You can probably get away with it by doing multiplication*" was one suggestion.

- **CUDA pow kernels get compiled to exp and log**: It was highlighted that CUDA pow kernels, when using fast math, compile to a sequence involving `exp` and `log` instructions. The generated code for pow is more complex without fast math, implying a tradeoff between precision and performance.

- **Accuracy considerations for pow in deep learning**: The conversation emphasized that while the exp+log simulation of pow is less accurate, this inaccuracy is generally acceptable in deep learning contexts. "*The exp + log thing is inaccurate but it doesn't really matter for deep learning*" summarizes the sentiment.

- **Verification of Triton's generated PTX code suggested**: To ensure optimal performance, it was recommended to verify that Triton generates the 'fast' version of pow code. The original poster agreed to check, noting Triton currently uses CUDA's slower pow implementation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developer.download.nvidia.com/cg/pow.html">pow</a>: no description found</li><li><a href="https://triton-lang.org/main/python-api/triton.language.html">triton.language &mdash; Triton  documentation</a>: no description found</li><li><a href="https://github.com/triton-lang/triton/issues/4190">How to add a pow function in python.triton.language.core? · Issue #4190 · triton-lang/triton</a>: I tried to use pow operation in a triton.jitted function as: output = x + y**3 ^ However got AttributeError(&quot;&#39;tensor&#39; object has no attribute &#39;__pow__&#39;&quot;). In file python/trit...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

gau.nernst: https://github.com/efeslab/Atom
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1255630374743052378)** (16 messages🔥): 

- **TorchAO 0.3.0 Released**: Check out the new [TorchAO 0.3.0 release](https://github.com/pytorch/ao/releases/tag/v0.3.0) which includes a plethora of new features like a new quantize API, MX format, and FP6 dtype. This release aims at providing better performance and optimization for PyTorch.

- **Discussing `choose_qparams_affine()` Behavior**: There was a query about `choose_qparams_affine()` returning fewer dimensions when the block size equaled the dimension. It's confirmed that this is intentional, as detailed in [the source code](https://github.com/pytorch/ao/blob/c2f9b84604536a72804787001c1b63daae792ee9/torchao/quantization/quant_primitives.py#L335).

- **Contributions Welcome for New Members**: New community members are encouraged to start by using the project, identifying potential improvements, and suggesting changes. An example is [danielpatrickhug](https://github.com/pytorch/pytorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen) who plans to use custom static analysis tools.

- **Issue with `TORCH_LOGS="output_code"`**: An issue was raised where using `TORCH_LOGS="output_code"` does not output the code in certain scenarios, especially after adding quantization. Users are urged to report these issues and provide minimal reproducible snippets to aid in troubleshooting.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/blob/c2f9b84604536a72804787001c1b63daae792ee9/torchao/quantization/quant_primitives.py#L335">ao/torchao/quantization/quant_primitives.py at c2f9b84604536a72804787001c1b63daae792ee9 · pytorch/ao</a>: Create and integrate custom data types, layouts and kernels with up to 2x speedups with 65% less VRAM for inference and support for training - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/releases/tag/v0.3.0">Release v0.3.0 · pytorch/ao</a>: v0.3.0 Highlights We are excited to announce the 0.3 release of torchao! This release adds support for a new quantize API, MX format, FP6 dtype and bitpacking, 2:4 sparse accelerated training and b...</li><li><a href="https://github.com/pytorch/pytorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1255624486976225290)** (131 messages🔥🔥): 

- **Innovative Pointer Handling in CUDA**: In discussions about optimizing CUDA programming, an insightful explanation was given: *"think of ... as just applying the operation there to each element of that 'Parameter Pack'"* (in reference to using parameter packs for pointer initialization). It emphasized reducing redundancy in specifying types and explored alternative designs for utilities like `dtype_of`.

- **16 GPUs Training Triumph**: A member shared progress on training using **16 GPUs on Lambda**, noting how "glorious" it was and describing a significant speedup, almost achieving a 1.97x improvement. Despite the challenging setup process involving MPI errors and SSH keys, the efforts were successful.

- **Debate on CUDA Allocations and Performance**: Members discussed the trade-offs of various memory allocation methods, particularly regarding `cudaMallocHost()`, sharing experiences about its impact on performance. One suggestion was "`to get closer to an allocate-only-once state where you're guaranteed to not get OOMs after the first step`."

- **Async Checkpointing PR Review**: A PR for "Async State and Model checkpointing" was scrutinized, with concerns about added complexity and memory allocation impacts. One member argued, *"I'm not sure if this is worth it right now,"* hinting at a preference to defer such updates until post-version 1.0.

- **Gemma 2 AI Model Release Noted**: Excitement surrounded Google's release of **Gemma 2**, praised for beating larger models like Llama3 70B and Qwen 72B. Highlights included its efficient performance with fewer tokens, utilization of local and global attention layers, and innovative training techniques such as *"Soft attention capping"* and *"WARP model merging."* Links provided: [Reach_vb Gemma](https://x.com/reach_vb/status/1806343018640781675), [Danielhanchen Gemma](https://x.com/danielhanchen/status/1806372357684220308).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1806372357684220308">Tweet from Daniel Han (@danielhanchen)</a>: Just analyzed Google&#39;s new Gemma 2 release! The base and instruct for 9B & 27B is here!  1. Pre & Post Layernorms = x2 more LNs like Grok 2. Uses Grok softcapping! Attn logits truncated to (-30, 3...</li><li><a href="https://x.com/reach_vb/status/1806343018640781675">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s fucking gooo! Google just dropped Gemma 2 27B & 9B 🔥  &gt; Beats Llama3 70B/ Qwen 72B/ Command R+ in LYMSYS Chat arena & 9B is the best &lt; 15B model right now. &gt; 2.5x smaller than Llam...</li><li><a href="https://github.com/karpathy/llm.c/pull/652">Make cuDNN deterministic for Flash Attention backward by ademeure · Pull Request #652 · karpathy/llm.c</a>: cuDNN Frontend 1.5 released on June 13th added a new setting to make their backward algorithm deterministic which is disabled by default:  NVIDIA/cudnn-frontend@47d800c https://github.com/NVIDIA/cu...</li><li><a href="https://github.com/karpathy/llm.c/pull/644">Mixed dtypes by ngc92 · Pull Request #644 · karpathy/llm.c</a>: This is currently based on #635 , because there we ended up needing to reduce losses in  bf16. So this PR first does some malloc-and-point generalization for allowing multiple dtypes, and then chan...</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/commit/47d800ccd9449e1bbc255d64d794ae88d99b043d">Release notes for cudnn-frontend 1.5.0: (#81) · NVIDIA/cudnn-frontend@47d800c</a>: [New feature] With cudnn backend 9.2.0 and above, `Graph::check_support`
 can determine support check for runtime engines without invoking the
 nvrtc compiler. This allows users to check the suppor...</li><li><a href="https://github.com/karpathy/llm.c/pull/653">Matmul refactor using only cuBLASLt + GELU Fusion by ademeure · Pull Request #653 · karpathy/llm.c</a>: In preparation for FP8, this replaces all cuBLAS calls by cuBLASLt which is now wrapped by a single matmul_cublaslt() function. It also adds support for GELU fusion which can be controlled on the c...</li><li><a href="https://github.com/karpathy/llm.c/actions/runs/9699238303/job/26767716987?pr=653">Matmul refactor using only cuBLASLt + GELU Fusion · karpathy/llm.c@7082ab6</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/microsoft/mutransformers/tree/ed0e4af9700247e2067a131c2757a85133ab7d09">GitHub - microsoft/mutransformers at ed0e4af9700247e2067a131c2757a85133ab7d09</a>: some common Huggingface transformers in maximal update parametrization (µP) - GitHub - microsoft/mutransformers at ed0e4af9700247e2067a131c2757a85133ab7d09</li><li><a href="https://github.com/karpathy/llm.c/pull/651">Async optimizer state and model checkpointing by chinthysl · Pull Request #651 · karpathy/llm.c</a>: Additional feature to checkpoint optimizer state and model parameters using a non blocking background thread. Memcopy device buffers to pined host buffer in one shot and let the background thread d...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/1255923599504707594)** (1 messages): 

- **2:4 Sparsity Accelerates Neural Network Training**: Recent work with [xFormers](https://pytorch.org/blog/accelerating-neural-network-training/) using 2:4 sparsity shows a 10% speedup in inference for the [Segment Anything](https://github.com/pytorch/ao/tree/main/torchao/sparsity#segment-anything) project. Expanding this approach, they achieved a 1.3x speedup in model training on NVIDIA A100 with the `SemiSparseLinear` layer, cutting wall time by 6% for [DINOv2 ViT-L](https://github.com/facebookresearch/dinov2) training.

**Link mentioned**: <a href="https://pytorch.org/blog/accelerating-neural-network-training/">Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity</a>: Over the past year, we’ve added support for semi-structured (2:4) sparsity into PyTorch. With just a few lines of code, we were able to show a 10% end-to-end inference speedup on segment-anything by r...

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1255599091279466687)** (27 messages🔥): 

- **LLM OOD Detection Mechanisms**: Members discussed whether **infinigram ensemble** techniques could improve LLM performance. One user shared the [Extrapolates Predictably](https://arxiv.org/abs/2402.04362) paper, explaining that neural LMs learn low-order moments early in training and lose this ability later.
- **N-Gram and Neural LMs**: There was a debate about the efficacy of **n-gram / bag of words** features in improving neural LM training. One user clarified that infinigram isn't typically used for feature generation but is interested in literature supporting this.
- **New Research Papers Shared**: A user shared links to [papers on the distributional simplicity bias](https://arxiv.org/abs/2402.04362) and [Efron-Stein decomposition](https://arxiv.org/abs/2111.09375), sparking further discussion on their implications.
- **Mozilla Grant Call for Local AI Projects**: A member informed the channel about Mozilla's new call for grants focusing on **local AI**, including fine-tuning techniques and new UI paradigms. Another user noted that the Discord invite link in the PDF is dead, advising to email the contact person instead.
- **Seeking Mozilla AI Discord Invite**: A member was looking for a working invite link to the Mozilla AI Discord after finding the one in the PDF expired. Another user managed to find a related link through a quick online search, resolving the issue.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.04362">Neural Networks Learn Statistics of Increasing Complexity</a>: The distributional simplicity bias (DSB) posits that neural networks learn low-order moments of the data distribution first, before moving on to higher-order correlations. In this work, we present com...</li><li><a href="https://arxiv.org/abs/2111.09375">Hypercontractivity on High Dimensional Expanders: Approximate Efron-Stein Decompositions for $\varepsilon$-Product Spaces</a>: We prove hypercontractive inequalities on high dimensional expanders. As in the settings of the p-biased hypercube, the symmetric group, and the Grassmann scheme, our inequalities are effective for gl...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1255611941532209282)** (161 messages🔥🔥): 

- **Neurons and Weight Permutations In Theory**: A discussion centered around the idea that neuron activations form a predictable pattern, proposing the possibility of training directly on permutation distributions which might be refined using Zipf heuristics and Monte Carlo search. The hypothesis suggests potential efficiency gains by reframing the order of neuron weights.

- **SPARSEK Attention Reduces Complexity**: A new paper introducing **SPARSEK Attention**, which aims to overcome the computational and memory limitations of self-attention in autoregressive Transformers by using a sparse selection mechanism. The method promises linear time complexity and significant speed improvements ([arxiv link](https://arxiv.org/abs/2406.16747)).

- **Manifold Hypothesis Testing Inquiry**: A member inquired about available code for testing the manifold hypothesis on datasets, seeking advice on the best approach. No specific links or resources were provided.

- **Advancements and Comparisons in Optimizers**: The introduction of **Adam-mini**, an optimizer that claims to perform as well as AdamW but with significantly less memory usage. A detailed comparative discussion with NovoGrad highlights architectural choices and practical considerations ([arxiv link](https://arxiv.org/abs/2406.16793)).

- **Debate on Optimal Layer Ordering in Transformers**: A robust debate on whether linear attention or sliding window attention should be preferred in hybrid models. The conversation references several recent papers and presentations with diverging views on the optimal layer ordering strategy in large-scale models ([additional arxiv link](https://arxiv.org/abs/2405.05254v2)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the learning rate resources in ...</li><li><a href="https://arxiv.org/abs/2406.17711">Data curation via joint example selection further accelerates multimodal learning</a>: Data curation is an essential component of large-scale pretraining. In this work, we demonstrate that jointly selecting batches of data is more effective for learning than selecting examples independe...</li><li><a href="https://arxiv.org/abs/1911.03864">Improving Transformer Models by Reordering their Sublayers</a>: Multilayer transformer networks consist of interleaved self-attention and feedforward sublayers. Could ordering the sublayers in a different pattern lead to better performance? We generate randomly or...</li><li><a href="https://arxiv.org/abs/2406.13155">Convolutional Kolmogorov-Arnold Networks</a>: In this paper, we introduce the Convolutional Kolmogorov-Arnold Networks (Convolutional KANs), an innovative alternative to the standard Convolutional Neural Networks (CNNs) that have revolutionized t...</li><li><a href="https://arxiv.org/abs/2406.18532">Symbolic Learning Enables Self-Evolving Agents</a>: The AI community has been exploring a pathway to artificial general intelligence (AGI) by developing &#34;language agents&#34;, which are complex large language models (LLMs) pipelines involving both ...</li><li><a href="https://x.com/ericzhang0410/status/1805814432595165567">Tweet from YushunZhang (@ericzhang0410)</a>: Thanks a lot for mentioning NovoGrad! We have updated our paper  https://arxiv.org/pdf/2406.16793 and discussed their difference with Adam-mini. In short, there are at least two-fold major differences...</li><li><a href="https://arxiv.org/abs/2406.17245">Unlocking Continual Learning Abilities in Language Models</a>: Language models (LMs) exhibit impressive performance and generalization capabilities. However, LMs struggle with the persistent challenge of catastrophic forgetting, which undermines their long-term s...</li><li><a href="https://arxiv.org/abs/2406.07887">An Empirical Study of Mamba-based Language Models</a>: Selective state-space models (SSMs) like Mamba overcome some of the shortcomings of Transformers, such as quadratic computational complexity with sequence length and large inference-time memory requir...</li><li><a href="https://arxiv.org/abs/2406.14596">ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights</a>: Large-scale generative language and vision-language models (LLMs and VLMs) excel in few-shot in-context learning for decision making and instruction following. However, they require high-quality exemp...</li><li><a href="https://arxiv.org/abs/2406.16747">Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers</a>: Accommodating long sequences efficiently in autoregressive Transformers, especially within an extended context window, poses significant challenges due to the quadratic computational complexity and su...</li><li><a href="https://x.com/BlancheMinerva/status/1741855005601141091">Tweet from Stella Biderman (@BlancheMinerva)</a>: Many people seem to think they can&#39;t do interesting LLM research outside a large lab, or are shoehorned into crowded topics. In reality, there are tons of wide-open high value questions. To prove ...</li><li><a href="https://arxiv.org/abs/2405.05254v2">You Only Cache Once: Decoder-Decoder Architectures for Language Models</a>: We introduce a decoder-decoder architecture, YOCO, for large language models, which only caches key-value pairs once. It consists of two components, i.e., a cross-decoder stacked upon a self-decoder. ...</li><li><a href="https://github.com/google/gemma_pytorch">GitHub - google/gemma_pytorch: The official PyTorch implementation of Google&#39;s Gemma models</a>: The official PyTorch implementation of Google&#39;s Gemma models - google/gemma_pytorch</li><li><a href="https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_novograd.cu#L108>).">apex/csrc/multi_tensor_novograd.cu at master · NVIDIA/apex</a>: A PyTorch Extension:  Tools for easy mixed precision and distributed training in Pytorch - NVIDIA/apex
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1255602423666446419)** (7 messages): 

- **Understanding Hopfield Networks in Practice**: A member expressed confusion on how **Hopfield layers** are utilized as attention mechanisms in neural networks. Another member clarified that memorization occurs during pre-training, while retrieval happens during the forward pass, similar to self-attention in transformers.
- **Hopfield vs. Self-Attention Computation**: Concerns were raised about the difference in computation between **Hopfield networks** and **self-attention** mechanisms. It was explained that the Hopfield layer, when used as self-attention, updates for only one step towards the trained patterns, acting like a single step of a Hopfield network.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1255599616657723513)** (193 messages🔥🔥): 

- **"A6000 vs. 3090s GPU showdown"**: Members discussed the advantages and disadvantages of **NVIDIA A6000** GPUs with **48GB of VRAM** and NVLink connectivity for **96GB of VRAM**. Comparatively, some showed favoritism towards **quad 3090s** given their pricing and computational capabilities for **multi-GPU setups**.
  
- **"Exploring budget-friendly GPU options"**: Discussions included recommendations for cheaper GPUs such as **certified refurbished P40s** or **K80s**, arguing they can still handle large models significantly under budget. *"You can run a reasonable quant of L3 70B on two P40/P100s which are like 1/4-1/2 the cost of a single 3090"*, a member noted.
  
- **"The inflexibility of specialized AI chips"**: Skepticism surrounded the **Sohu chip** by Etched, claiming it specializes solely in transformers, projecting limitations. This was countered with *"Oh god, I didn't think it'd be that inflexible”* and further arguments about Nvidia’s anticipated **transformer cores** to rival such chips.
  
- **"Ethical considerations in AI model training"**: Members debated the exclusion of child-related data to prevent misuse in NSFW contexts, countering with arguments for maintaining model diversity. Concerns were raised about the model's usability for normal tasks, *"Models can’t generate normal scenes of a family is useless"*.

- **"The role of NSFW in AI models"**: Deliberations occurred about whether foundational AI models need NSFW data, concluding it isn’t essential since *"Models can be trained to do pretty much whatever you want after pre-training"*. However, opinions diverged significantly on the best practices for balancing ethical concerns with practical AI applications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fastvoiceagent.cerebrium.ai">The World&apos;s Fastest Voice Bot Demo</a>: A demo showcasing the potential capabilities of voice-driven AI chatbots when optimized and deployed to minimize network and model latency.</li><li><a href="https://www.etched.com/announcing-etched">Etched is Making the Biggest Bet in AI</a>: no description found</li><li><a href="https://tenor.com/view/jim-halpert-the-office-confused-gif-25227530">Jim Halpert GIF - Jim Halpert The - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/james-crashes-into-tar-hit-cargo-thomas-the-train-gif-17279386">James Crashes Into Tar Hit GIF - James Crashes Into Tar Hit Cargo - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2405.07992">MambaOut: Do We Really Need Mamba for Vision?</a>: Mamba, an architecture with RNN-like token mixer of state space model (SSM), was recently introduced to address the quadratic complexity of the attention mechanism and subsequently applied to vision t...</li><li><a href="https://x.com/bryan_johnson/status/1805629207374086490">Tweet from Bryan Johnson /dd (@bryan_johnson)</a>: Excited to invest in @Etched&#39;s $120 million series A.    10x cheaper AI models will allow us to solve aging 100x faster.  Quoting Etched (@Etched)   Meet Sohu, the fastest AI chip of all time.  Wi...</li><li><a href="https://www.reddit.com/r/singularity/comments/1dpxocg/gege_ai_a_new_music_maker_ai_that_allows_you_to/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/tenstorrent/tt-firmware">GitHub - tenstorrent/tt-firmware: Tenstorrent Firmware repository</a>: Tenstorrent Firmware repository. Contribute to tenstorrent/tt-firmware development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1255607529866526827)** (2 messages): 

- **AIW+ problem remains ambiguous**: A member discussed the complexity of solving the AIW+ problem compared to common sense AIW. They highlighted issues with calculating cousins and potential ambiguities such as subtracting Alice and her sisters and the implications of Alice's father's siblings, concluding that *"it's not proof there is no ambiguity."*
  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1255710442966225018)** (12 messages🔥): 

- **Seeking formula to predict model memory requirements**: A user asked for a formula to predict the required memory a model will use with an X-sized context window, intending to estimate context length settings based on gguf metadata.
- **Model-specific variance on memory usage**: Another user pointed out that memory usage might vary by model due to differences in attention heads, emphasizing the need for a specific formula.
- **Metadata insights for context settings**: A user shared metadata details from a llama model that could help estimate memory requirements, noting that exceeding certain context values (e.g., 8192) might degrade performance.
- **"Dumb" empirical approach suggested**: A user suggested an empirical method, advocating for a script that loads and measures RAM usage across different context lengths, with data gathered and plotted to find the rate of change.
- **Claude's convincing yet uncertain answer**: A user mentioned receiving a convincing response from Claude but remained skeptical, leaning towards empirical approaches due to the complexity and numerous variables involved.
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1255619972663742566)** (16 messages🔥): 

- **Get your dog a horoscope app**: Members discussed creating a horoscope app for dogs with an emphasis on making it "kinda goofy." One member was particularly interested in whether this app has been developed.
- **Ultra-wealthy Tinder for blood boys**: A member shared a [link](https://x.com/yaya_labs_/status/1806252865628860494?t=YEplRGt5YkcKxbA1bQMCPw&s=19) about a humorous proposal for a Tinder-like app that matches ultra-wealthy individuals with "blood boys" for transfusions. The discussion touched on the practicality and ethical concerns of such an idea.
- **Blood as a Service (BAAS)**: Members discussed the concept of "blood as a service" (BAAS), contemplating the logistics and potential benefits of direct blood transfusions to absorb vitality. They noted that while Bryan didn't benefit much from his son's blood, his father (Bryan's son's grandfather) saw more significant benefits.

**Link mentioned**: <a href="https://x.com/yaya_labs_/status/1806252865628860494?t=YEplRGt5YkcKxbA1bQMCPw&s=19">Tweet from Yaya Labs (@yaya_labs_)</a>: How  about a tinder-like app for matching the ultra wealthy to their blood boys. Would you install ?

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1255902714064273500)** (5 messages): 

- **Rakis system revolutionizes LLM inference**: The [Rakis GitHub repository](https://github.com/hrishioa/rakis) offers a **100% browser-based decentralized LLM inference** solution. A member found this approach *"so cool"* and potentially game-changing for decentralized AI applications.

- **Meta unveils LLM Compiler**: [Meta's LLM Compiler](https://x.com/AIatMeta/status/1806361623831171318) integrates advanced code optimization and compiler capabilities, boosting code size optimization and disassembly. They released 7B & 13B models under a permissive license to aid both research and commercial use, showing AI's potential in optimizing code.

- **JinaAI introduces PE-Rank for efficient reranking**: [PE-Rank](https://x.com/JinaAI_/status/1806331488247402524) by JinaAI leverages passage embeddings for efficient listwise reranking with LLMs. By encoding passages as special tokens and restricting output space, the method reduces reranking latency from 21 seconds to 3 seconds for 100 documents.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1806361623831171318">Tweet from AI at Meta (@AIatMeta)</a>: Today we’re announcing Meta LLM Compiler, a family of models built on Meta Code Llama with additional code optimization and compiler capabilities. These models can emulate the compiler, predict optima...</li><li><a href="https://x.com/JinaAI_/status/1806331488247402524">Tweet from Jina AI (@JinaAI_)</a>: Can&#39;t we just use LLM for reranking? 🤔Just throw the query, doc1, doc2,...docN into the context window and let LLM figure out the top-K? Turns out we can, and it&#39;s almost in the way you thoug...</li><li><a href="https://github.com/hrishioa/rakis">GitHub - hrishioa/rakis</a>: Contribute to hrishioa/rakis development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1255974809028792350)** (1 messages): 

- **Hermes 2 Pro 70B Launches**: *"We just released a Hermes 2 Pro 70B! This one is pure Hermes, no merge with Llama-3 Instruct."* The release promises to solve function call issues or refusals, albeit with a slight performance cost. Check it out on [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B).

- **Model Description Highlights**: Hermes 2 Pro is an upgraded version of Nous Hermes 2 with *"an updated and cleaned version of the OpenHermes 2.5 Dataset"* and a new **Function Calling and JSON Mode dataset**. It scores **90%** on function calling evaluation and **84%** on structured JSON Output eval in partnership with Fireworks.AI.

**Link mentioned**: <a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B">NousResearch/Hermes-2-Pro-Llama-3-70B · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1255624397843337340)** (96 messages🔥🔥): 

- **LLM Repetition Issues Linked to Sampling Settings**: Members discussed causes of **instruction-tuned LLMs repeating text**. One suggested, *"Lack of repetition penalty or bad sampling settings in general,"* while another concurred, calling it a *"repetition penalty issue or sampling issues."*

- **Big-AGI Frontend and Security Concerns**: A member questioned the safety of using **Big-AGI** for their ChatGPT key. Others suggested alternatives like [librechat](https://librechat.github.io/) and [huggingchat](https://huggingface.co/huggingchat), noting most options require self-hosting.

- **Available Frontends for GPT API usage**: Teknium shared an open-source web app, the [Prompt-Engineering-Toolkit](https://github.com/teknium1/Prompt-Engineering-Toolkit), as a frontend for GPT APIs. Another member introduced their own cost-effective platform, [FreeAIChatbot.org](https://freeaichatbot.org), supporting multiple functionalities and storing data locally.

- **Meta's Advanced LLM Compiler Models Released**: Meta announced the release of **LLM Compiler models** optimized for code size and disassembly tasks. The announcement included links to the [Hugging Face repo](https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3) and [research paper](https://arxiv.org/abs/2405.00675).

- **New Advanced Models in the Wild**: Users discussed new releases like **Llama-3-Instruct-8B-SPPO**, finding it impressively smart and context-aware for an 8B model. Link shared: [Meta Llama 3-8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://get.big-agi.com/">big-AGI</a>: Launch big-AGI to unlock the full potential of AI, with precise control over your data and models. Voice interface, AI personas, advanced features, and fun UX.</li><li><a href="https://freeaichatbot.org">FreeAIChatbot.org</a>: A free AI chatbot for coding, learning, and more. Use Claude, GPT-4, etc to generate images, process CSVs, chat with images, and more.</li><li><a href="https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3">UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/dancing-duck-dance-duck-duck-ooontz-dance-gif-10943740227711557279">Dancing Duck Dance Duck GIF - Dancing duck Dance duck Duck - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/peakcooper/status/1804867319350394912>">Tweet from Cooper (@peakcooper)</a>: Yes, Claude 3.5 Sonnet is crazy good</li><li><a href="https://x.com/AIatMeta/status/1806361623831171318">Tweet from AI at Meta (@AIatMeta)</a>: Today we’re announcing Meta LLM Compiler, a family of models built on Meta Code Llama with additional code optimization and compiler capabilities. These models can emulate the compiler, predict optima...</li><li><a href="https://lluminous.chat">lluminous</a>: no description found</li><li><a href="https://github.com/teknium1/Prompt-Engineering-Toolkit">GitHub - teknium1/Prompt-Engineering-Toolkit</a>: Contribute to teknium1/Prompt-Engineering-Toolkit development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1255618178470383800)** (25 messages🔥): 

- **Boolean confusion in Hermes Pro**: Members noted an issue with function calls in **Hermes Pro** where boolean values are returning `True` instead of `true` in JSON. **Teknium** suggests this might be due to **Python** format creeping into schema builds, while **.interstellarninja** confirmed that both formats were trained and accepted.
- **Discussing dataset integrity**: **Teknium** raises the question of whether datasets should be fixed to only output valid JSON. **.interstellarninja** mentioned that function calls were generated by OAI API, sparking further discussion on whether invalid bool labeling exists in the data.
- **Model-specific issues?**: **Teknium** speculates that merging issues could have affected **Theta** model, implying this might impact boolean validity. However, **craigsdenniscf** clarified the issue was observed in **Hermes Pro** not **Theta**.
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1255825144639918141)** (8 messages🔥): 

- **Glaive-RAG-v1 dataset release**: "We released a rag dataset and model this week," said a member, pointing to the [Glaive-RAG-v1](https://huggingface.co/datasets/glaiveai/RAG-v1) dataset which contains ~50k samples built using the Glaive platform for fine-tuning models for RAG use cases. The data includes documents for context, questions, an answer mode, and answers with citations.

- **Clarifying the source of citation tags**: One user asked about the origin of the `<co: tags` in the dataset system prompts, and another member clarified that while they used the same citation method as Cohere, the data was not generated using Cohere models.

- **Adapting Glaive-RAG-v1 format to Hermes RAG**: An interstellarninja pointed out that the format of input-context-output-citations in Glaive-RAG-v1 can be adapted to Hermes RAG. They also expressed openness to collaboration for generating domain-specific datasets.

- **Choosing domains for new dataset generation**: sahilch identified web search and Wikipedia as potential domains for generating new data, reflecting on domains listed in a Google sheet discussed among members.

- **Ideal dataset size discussion**: In the discussion about the ideal size of the dataset, it was mentioned that for the entirety of all RAG samples, the target size would be around 5-20k. It was noted that the current Glaive-RAG-v1 dataset already covers this range but could be expanded to add diversity or address lacking areas.

**Link mentioned**: <a href="https://huggingface.co/datasets/glaiveai/RAG-v1">glaiveai/RAG-v1 · Datasets at Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1255885748243140659)** (11 messages🔥): 

- **Spooky Gifs Galore**: A user shared a series of gif links from Tenor, including ones related to *Scary Movie Matrix Chair*, *Halloween Ghost*, *Doctor Strange*, and *The Black Hole*. These gifs capture various themes from movies, anime, and space.

- **Hello Exchange**: A brief conversation took place where user @teknium greeted the group with *"Hows it going"* accompanied by a custom emoji, followed by @rezonaut replying with *"hello"*. The interaction then concluded with @teknium responding with a waving emoji.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/halloween-ghost-ghosts-spooky-gif-15339491">Halloween Ghost GIF - Halloween Ghost Ghosts - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/the-black-hole-space-galaxy-gif-14315492">The Black Hole Space GIF - The Black Hole Space Galaxy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/plato-cave-rave-platonic-solids-gif-13990069">Plato Cave GIF - Plato Cave Rave - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/lain-iwakura-lain-serial-experiments-lain-the-wired-computer-gif-22576121">Lain Iwakura Lain GIF - Lain Iwakura Lain Serial Experiments Lain - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/scary-movie-matrix-chair-gif-12964507">Scary Movie Matrix GIF - Scary Movie Matrix Chair - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/doctor-strange-ancient-one-marvel-gif-14594882">Doctor Strange Ancient One GIF - Doctor Strange Ancient One Marvel - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/lain-lain-iwakura-serial-experiments-lain-wires-wired-gif-1481475804337586659">Lain Lain Iwakura GIF - Lain Lain iwakura Serial experiments lain - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/fmab-edward-elric-fullmetal-alchemist-fullmetal-alchemist-brotherhood-fma-gif-19554771">Fmab Edward Elric GIF - Fmab Edward Elric Fullmetal Alchemist - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1255599689118777468)** (163 messages🔥🔥): 

- **Debate over Apple hardware and AI tasks**: A member considered using a 6 or 8GB MacBook Air for AI tasks but lacked sufficient feedback from others. Another member confessed ignorance about Apple's suitability for AI work: *“Not sure how many here have knowledge with Apple computers... couldn't find one.”*
- **Quality and Parameters in LoRA Training**: A member explained different results using varied batch sizes and epochs in LoRA training. They sought advice on balancing detail and quality in training models, stating, *"16 batches and 200 epochs... the image is less detailed but has a good shape."*
- **Conflict over SD3 and civitai Licensing**: A member inquired about updates on SD3 and civitai, learning that models are still banned due to licensing issues. Another commented on licensing constraints: *“under current SD3 license”... posting models might not be allowed as civit operates commercially*.
- **Kaggle Offering Free GPU Resources**: A user highlighted that Kaggle provides two T4 GPUs (32GB VRAM) for free, beneficial for those training models. They shared links to Kaggle and a Stable Diffusion notebook on GitHub to assist others: [Kaggle](https://www.kaggle.com) and [SD WebUI Notebook](https://github.com/DEX-1101/sd-webui-notebook).
- **Interest in Restoring Old Channels**: Several members discussed the archiving of old channels containing generative AI content, with interest in possibly restoring them. One member reminisced about the value of subject-specific discussions: *“Sad, was nice to have subject specific discussion channels and build relationships.”*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/etched/status/1805625693113663834?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from Etched (@Etched)</a>: Meet Sohu, the fastest AI chip of all time.  With over 500,000 tokens per second running Llama 70B, Sohu lets you build products that are impossible on GPUs. One 8xSohu server replaces 160 H100s.  Soh...</li><li><a href="https://www.kaggle.com/">Kaggle: Your Machine Learning and Data Science Community</a>: Kaggle is the world&#x2019;s largest data science community with powerful tools and resources to help you achieve your data science goals.</li><li><a href="https://github.com/DEX-1101/sd-webui-notebook">GitHub - DEX-1101/sd-webui-notebook: Stable Diffusion Web UI Notebook</a>: Stable Diffusion Web UI Notebook . Contribute to DEX-1101/sd-webui-notebook development by creating an account on GitHub.</li><li><a href="https://opendata.blender.org/">Blender - Open Data</a>: Blender Open Data is a platform to collect, display and query the results of hardware and software performance tests - provided by the public.
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1255612709626708078)** (102 messages🔥🔥): 

- **Charting Libraries Discussion Highlights Trade-offs**: Discussions on charting libraries highlighted decisions like static vs. interactive charts, native vs. browser rendering, and data input formats. For example, *"The community needs to decide what they want out of a charting library."*

- **Access and Performance in Rendering Large Datasets**: Debate over rendering options like browser vs. native solutions centered on accessibility and efficiency. *"Sometimes you just need to plot 200 million datapoints and chrome will choke and die on that."*

- **Docker Containers for Mojo Nightly Builds**: There were questions about setting up Docker containers for different Mojo builds, with specific advice and examples shared. *"Your install line is wrong though... you can demonstrate the basics by just using `modular install nightly/mojo`."*

- **Mojo K-Means Example Issues**: A member faced errors running the K-Means example from Mojo, with discussions around outdated code and possible fixes. *"Here is one of the error message: error: use of unknown declaration 'mod'."*

- **Mojo Community Meeting Announcement**: A Mojo Community meeting was announced with details shared on date, time, and access links. *"We will have the next Mojo Community meeting... Zoom: [link]."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://microsoft.github.io/SandDance/">SandDance Home</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/examples/docker/Dockerfile.mojosdk#L54">mojo/examples/docker/Dockerfile.mojosdk at 8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/1887#issuecomment-1998929184).">[BUG]: regression with modular install --install-version · Issue #1887 · modularml/mojo</a>: Bug description --install-version was added to support version pinning of mojo, ex: in CI/CD environments. I have been using this successfully in the mojo-pytest project. With 24.1 this version pin...</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/blogs/mojo-kmeans-from-python">devrel-extras/blogs/mojo-kmeans-from-python at main · modularml/devrel-extras</a>: Contains supporting materials for developer relations blog posts, videos, and workshops - modularml/devrel-extras</li><li><a href="https://github.com/modularml/mojo/branches">Branches · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://lobste.rs/s/wnff6n/python_still_surprises">Python still surprises | Lobsters</a>: no description found</li><li><a href="https://idl.uw.edu/mosaic/examples/linear-regression-10m.html">Linear Regression 10M | Mosaic</a>: no description found</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/encodeInto">TextEncoder: encodeInto() method - Web APIs | MDN</a>: The TextEncoder.encodeInto() method takes a   string to encode and a destination Uint8Array to put resulting UTF-8 encoded text into, and returns a dictionary object indicating the   progress of the e...</li><li><a href="https://modul.ar/community-meeting-zoom.">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1255631651497709568)** (2 messages): 

- **Modular shares Twitter updates**: ModularBot posted tweets from the [Modular Twitter account](https://twitter.com/Modular/status/1806070670293692594). The updates include another tweet linked [here](https://twitter.com/Modular/status/1806356878282371398).
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1255770692662726767)** (1 messages): 

- **Human Mind: Magic and Neural Networks**: A member proposed that the human mind consists of a "magic thing" that creates, uses, and edits a neural network. This dichotomy is referred to as "intelligence" (the magic) and "cognition" (the neural network).
- **Contextual Iteration and Entropy**: The hypothesis states a given human neural network is a contextual iteration of the "magic thing," useful within close range but affected by "unnamable entropy" beyond that scope.
- **Transcription of Intelligence into Cognition**: Most behaviors of intelligence are described as resulting from a cognitive process. This transcription usually translates into a cognitive layer before manifesting in the real world, and even when taking shortcuts, cognitive equivalents can often be found.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1255631646187585576)** (23 messages🔥): 

- **Encouragement to report issues on GitHub**: Members discussed the importance of reporting any issues in Mojo to GitHub, with one member stating, *"they have been encouraging about seeing issues on github."* Users were reassured not to hesitate in raising concerns even if they use nightly builds.

- **Seeking automation for Mojo version compatibility**: A member inquired about identifying Mojo code versions automatically, suggesting a trial-and-error approach starting from a low compiler version. Another member offered help to adapt the code to the main branch if needed.

- **Curiosity about Mojo object identity**: A user asked for more information on the "No Pin requirement" section from a [Mojo vs Rust blog post](https://www.modular.com/blog/mojo-vs-rust-is-mojo-faster-than-rust), specifically regarding object identity and self-referential types.

- **Concern over network activity when running Mojo**: Multiple users observed that running Mojo in the terminal attempts to connect to the internet. One member mentioned it may be related to account authentication and suggested opening an issue on GitHub to investigate further.

**Link mentioned**: <a href="https://youtu.be/UOTAzCYQjHs">Mojo - First Impression [Programming Languages Episode 29]</a>: ►Full First Look Series Playlist: https://www.youtube.com/playlist?list=PLvv0ScY6vfd-5hJ47DNAOKKLLIHjz1Tzq►Find full courses on: https://courses.mshah.io/►Jo...

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1255601754675089478)** (31 messages🔥): 

- **Mojo Compiler Nightly Release Announced**: Nightly updates announced a new Mojo compiler release `2024.6.2705` with various changes, including moving the `tensor` module to `max` and changing `print()` requirements to `Formattable`. [Raw diff and changelog links](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) were provided.
- **Tensor Module Move Causes Issues**: Members discussed the issues caused by moving the `tensor` module to `max`, breaking dependencies like BlazeSeq. Alternatives such as `Buffer` and `NDBuffer` were suggested, and the reasoning was laid out [here](https://docs.modular.com/mojo/stdlib/buffer/buffer/).
- **Static Lifetime Discussion**: The chat included a deep dive into `ImmutableStaticLifetime`, clarifying it pertains to the lifetime of a "static" item, like what's stored in an alias. This allows taking references to `alias` items, similar to `let`.
- **Graph API and Indexing Updates**: Improvements were made to the Graph API, specifically integer literal slices and slicing across all dimensions, with some use restrictions for optimization. `ops.unsqueeze` was suggested as a workaround for certain indexing patterns not yet supported.
- **Call for More Detailed Changelogs**: Members expressed a need for more detailed changelogs, especially relating to API changes like in `max`. Developers acknowledged the oversight and committed to better documentation in future releases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/buffer/buffer/">buffer | Modular Docs</a>: Implements the Buffer class.</li><li><a href="https://github.com/modularml/mojo/issues/3098">[BUG] `Tensor` initialised from a list with wrong type shows weird behaviour · Issue #3098 · modularml/mojo</a>: Bug description To be more specific, Tensor[DType.int8] initialised with List[UIn8] doesn&#39;t compute its total number of elements correctly. I think it&#39;s again somehow related to implicit conve...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md#-removed">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modular">Modular Inc</a>: Modular is an integrated, composable suite of tools that simplifies your AI infrastructure so your team can develop, deploy, and innovate faster. - Modular Inc</li><li><a href="https://github.com/modul">modul - Overview</a>: modul has 20 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3126">[BUG] `List` doesn&#39;t work at compile time. · Issue #3126 · modularml/mojo</a>: Bug description As title. At least List.__getitem__ doesn&#39;t work. Steps to reproduce fn main(): alias l = List[Int](1, 2, 3) print(l[0]) # prints 0 System information Mojo 2024.6.2614 (366c690a) o...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1255604538715541585)** (137 messages🔥🔥): 

- **Seeking Prompt Directories for Perplexity**: A member asked for guidance on an open prompt directory compatible with the latest agent-based Perplexity Pro search. They were seeking resources for enhanced utilization.
- **API Issues and Lack of Status Page Annoy Users**: Several members complained about receiving 5xx errors and expressed frustration over the absence of a status page for the Perplexity API. One user mentioned, "WHy is there no status page about the API. I mean common guys this is very basic stuff."
- **Usage Limits and Comparison of AI Tools**: Discussions focused on the differences between Perplexity and Claude.ai, the limits of Sonnet, and user preferences between different models like GPT-4 Turbo, GPT-4o, and Claude 3.5 Sonnet. A member noted, "I think they got rid of it because there are already better models in the app."
- **Generative AI and Privacy Concerns**: A user highlighted an article discussing Perplexity's increasing citation of AI-generated sources, raising concerns about the reliability of the information. Another discussion centered on privacy friendliness, with users noting, "It seems to be better than using chatGPT since it uses the enterprise api."
- **Features and Preferences**: Members pondered the effectiveness and limitations of Perplexity's features, such as image interpretation versus Google Lens, and discussed potential future enhancements like artifact implementation for better file handling. One noted, "perplexity implementation of artifacts... would be an easy win."

**Link mentioned**: <a href="https://www.forbes.com/sites/rashishrivastava/2024/06/26/search-startup-perplexity-increasingly-cites-ai-generated-sources/">Garbage In, Garbage Out: Perplexity Spreads Misinformation From Spammy AI Blog Posts</a>: As Perplexity faces criticism for allegedly plagiarizing journalistic work and distributing it like a media company, it is increasingly citing AI-generated blogs and LinkedIn posts riddled with inaccu...

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1255613146547355728)** (8 messages🔥): 

- **Conquer Trauma with Perplexity AI**: A member shared a [Perplexity AI page](https://www.perplexity.ai/page/Overcoming-Trauma-and-9_3ox12FRFaMON3Zk8lezQ) on overcoming trauma, hinting at its potential usefulness in mental health discussions.
- **Julian Assange Release News**: Another user posted a [Perplexity AI page](https://www.perplexity.ai/page/Julian-Assange-Released-cLtbci_iSxW32Xve2NgKGA) covering the release of Julian Assange, indicating an interest in current events and legal matters.
- **Gravity's Influence Explored**: A member shared a link exploring gravity’s effects, which you can check [here](https://www.perplexity.ai/search/If-Gravity-affects-20bEiugFSnudRflOAGYlnA), pointing towards scientific curiosity.
- **Unified Theory Insight**: Interest in theoretical frameworks was shown by linking to a [search on Unified Theory](https://www.perplexity.ai/search/What-is-Unified-TbPeDD79TTKQPDPRpQXVFA), which is valuable for those in scientific research or physics.
- **Android 14 Boost Analyzed**: Learn about the performance boosts in Android 14 in this [page](https://www.perplexity.ai/page/Android-14-boosts-WxN8GGdgRQSKPPn7DftxtQ), highlighting advancements in mobile OS development.
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1255650427219873792)** (5 messages): 

- **Users report Perplexity API errors**: One user inquired about **5xx errors** and asked if there is a status page to check the API server's status. Another user reported getting **401 errors** and questioned if others faced the same issue.
- **Authentication issues discussed**: A member clarified that **401 errors** are typically related to authentication issues with the API key, but mentioned their own usage was unaffected. The original user experiencing the issue noted that their key had not changed, prompting them to reach out to support for help.
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1255607370642358332)** (63 messages🔥🔥): 

```html
<ul>
  <li><strong>Figma AI is free for a year</strong>: According to <a href="https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">@AustinTByrd</a>, <em>"Figma AI is free for a year before they start billing everyone."</em> Follow the link for full details: <a href="https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Config2024 thread</a>.</li>
  
  <li><strong>Conference talks now available via livestream</strong>: Recordings of non-livestreamed tracks like the RAG talks are still being awaited. Meanwhile, members can watch select livestreams on the <a href="https://youtube.com/@aidotengineer">AI Engineer YouTube channel</a>.</li>
  
  <li><strong>Compass transcript site shared</strong>: <a href="https://aie.compasswearable.com">Compass transcript site</a> was shared for viewing conference transcripts. These resources were mentioned to be useful and solid.</li>
  
  <li><strong>LangGraph Cloud launches</strong>: <a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJ">@LangChainAI</a> launched <a href="http://bit.ly/langgraph-cloud-beta-1">LangGraph Cloud</a>, offering scalable infrastructure for fault-tolerant agents and integrated tracing & monitoring. However, some members questioned the necessity for specialized infrastructure for state machines.</li>
  
  <li><strong>Lots of wearable tech emerging</strong>: Discussions included new wearables like <a href="https://bee.computer/">Bee.computer</a> and their features like recording, transcription, and task execution. The service even offers an Apple Watch app, making extra devices optional.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/@aidotengineer?si=KfTkCwPDCRU7jY3t">AI Engineer</a>: Talks, workshops, events, and training for AI Engineers. </li><li><a href="https://x.com/llama_index/status/1806116419995844947?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">Tweet from LlamaIndex 🦙 (@llama_index)</a>: ✨ Just announced on stage at @aiDotEngineer World&#39;s Fair! ✨ A brand new framework for getting multi-agent AI systems into production!   Currently an alpha release, llama-agents provides: ⭐️ Distri...</li><li><a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJ">Tweet from LangChain (@LangChainAI)</a>: 🚀 Introducing LangGraph Cloud 🚀  LangGraph helps you build reliable agents that actually work. Today, we&#39;ve launched LangGraph Cloud, our new infrastructure to run fault-tolerant LangGraph agent...</li><li><a href="https://www.youtube.com/watch?v=ziGNnhNABqA&t=1334s">David Luan: Why Nvidia Will Enter the Model Space &amp; Models Will Enter the Chip Space | E1169</a>: David Luan is the CEO and Co-Founder at Adept, a company building AI agents for knowledge workers. To date, David has raised over $400M for the company from ...</li><li><a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJKPq_IjA&s=19">Tweet from LangChain (@LangChainAI)</a>: 🚀 Introducing LangGraph Cloud 🚀  LangGraph helps you build reliable agents that actually work. Today, we&#39;ve launched LangGraph Cloud, our new infrastructure to run fault-tolerant LangGraph agent...</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today</a>: no description found</li><li><a href="https://x.com/DavidKPiano/status/1806417216914817514?t=99I0TJJfrKHHDQYeiizv8A&s=19">Tweet from David K 🎹 (@DavidKPiano)</a>: I love how AI startups are gradually (re)discovering state machines and the actor model for agent behavior & systems  Still unsure why you would need specialized infra for it though; it&#39;s all just...</li><li><a href="https://bee.computer/">Bee AI</a>: no description found</li><li><a href="https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Austin Byrd (@AustinTByrd)</a>: Figma AI is free for a year before they start billing everyone</li><li><a href="https://aie.compasswearable.com">AI Engineers World Fair Recaps - Powered by Compass</a>: Experience the biggest technical AI conference with live transcriptions and AI-generated summaries.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1255600076898963468)** (70 messages🔥🔥): 

```html
- **Members discuss information processing amid AGI**: One member joked *"ngmi when AGI comes if you can only process one information stream at a time"*, expressing the importance of multitasking in the future. Another member humorously referred to the ongoing discussion as *"PEAK SCHIZO"*.

- **Technical difficulties during event presentation**: Multiple members reported issues with hearing and screen sharing during a live event. One member suggested an alternative to *"leave stage and come back"*, while another proposed sharing slides directly for a smoother presentation.

- **Planning and coordination for AI Engineer World Fair**: Members discussed logistics and coordination for an event, including ensuring hosts had necessary compasses and special instructions. A YouTube link [AI Engineer](https://www.youtube.com/@aiDotEngineer) was shared highlighting talks, workshops, and events for AI Engineers.

- **Recap request for AI Engineer Conference**: There was a request for a Sunday recap or summary of the AI Engineer Conference. Responses highlighted the challenge of managing multiple conferences and events simultaneously.

- **Managing event resources and logistics**: Members coordinated the availability of resources like poster boards and having founders ready for their sessions. Special instructions were given to ensure a seamless experience for presenters and guests, with updates on team reliance on transcripts and wearable tech sightings.
```

**Link mentioned**: <a href="https://www.youtube.com/@aiDotEngineer">AI Engineer</a>: Talks, workshops, events, and training for AI Engineers. 

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1255599781514973315)** (75 messages🔥🔥): 

```html
- **LM Studio lacks document training capabilities**: Members clarified that LM Studio does not support document-based training or RAG capabilities. A member highlighted, "When the majority say 'train' they mean feeding documents to an existing model."
  
- **AnythingLLM integrates with LM Studio for document summaries**: AnythingLLM supports various document types and generates concise summaries, integrating seamlessly with LM Studio. "It's completely free and open-source with no subscription required," shared a user.

- **Claude 3.5 Sonnet praised as top code model**: Community members expressed high praise for Claude 3.5 Sonnet, available on Poe and Anthropic, calling it their "new daily driver" for coding assistance.

- **Requirements for training Llama 3**: Discussions on training Llama 3 highlighted that significant hardware investment is required, particularly for the 70B model. "You'll see the majority of them are trained on rented 8xH100 GPU clusters," explained a user.

- **Gemma 2 support in progress**: Members shared updates on upcoming support for Gemma 2 in LM Studio and llama.cpp. "I know that lmstudio devs are working on getting a release of lmstudio out ASAP with gemma 2 support," mentioned a user.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://lu.ma/yzzespyu">AI Study Group @ Block: Andrej Karpathy&#x27;s Zero to GPT Hero · Luma</a>: ______ When signing up for this event, you will be asked via email to enrol in the study group via the following…</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8156">Add support for Gemma2ForCausalLM by pculliton · Pull Request #8156 · ggerganov/llama.cpp</a>: Adds inference support for the Gemma 2 family of models. Includes support for:  Gemma 2 27B Gemma 2 9B  Updates Gemma architecture to include post-norm among other features.   I have read the contr...</li><li><a href="https://llamaimodel.com/requirements">Llama 3 Requirements [What you Need to Use It] 💻</a>: Llama 3 stands as a formidable force in the realm of AI, catering to developers and researchers alike. To fully harness the capabilities of Llama 3, it&#8217;s crucial to meet specific hardware and so...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1255604163992092775)** (23 messages🔥): 

- **DeepCoder V2 runs well on high RAM setups**: A member successfully runs **Bartkowski's Q4KM DeepCoder V2 230B** with 8K and 16K context on LM Studio, noting the use of **160GB RAM** and **64GB VRAM**. Achieving a speed of **2.68-2.8 tokens/sec**, they mention hitting memory limits when attempting 32K context.

- **Mac Studio struggles with DeepCoder V2**: Another member reports running the same model on a **Mac Studio M2 Ultra with 192GB RAM**, resulting in **frequent crashes** due to low memory. Performance clocks in at **9 tokens/sec with 8K context**.

- **Failed model loading due to memory issues**: A post with a **"Failed to load model"** error indicates insufficient GPU memory for offloading, with a suggestion to **turn off GPU offload** to resolve the issue.

- **Gemma 2's limited context window disappoints**: The release of **Gemma 2 on Kaggle** gets mixed reactions due to its **4K context limit**. Despite some suggesting it’s actually **8K**, members express dissatisfaction with the small context window in today's standards.

- **Meta's new LLM Compiler models announced**: **Meta unveils new LLM Compiler models** with code optimization and compiler capabilities. [Further details and resources](https://go.fb.me/tdd3dw) are shared for interested developers and researchers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/models/google/gemma-2">Google | Gemma 2 | Kaggle</a>: Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://x.com/AIatMeta/status/1806361623831171318">Tweet from AI at Meta (@AIatMeta)</a>: Today we’re announcing Meta LLM Compiler, a family of models built on Meta Code Llama with additional code optimization and compiler capabilities. These models can emulate the compiler, predict optima...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dpr487/gemma_2_is_live_on_kaggle_27b_9b/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1255858134472196136)** (1 messages): 

- **Balancing Expectations for Different AI Models**: One member emphasized the need to balance expectations for different AI models like **GPT and Bard**. They suggested employing different models for specific needs such as coding, storytelling, and joke writing, but cautioned about the trade-offs between **speed and quality** due to hardware limitations.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1255615156239728730)** (17 messages🔥): 

- **Custom 3D Printed Airflow Designs Gain Praise**: One member shared, "I don’t know how the interior looks like but you can design a custom 3d printed airflow," boasting a setup with 2xP40 cards and multiple fans keeping temperatures at 66 degrees under load.
- **Quiet Cooling Options for Servers**: A discussion ensued about cooling servers quietly, where one noted, "I could get away with swapping to the noctua server fans and be just fine" to reduce noise levels effectively.
- **Power Consumption Insights on Dual GPUs**: In a conversation about power draw, it was shared that "I’m pulling for 2xP40 at idle 18W each," highlighting the specific power usage at different loads.
- **AVX2 Requirement in LM Studio**: There was surprise and confirmation that "LM Studio needs AVX2 now," with a member clarifying it’s been required since around version 0.2.10.
- **Troubleshooting VRAM Issues**: One user faced a "Failed to load model" error due to VRAM limitations, and upon clarification that their GPU shares VRAM, it was advised to upgrade to a 3060 12GB for better support of large models.
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1255685222679445505)** (1 messages): 

- **Mamba-2 support in llama.cpp**: Discussion about tracking support for **Mamba-2** in llama.cpp. The best way to find updates is to search the [GitHub issues page](https://github.com/ggerganov/llama.cpp/issues/7727).

**Link mentioned**: <a href="https://github.com/ggerganov/llama.cpp/issues/7727">llama : support Mamba-2 · Issue #7727 · ggerganov/llama.cpp</a>: Mamba-2 is a new version of the Mamba architecture: Blog: https://tridao.me/blog/2024/mamba2-part1-model/ Paper: https://arxiv.org/abs/2405.21060

  

---


### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1255598967190982757)** (1 messages): 

- **Interpreter bans direct file moves**: A user questioned why they couldn't move documents or images directly into their terminal in the interpreter. The terminal seems to deny such actions, effectively giving "the ban" and not granting consent for these operations.
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1255622188237852682)** (1 messages): 

```html
- **Extracting Token Generation Data Using Python**: A member inquired on how to utilize Python to retrieve data from the local LM Studio server. They are specifically interested in the **speed of tokens** and the **time taken for the generation of the first token**.
```
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1255694219734159420)** (28 messages🔥): 

- **Directly printing stream object in Python fails**: One member pointed out that you "cannot the print the stream object directly," and suggested iterating over it instead. They provided a Python snippet to demonstrate proper usage: *"for token in llm.stream(input_text): print(token.content,end='')"*.
- **Query relevant vectors for user queries**: A member described an issue where the retrieval system fetches non-relevant vectors when a user selects an option from a list, despite initial relevant vectors being retrieved. They discussed potential solutions, such as *"keeping previous retrieval in chat history"* or using a function like `query_knowledge_base("Green light printer problem")` as suggested by another member.
- **Using `astream_events` with Streamlit**: When asked about integrating `astream_events` with Streamlit, the response highlighted the lack of specific examples in the provided knowledge base. Recommendations were given to refer to the LangChain [documentation](https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events) for more details.
- **Converting LLM responses to audio for mobile apps**: One member shared their approach of converting LLM text responses to audio files using Google Text-to-Speech and sending them back to devices. They inquired if Gemini supports direct streaming of text responses in audio format.
- **Example code for using MemGPT with LangGraph**: A request was made for example code showing how MemGPT can be used for infinite context memory using agents like LangGraph. They were interested in implementations involving both open-source LLMs and OpenAI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://192.168.1.70:11434")`">no title found</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/17703>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events>)">How to stream runnables | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/#streaming-tokens>).">Build an Agent | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1255814096322957312)** (70 messages🔥🔥): 

- **Building LangChain endpoints with parallel processing**: Members discussed how to build endpoints using `add_routes` and enabling parallel processing with `RunnableParallel`. Extensive documentation and examples were shared, including code snippets for FastAPI and language models.

- **Providing documents to LangChain server-side**: The community explored how to use `load_qa_chain()` for providing documents server-side with examples given for both `run()` and dictionary-based methods. This included handling parallel processing using the `map_reduce` chain type.

- **Handling high volume requests in LangChain**: Questions were raised about managing 100 requests for an RAG endpoint, and guidance was given on using modern web servers like Flask or FastAPI for concurrent request handling. Configuration to increase worker processes or threads was recommended for better performance.

- **LangChain Expression Language (LCEL) capabilities**: Detailed insights into LCEL's features like first-class streaming support, async support, optimized parallel execution, and retries were discussed. Documentation links were provided for more comprehensive understanding.

- **Concurrent processing in LangChain chains**: The community sought clarification on whether the `invoke()` method in chains like `RetrievalQA` supported concurrent processing. It was clarified that while `invoke()` handles single requests, overall concurrency depends on server setup to handle multiple concurrent requests individually.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://js.langchain.com/v0.2/docs/how_to/sequence/#coercion>).">How to chain runnables | 🦜️🔗 Langchain</a>: One point about [LangChain Expression</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/functions/#next-steps>).">How to run custom functions | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/langserve/#endpoints>)">🦜️🏓 LangServe | 🦜️🔗 LangChain</a>: Release Notes</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/lcel_cheatsheet/#runnableparallel>)">LangChain Expression Language Cheatsheet | 🦜️🔗 Langchain</a>: This is a quick reference for all the most important LCEL primitives.</li><li><a href="https://github.com/langchain-ai/langchain/issues/11433>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/12423>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/7876>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13696>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/1145>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/8399>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/16980>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/20492>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/retrievers/flashrank-reranker/#qa-reranking-with-flashrank>)">FlashRank reranker | 🦜️🔗 LangChain</a>: FlashRank is the Ultra-lite &amp; Super-fast Python library to add re-ranking to your existing search &amp; retrieval pipelines. It is based on SoTA cross-encoders, with gratitude to all the model own...</li><li><a href="https://github.com/langchain-ai/langchain/issues/9865>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/4950>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/#langchain-expression-language-lcel>)">How-to guides | 🦜️🔗 LangChain</a>: Here you’ll find answers to “How do I….?” types of questions.</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel>)">Conceptual guide | 🦜️🔗 LangChain</a>: This section contains introductions to key parts of LangChain.</li><li><a href="https://js.langchain.com/v0.2/docs/concepts/#langchain-expression-language>)">Conceptual guide | 🦜️🔗 Langchain</a>: This section contains introductions to key parts of LangChain.</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/rag/#built-in-chains>)">Build a Retrieval Augmented Generation (RAG) App | 🦜️🔗 LangChain</a>: One of the most powerful applications enabled by LLMs is sophisticated question-answering (Q&amp;A) chatbots. These are applications that can answer questions about specific source information. These ...</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/providers/dspy/#normal-lcel>)">DSPy | 🦜️🔗 LangChain</a>: DSPy is a fantastic framework for LLMs that introduces an automatic compiler that teaches LMs how to conduct the declarative steps in your program. Specifically, the DSPy compiler will internally trac...</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/rag/#retrieval-and-generation-generate>)">Build a Retrieval Augmented Generation (RAG) App | 🦜️🔗 Langchain</a>: One of the most powerful applications enabled by LLMs is sophisticated
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1255818992447258685)** (4 messages): 

- **Merlinn debuts as a Slack on-call developer bot**: Introducing [Merlinn](https://github.com/merlinn-co/merlinn), an open-source AI tool that assists with **troubleshooting production incidents** by connecting to various tools and providing root cause analysis. The developers highlight its ability to improve on-call efficiency by leveraging LangChain for its workflows, and they invite feedback and stars on GitHub.
  
- **Evidently AI shares ML system design case studies**: A comprehensive [Airtable of 450 case studies](https://www.evidentlyai.com/ml-system-design) details real-world applications and design takeaways for ML and LLM systems from over 100 companies. The database includes filters by industry and ML use case, with tags to help users quickly find relevant studies.

- **ZenGuard AI adds security features to LangChain**: The new integration with [ZenGuard AI](https://python.langchain.com/v0.2/docs/integrations/tools/zenguard) includes features like prompt injection protection, jailbreak prevention, and data leak prevention. ZenGuard AI aims to safeguard applications from malicious activities and unauthorized access, and invites feedback on GitHub.

- **YouTube tutorial on No Code Chrome Extension Chat Bot**: A [YouTube video](https://www.youtube.com/watch?v=-OKC7CY2bbQ) demonstrates how to create a no-code Chrome extension chat bot using Visual LangChain. The demo showcases designing a LangChain RAG application with interactive chat capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.evidentlyai.com/ml-system-design">Evidently AI - ML system design: 450 case studies</a>: How do top companies apply ML? We made a database of 450 case studies from 100+ companies with practical ML use cases and learnings from designing ML systems.</li><li><a href="https://www.youtube.com/watch?v=-OKC7CY2bbQ">No Code Chrome Extension Chat Bot Using Visual LangChain</a>: In this demo, I show an exciting new feature of Visual Agents where you can design your LangChain RAG application, including an interactive chat feature to a...</li><li><a href="https://github.com/merlinn-co/merlinn">GitHub - merlinn-co/merlinn: Open source AI on-call developer 🧙‍♂️ Get relevant context &amp; root cause analysis in seconds about production incidents and make on-call engineers 10x better 🏎️</a>: Open source AI on-call developer 🧙‍♂️ Get relevant context &amp; root cause analysis in seconds about production incidents and make on-call engineers 10x better 🏎️ - merlinn-co/merlinn</li><li><a href="https://zenguard.ai)">no title found</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/new?assignees=&labels=03+-+Documentation&projects=&template=documentation.yml&title=DOC%3A+%3CIssue+related+to+/v0.2/docs/integrations/tools/zenguard/%3E&url=https://python.langchain.com/v0.2/docs/integrations/tools/zenguard/">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

emarco: https://www.youtube.com/watch?v=Q_yKRLACx78&t=1s
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1255675507643519060)** (2 messages): 

- **LlamaIndex unveils new multi-agent AI framework**: Announced at the AI Engineer World's Fair, LlamaIndex introduces **llama-agents**, a new framework for deploying multi-agent AI systems in production. The alpha release offers a distributed, service-oriented architecture with communication via standard HTTP APIs. [Twitter Announcement](https://twitter.com/llama_index/status/1806116419995844947).

- **LlamaCloud launches waitlist**: LlamaIndex has opened a waitlist for **LlamaCloud**, its fully-managed ingestion service. Interested users can [sign up](https://cloud.llamaindex.ai/) and are invited to share their email addresses to gain access at a measured pace. [Additional details](https://twitter.com/llama_index/status/1806117132956299497).

**Link mentioned**: <a href="https://t.co/kAY9YEmOkx">LlamaCloud Waitlist</a>: Thanks for your interest in LlamaCloud! Sign up and tell us below which email address you used, we&#39;ll be letting people in at a measured pace.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1255599153879453798)** (56 messages🔥🔥): 

- **Users question lack of JSONReader in Readers map**: Multiple users discussed why the JSONReader is not included in the default Readers map for file extensions in LlamaIndex. A member suggested a PR to add the mapping, and another noted it can be overridden by passing a custom `file_extractor`.

- **LlamaParse struggles with hallucination**: A user reported LlamaParse performing better than GPT-4 on financial documents but still hallucinating basic information. The user was asked to send their file for debugging purposes.

- **New AI stack announcement**: Users discussed the announcement of LlamaIndex's new AI stack and shared the [link to the blog post](https://www.llamaindex.ai/blog/introducing-llama-agents-a-powerful-framework-for-building-production-multi-agent-ai-systems).

- **BM25 requires frequent re-indexing**: A discussion arose around the need for re-indexing BM25 when new documents are added, with suggestions pointing to the inefficiency of this process. Alternative sparse embedding methods like Splade were suggested.

- **Performance issues in ingestion pipelines**: A user flagged that large document management in LlamaIndex's ingestion pipeline causes significant performance hits. They suggested batching node deletions before updating reference documentation info, and the idea was welcomed as a potential enhancement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.llamaindex.ai/blog/introducing-llama-agents-a-powerful-framework-for-building-production-multi-agent-ai-systems">Introducing llama-agents: A Powerful Framework for Building Production Multi-Agent AI Systems — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb">RetrievalTutorials/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb at main · FullStackRetrieval-com/RetrievalTutorials</a>: Contribute to FullStackRetrieval-com/RetrievalTutorials development by creating an account on GitHub.</li><li><a href="https://github.com/run-llama/llama_index/blob/a24292c79424affeeb47920b327c20eca5ba85ff/llama-index-core/llama_index/core/storage/docstore/keyval_docstore.py#L485),">llama_index/llama-index-core/llama_index/core/storage/docstore/keyval_docstore.py at a24292c79424affeeb47920b327c20eca5ba85ff · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/pull/14419">added JSONReader to file mappings by denen99 · Pull Request #14419 · run-llama/llama_index</a>: Description The default_file_reader_cls dict that maps file extensions to the proper XReader, did not include a mapping for the new JSONreader.  This PR adds the mapping for .json files to the JSON...</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode.REFINE>).">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode>).">Index - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/json.py#L51">llama_index/llama-index-core/llama_index/core/readers/json.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/file/base.py#L69">llama_index/llama-index-core/llama_index/core/readers/file/base.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1255656405982253197)** (13 messages🔥): 

```html
- **OpenAI API outsells Microsoft on Azure**: OpenAI now generates more revenue from API sales than Microsoft does from reselling it on Azure. This news was shared via a tweet by Aaron P. Holmes, highlighting a surprising turn in the market dynamics. [Source](https://x.com/aaronpholmes/status/1806312654505443347?s=46)
- **Meta releases LLM Compiler for code optimization**: Meta has introduced the **Meta Large Language Model Compiler**, geared towards compiler optimization tasks using pre-trained models. The suite focuses on LLVM-IR and assembly code, leveraging a vast corpus of 546 billion tokens. [Research Publication](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/)
- **Character.AI launches Character Calls**: Character.AI has launched **Character Calls**, allowing users to have voice conversations with AI characters. The feature is accessible via their app and aims to create more immersive AI experiences but received mixed reviews on its performance and fluidity. [Blog Post](https://blog.character.ai/introducing-character-calls/)
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.character.ai/introducing-character-calls/">Introducing Character Calls</a>: 0:00  /0:07   1×                  Calling the Character.AI Community!  We&#x27;re thrilled to ring in an exciting new feature that&#x27;s set to redefine your Character.AI experience: Character Calls!...</li><li><a href="https://x.com/giffmana/status/1806411302190915603?s=46">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: @fouriergalois @character_ai just tried it, it&#39;s not comparable unfortunately, would have been super impressive!  It&#39;s not fluid at all. 5sec delay when I&#39;m done talking. I can&#39;t inter...</li><li><a href="https://huggingface.co/collections/facebook/llm-compiler-667c5b05557fe99a9edd25cb">LLM Compiler - a facebook Collection</a>: no description found</li><li><a href="https://x.com/aaronpholmes/status/1806312654505443347?s=46">Tweet from aaron holmes (@aaronpholmes)</a>: New: OpenAI is now making more from sales of its API than Microsoft makes from reselling it on Azure https://www.theinformation.com/articles/in-a-surprise-openai-is-selling-more-of-its-ai-models-than-...</li><li><a href="https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1255630429231517767)** (15 messages🔥): 

- **Interview Challenge Levels Eyed as Excessive**: One member shared an experience where a friend's online assessment for an interview involved *"the hardest tier codeforces question"*, which seemed *"a bit excessive"*. The same member mentioned potential contract violations due to "paid labor" requirements during the interview process.
  
- **Coding Interview Arbitrary Bars**: Another member expressed frustration after performing well on a coding interview but failing to clear an *"arbitrary bar"* set by the interviewers. This sentiment resonated with another member who was glad they weren’t the only one facing this issue.

- **Advanced Voice Capabilities in ChatGPT Subreddit**: A tweet ([AndrewCurran](https://x.com/AndrewCurran_/status/1806178276001329373)) reported someone on the ChatGPT subreddit claimed access to an advanced voice feature that could generate sound effects along with speech. This instance was shared as *"Interesting audio in thread"*.

- **Chain of Thought Prompting Patent**: Members discussed the discovery of a patented *"chain of thought prompting strategy"* after a tweet ([andrewwhite01](https://x.com/andrewwhite01/status/1806347002126446736?s=46)). They questioned whether the patent was granted or merely applied for, and debated the practical enforceability of such patents.

- **Google's Transformer Architecture Patent Non-Enforcement**: It was noted that Google holds a patent on transformer architecture but *"doesn’t enforce it"*—sparking curiosity on why enforcement wasn’t pursued, with guesses leaning towards potential failure in court.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/james-bond-007-tomorrow-never-dies-elliot-carver-stamper-gif-24934311">James Bond 007 GIF - James Bond 007 Tomorrow Never Dies - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/andrewwhite01/status/1806347002126446736?s=46">Tweet from Andrew White 🐦‍⬛/acc (@andrewwhite01)</a>: TIL chain of thought prompting is patented. Didn&#39;t realize prompting strategies were patentable.</li><li><a href="https://x.com/AndrewCurran_/status/1806178276001329373">Tweet from Andrew Curran (@AndrewCurran_)</a>: Someone on the reddit ChatGPT sub claims they had access to advanced voice this morning, and before they lost it they recorded 4o telling a story. The interesting part was 4o generated the accompanyin...</li><li><a href="https://patents.google.com/patent/US20230394328A1/en?oq=US+2023/0394328+A1">US20230394328A1 - Prompting Machine-Learned Models Using Chains of Thought 
      - Google Patents</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1255684248225189961)** (23 messages🔥): 

- **Shoe guessing game stumps members**: A member shared a fun challenge, asking others to *"Guess the popular ML figure by the shoes"*. Discussion identified incorrect guesses with claims like *"Yea that's not Sam. Sam wears nikes"* and highlighted grooming updates with a link to a tweet speculating on hairstyle changes: [Twitter](https://x.com/1vnzh/status/1802093900993073611).

- **Cohere founders’ cool factor**: Members discussed the founders of Cohere, sharing a video interview and a tweet that praised their practical focus on AI solutions and highlighted their rockstar-like status. A link to the full video was shared: [YouTube](https://www.youtube.com/watch?v=4JF1V2hzGKE).

- **Fun banter over interviews**: One member humorously dissected a YouTube interview with OpenAI's Sam Altman and Airbnb's Brian Chesky, noting how the questioning appeared lopsided with more serious questions directed at Sam compared to lighter, more casual ones for Brian. Link shared: [YouTube](https://www.youtube.com/watch?v=8e8RpbO2lNU).

- **Variety of humorous links**: Various playful and off-topic links were shared, including a meme-worthy context video referencing Cohere's founder and a Bourne Supremacy movie clip: [YouTube](https://youtu.be/I3znSbbu9IU?si=EbbsoUgHAFS1wuMY&t=65). Another tweet humorously depicted Cohere's CTO as a skilled gamer under pressure: [Twitter](https://x.com/internetvin/status/1800019341343084662).

- **Emoji brings context union**: One member theorized about the implied meaning behind emoji use in a thread, suggesting it conveyed unspoken agreements or context: *"you got things right we aren't tryin to say"*.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ok5XRYhgH9Q">imagine if ninja got a low taper fade</a>: no description found</li><li><a href="https://x.com/internetvin/status/1800019341343084662">Tweet from internetVin (@internetvin)</a>: pov of playing against @1vnzh, cto of cohere, when he is using an awp, and his back against the wall, with visions of betakit headline about cohere mans bottom fragging against new demos and walk n ta...</li><li><a href="https://x.com/youraimarketer/status/1805629336973688853">Tweet from Muratcan Koylan (@youraimarketer)</a>: I am bullish on Cohere because their founders not just talk like rockstars but they are indeed rockstars.  Quoting cohere (@cohere)   In this clip, @MLStreetTalk chats with Cohere Co-Founder, Nick Fro...</li><li><a href="https://www.youtube.com/watch?v=8e8RpbO2lNU">Lester Holt interviews Open AI&#39;s Sam Altman and Airbnb&#39;s Brian Chesky</a>: The CEO of OpenAI, Sam Altman and co-founder and CEO of Airbnb, Brian Chesky join NBC News’ Lester Holt to talk about the benefits of artificial intelligence...</li><li><a href="https://youtu.be/I3znSbbu9IU?si=EbbsoUgHAFS1wuMY&t=65">The Bourne Supremacy (9/9) Movie CLIP - Final Call to Pamela (2004) HD</a>: The Bourne Supremacy movie clips: http://j.mp/1uvIXs9BUY THE MOVIE: http://amzn.to/tor8HhDon&#39;t miss the HOTTEST NEW TRAILERS: http://bit.ly/1u2y6prCLIP DESCR...</li><li><a href="https://x.com/aidangomez/status/1797900448776822948">Tweet from Aidan Gomez (@aidangomez)</a>: Spent a day with Cohere’s xrisk safety team keeping an eye on Command R++ training.</li><li><a href="https://x.com/1vnzh/status/1802093900993073611">Tweet from Ivan Zhang (@1vnzh)</a>: what if aidan gomez got a low taper fade
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1255766574044938240)** (3 messages): 

- **Clarification on "bases" terminology**: A member asked about the term "bases" in the context of a recent synthetic data article. Another member clarified that it referred to **base models**.
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1255855049050161172)** (1 messages): 

- **Stheno 8B debuts on OpenRouter**: The **[L3 Stheno 8B 32K](https://openrouter.ai/models/sao10k/l3-stheno-8b)** is now available on OpenRouter. This model is launched by **OpenRouter, LLC** for the year 2023-2024.
- **Flavor of the Week**: **Stheno 8B** is highlighted as **[this week's flavor](https://openrouter.ai/models/openrouter/flavor-of-the-week)** on OpenRouter. The model continues to catch users' interest under the promotion by OpenRouter, LLC for the year 2023-2024.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/sao10k/l3-stheno-8b)">Llama 3 Stheno 8B v3.3 32K by sao10k</a>: Stheno 8B 32K is a creative writing/roleplay model from [Sao10k](https://ko-fi.com/sao10k). It was trained at 8K context, then expanded to 32K context.  Compared to older Stheno version, this model is...</li><li><a href="https://openrouter.ai/models/openrouter/flavor-of-the-week>)!">Flavor of The Week by sao10k</a>: This is a router model that rotates its underlying model weekly. It aims to be a simple way to explore the capabilities of new models while using the same model ID.  The current underlying model is [L...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1255767355569475686)** (39 messages🔥): 

- **NVIDIA Nemotron page issues perplex user**: A user reported getting a 'page not working' error when trying to select **NVIDIA Nemotron** on their phone, even though it seemed to work for another user. They noted it was not a big issue if it was just an isolated problem with their phone.

- **OpenRouter API key compatibility inquiry**: One user asked if applications that expect an OpenAI key could work with an **OpenRouter API key**. They were advised to try overriding the base API URL, though solutions may vary by application.

- **Model recommendations amidst censorship concerns**: A user requested recommendations for uncensored models. **Cmd-r** and **Euryale 2.1** (a fine-tuned Llama 3) were suggested, with **Magnum** mentioned as pending inclusion on OpenRouter, while **jailbroken Claude 3** was also highlighted.

- **Google Gemini API updates excite developers**: A shared [Google blog post](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/) announced access to a **2 million token context window** for **Gemini 1.5 Pro**, alongside code execution capabilities. This update aims to help developers manage input costs through context caching.

- **Artifacts feature in Claude 3 Web causes confusion**: A user desired a similar feature to Anthropic's Artifacts on OpenRouter. They were advised that **Sonnet-3.5** might offer a partial workaround by generating code via the usual prompt methods.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/docs/code-execution?lang=node">no title found</a>: no description found</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today</a>: no description found
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1255728961040678942)** (23 messages🔥): 

- **OpenRouter utilizes Cohere API**: Members discussed that OpenRouter uses the **Cohere API** to avoid breaching the NC (Non-Commercial) part of the license. One member confirmed, *"if they use the cohere api - NC does not apply."*

- **Command-R model exclusivity on OpenRouter**: A link to OpenRouter's page highlighted that the **Command-R** model is available via [OpenRouter](https://openrouter.ai/models/cohere/command-r/status) and in a Patreon post noted that it's for *"I'm All In Subscribers".* This model is recognized for its creativity and prompt-following capabilities.

- **Conflict on Command-R licensing concerns**: There was a concern that SpicyChat might be misusing the license, with an in-depth conversation about whether **Command-R's** usage aligns with NC licenses. Nonetheless, members clarified that payments to Cohere should cover any licensing issues.

- **Tool use error on Colab script resolved**: A member resolved an issue with running a Cohere API script on Colab and locally on PyCharm. By referencing the [Cohere documentation on multi-step tool use](https://docs.cohere.com/docs/multi-step-tool-use#step-2-ask-model-for-tool-calls-and-send-back-tool-results), the user corrected the error and got their script working perfectly.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/cohere/command-r/activity">Cohere: Command R – Recent Activity</a>: See recent activity and usage statistics for Cohere: Command R - Command-R is a 35B parameter model that performs conversational language tasks at a higher quality, more reliably, and with a longer co...</li><li><a href="https://openrouter.ai/models/cohere/command-r/status">Cohere: Command R – Provider Status and Load Balancing</a>: See provider status and make a load-balanced request to Cohere: Command R - Command-R is a 35B parameter model that performs conversational language tasks at a higher quality, more reliably, and with ...</li><li><a href="https://docs.cohere.com/docs/multi-step-tool-use#step-2-ask-model-for-tool-calls-and-send-back-tool-results">Multi-step Tool Use (Agents)</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1255618654142202156)** (2 messages): 

- **Rig incentivizes developer feedback**: A member announced the release of **Rig**, a Rust library for building LLM-powered applications, along with an incentivized feedback program where developers are rewarded for building use cases and providing feedback on the library. They inquired whether it was appropriate to post the details in the channel and were advised to ensure the library supports Cohere’s models.
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1255657271048798269)** (15 messages🔥): 

- **Join Block's Mission Office for GPT Deep Dive**: An event at Block's Mission office in SF offers a free, 4-week study group based on Andrej Karpathy’s [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) YouTube series. Sign up [here](https://lu.ma/yzzespyu) and enroll in the study group via this [Google Form](https://forms.gle/L4u3TMfTs5TjqWpt7).

- **Exploring Open Source Model for Interpreter**: A user inquired about the best open-source model for the interpreter, mentioning **GPT-4o** and seeking recommendations for local deployment with **Ollama** or **Groq**. The question reflects a common interest in optimizing interpreters using open-source models.

- **GitHub Acceptable Use Policy Concerns**: A user expressed concerns about a project potentially violating GitHub's Acceptable Use Policies and DMCA Takedown Policy. They emphasized the need to discuss these issues openly before submitting a formal notice.

- **Meta Unveils LLM Compiler**: Meta announced the **LLM Compiler**, featuring models built on **Meta Code Llama** for optimizing and disassembling code, which can be fine-tuned for various tasks. The announcement included links to the [HuggingFace repo](https://go.fb.me/tdd3dw) and the [research paper](https://go.fb.me/85zwgy), with models available under a permissive license for both research and commercial use.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aiatmeta/status/1806361623831171318?s=46">Tweet from AI at Meta (@AIatMeta)</a>: Today we’re announcing Meta LLM Compiler, a family of models built on Meta Code Llama with additional code optimization and compiler capabilities. These models can emulate the compiler, predict optima...</li><li><a href="https://x.com/mindbranches/status/1806370172506091843?s=46">Tweet from MindBranches (@MindBranches)</a>: @AIatMeta Summary of the full research paper: &#34;Meta Large Language Model Compiler: Foundation Models of Compiler Optimization&#34;</li><li><a href="https://lu.ma/yzzespyu">AI Study Group @ Block: Andrej Karpathy&#x27;s Zero to GPT Hero · Luma</a>: ______ When signing up for this event, you will be asked via email to enrol in the study group via the following…
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1255609762825900043)** (5 messages): 

- **No more `--local` option for 01**: A member pointed out that with the latest release, the `--local` option is no longer available for 01. This sparked curiosity about what models are available now.
- **Interpreter purchase logistics**: A user asked if they could buy the 01 interpreter using a friend's address and then send it to Spain. They also inquired about its functionality in Spanish.
- **Model availability and usage**: Questions were raised regarding what models are available with the latest release, specifically asking if only GPT-4 is supported. They also questioned whether usage is tied to OpenAI API keys or if account login with a €20 subscription would suffice.
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1255598742732800100)** (13 messages🔥): 

- **NCCL Watchdog CUDA Errors Spotted**: A member reported that they encountered **NCCL watchdog thread termination** with a **CUDA error** related to illegal memory access. They suggested passing `CUDA_LAUNCH_BLOCKING=1` for debugging and enabling device-side assertions by compiling with `TORCH_USE_CUDA_DSA`.

- **Gemma 2's New Release Details**: [Gemma 2 released!](https://x.com/_philschmid/status/1806343336292229234?s=46) **Google's Gemma 2** model, available in 9B & 27B sizes, has advanced capabilities with sliding window attention and logit soft-capping. The model boasts scores comparable to **Meta's Llama 3 70B**, indicating strong performance metrics.

- **Meta LLM Compiler Announcement**: [Meta announced LLM Compiler models](https://x.com/aiatmeta/status/1806361623831171318?s=46) built on **Meta Code Llama**, focusing on code optimization and compiler tasks. These models, achieving state-of-the-art results, are available under a permissive license for research and commercial use.

- **Gemma 2 Transformer Code Issues**: Discussions highlighted **issues with Transformers code** impacting Gemma 2's sample packing. A [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718) has been made to address these issues, with a pending upstream fix from Hugging Face.

- **Community Humor on HF Bugs**: The community laughed about typical **Hugging Face bugs**, with specific mentions of **Gemma2DecoderLayer.forward** corrupting the attention mask for sliding window operations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_philschmid/status/1806343336292229234?s=46">Tweet from Philipp Schmid (@_philschmid)</a>: Gemma 2 released! Google just released the next iteration of its open LLM! Gemma 2 comes in two sizes, 9B & 27B, trained on 13T tokens. Gemma 2 27B approaches @AIatMeta  Llama 3 70B performance!   Fir...</li><li><a href="https://x.com/aiatmeta/status/1806361623831171318?s=46">Tweet from AI at Meta (@AIatMeta)</a>: Today we’re announcing Meta LLM Compiler, a family of models built on Meta Code Llama with additional code optimization and compiler capabilities. These models can emulate the compiler, predict optima...</li><li><a href="https://x.com/mindbranches/status/1806370172506091843?s=46">Tweet from MindBranches (@MindBranches)</a>: @AIatMeta Summary of the full research paper: &#34;Meta Large Language Model Compiler: Foundation Models of Compiler Optimization&#34;</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718">support for gemma2 w sample packing by winglian · Pull Request #1718 · OpenAccess-AI-Collective/axolotl</a>: Description  Motivation and Context   How has this been tested?    Screenshots (if appropriate) Types of changes  Social Handles (Optional)
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1255625370359828540)** (1 messages): 

- **Mistral7B loves to repeat itself**: A member shared an issue with **Mistral7B** during full instruction-tuning, where the model sometimes repeats sentences or paragraphs even with high temperature settings. They clarified that the dataset does not contain such instances, and sought advice on potential causes.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1255930674641113149)** (2 messages): 

- **Gemma2:9B model impresses in first test**: A video titled ["Gemma2 First Test ! Incredible Results"](https://youtu.be/6SLuneidHYw) demonstrates the installation and testing of the **Gemma2:9B model** with **ollama**. The description highlights the incredible results from this initial test.
- **Gemma2:27B leaves much to be desired**: Another video titled ["Gemma2:27B First Test ! How Can it be THAT Bad ?!"](https://youtu.be/vIKNRiVxWeo) showcases the testing of the **Gemma2:27B model**, which was released an hour ago by **Google**. The video points out significant disappointments with this larger model's performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/6SLuneidHYw">Gemma2 First Test ! Incredible Results</a>: Today, we are going to install and test Gemma2 with ollama</li><li><a href="https://youtu.be/vIKNRiVxWeo">Gemma2:27B First Test ! How Can it be THAT Bad ?!</a>: Let&#39;s test the biggest version (27B) of the gemma2 release an hour ago by Google with ollama
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1255607970172112896)** (8 messages🔥): 

- **Watch PyTorch Documentary**: A YouTube video titled [Official PyTorch Documentary: Powering the AI Revolution](https://www.youtube.com/watch?v=rgP_LBtaUEc) was shared. The documentary offers an *authentic narrative of PyTorch’s inception* and its development by dedicated engineers.

- **FPGA Design Choices Clarified**: One member clarified that they are not using **Xilinx/AMD FPGAs** and emphasized that the design instantiated on the FPGA is **generic for all Transformer models**. They stated that the FPGA can load any model from the Huggingface Transformer library without requiring specialized RTL.

- **Tinygrad Project Contribution**: A member mentioned generating with **SDXL in tinygrad** and noted they still need to do some cleanup and performance enhancements. They plan to open a PR to upstream into examples once it is ready.

- **Today's Presentation**: George Hotz announced an **eight-minute presentation** scheduled for today, though the details were not specified.

- **Tinygrad Matching Engine Bounty Issue**: A $500 bounty issue for improving the [matching engine's speed](https://github.com/tinygrad/tinygrad/issues/4878) was highlighted. Members were encouraged to read PRs to determine if others are already working on it.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=rgP_LBtaUEc">Official PyTorch Documentary: Powering the AI Revolution</a>: This film unveils the authentic narrative of PyTorch’s inception, attributing its existence to a dedicated group of unsung heroes driving technological innov...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/4878">Matching engine is slow ($500 bounty) · Issue #4878 · tinygrad/tinygrad</a>: Rewrite the matching engine to get a 2x speedup on my machine (intel core ultra 7 155h) in &quot;model lower&quot; jesse@x1:~/tinygrad$ PROFILE=0 python3 test/external/external_benchmark_schedule.py *...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1255701453864042549)** (5 messages): 

- **MultiheadAttention port question on tinygrad**: A member asked for examples of **porting pytorch's MultiheadAttention to tinygrad** with `in_proj_bias` and `in_proj_weight` as single parameters. They later discovered that `pytorch.nn.functional.linear` **expects a pre-transposed matrix** as the weights.

- **Estimating VRAM for model training**: Another member inquired about creating a **NOOP backend** to estimate total memory allocations easily. They referenced a tweet discussing methods to **estimate VRAM needed for training** models with a given parameter size [source](https://x.com/skalskip92/status/1806293661014958330).

- **Understanding shapetracker in tinygrad**: By studying [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/), one member explained how **Shapetracker enables zero-cost movement operations** by representing data in various multi-dimensional structures without changing the underlying memory. They sought clarification on the role of **masks within shapetracker** usage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html">How ShapeTracker works</a>: Tutorials on tinygrad</li><li><a href="https://x.com/skalskip92/status/1806293661014958330">Tweet from SkalskiP (@skalskip92)</a>: does anyone have a good way of estimating how much VRAM I&#39;ll need to train the model with X amount of parameters?
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1255600251440730267)** (2 messages): 

- **Anthropic's Claude Contest Gets the Spotlight**: A link to the [Anthropic documentation](https://docs.anthropic.com/en/build-with-claude-contest/overview) was shared. This document outlines the details for a contest focusing on building with Claude.
- **Seeking Guidance on Fine-tuning LLMs for Cover Letter Generation**: A user is working on a project to fine-tune an LLM to draft cover letters using resume text and job descriptions. They inquired about how to use their test data effectively to evaluate the model's performance in this text-generation task.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/)** (1 messages): 

__dchx: was there an answer to your question <@610008277714993152> ?
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1255878416109142070)** (1 messages): 

- **User awaiting credit allocation**: A member sent a direct message to another regarding their account details and the status of their credit allocation. They have not yet received their credits.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1255750647853350954)** (1 messages): 

- **Evaluating LLM for Cover Letter Generation**: A member inquired about fine-tuning a **LLM model to write cover letters** using resume text and job descriptions. They asked how the test data can be utilized to evaluate model performance in this **text generation task**.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1255795071438295151)** (1 messages): 

- **Building a Tweet-based Bot with Flask**: Someone is developing a project to create an endpoint where a model can be trained with their tweets. They've integrated with the Twitter API using Flask and Tweepy and are seeking advice on how to incorporate a model, train it with the tweets, and enable it to respond to questions *"in my style"*.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1255876604278607882)** (1 messages): 

- **Assistance requested by Ajay**: A user reached out to another member, **@466291653154439169**, for assistance. The context or specific request details were not provided.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/)** (1 messages): 

lalithnarayan: Pinged you on DM.. please could you take a look 🙏
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1255710861746638932)** (3 messages): 

- **Student ponders over Copilot vs Cursor**: A student is debating whether to switch from **Copilot** to **Cursor** using the provided OpenAI credits. They are also curious if the paid version of **Cursor** offers better functionality than either option.
- **Mix and match of Copilot in Cursor**: A suggestion was made to test **Cursor** using the free version with given OpenAI credits and potentially install **Copilot** within **Cursor**. [Guide to install VSCode extensions in Cursor](https://www.cursor.com/how-to-install-extension) was shared to help with the installation process.
- **Career channel turns into Cursor evangelism**: It's humorously noted that the career channel has become dominated by discussions advocating for **Cursor** over other tools.

**Link mentioned**: <a href="https://www.cursor.com/how-to-install-extension">Extension | Cursor - The AI-first Code Editor</a>: no description found

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1255604696694128782)** (4 messages): 

- **angry.penguin steps up to prevent future spam**: A member volunteered to become a mod to help prevent future spam incidents. The admin responded with gratitude and promptly granted mod status.
- **Spam cleanup completed**: After being granted mod status, the same member reported that they had also cleaned up the existing spam, ensuring a cleaner chat experience moving forward.
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1255946358796582983)** (2 messages): 

- **Gemma 2 support on the horizon**: A user inquired about the addition of **Gemma 2 support**. The response highlighted that while it's on their radar, they also welcome community contributions if someone wants to try using the model immediately.
  

---



### **DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/)** (1 messages): 

le_mess: good work 🙂 Would you mind sharing the training code?
  

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
