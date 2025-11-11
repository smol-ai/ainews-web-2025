---
id: MjAyNS0x
title: not much happened today
date: '2025-11-10T05:44:39.731046Z'
description: >-
  **Moonshot AI's Kimi K2 Thinking** AMA revealed a hybrid attention stack using
  **KDA + NoPE MLA** outperforming full MLA + RoPE, with the **Muon optimizer**
  scaling to ~1T parameters and native **INT4** QAT for cost-efficient
  inference. K2 Thinking ranks highly on **LisanBench** and **LM Arena Text**
  leaderboards, offering low-cost INT4 serving and strong performance in Math,
  Coding, and Creative Writing. It supports heavy agentic tool use with up to
  300 tool requests per run and recommends using the official API for reliable
  long-trace inference. **Meta AI** released the **Omnilingual ASR** suite
  covering 1600+ languages including 500 underserved, plus a 7B wav2vec 2.0
  model and ASR corpus. Additionally, the **Gelato-30B-A3B** model for computer
  grounding in GUI manipulation agents outperforms larger VLMs, targeting
  immediate agent gains. Qwen's image-edit LoRAs and light-restoration app were
  also highlighted.
companies:
  - moonshot-ai
  - meta-ai-fair
  - togethercompute
  - qwen
models:
  - kimi-k2-thinking
  - kimi-k3
  - gelato-30b-a3b
  - omnilingual-wav2vec-2.0
topics:
  - attention-mechanisms
  - quantization
  - fine-tuning
  - model-optimization
  - agentic-ai
  - speech-recognition
  - multilingual-models
  - gui-manipulation
  - image-editing
  - dataset-release
people:
  - yuchenj_uw
  - scaling01
  - code_star
  - omarsar0
  - kimi_moonshot
  - anas_awadalla
  - akhaliq
  - minchoi
---


**a quiet day**

> AI News for 11/7/2025-11/10/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (201 channels, and 12566 messages) for you. Estimated reading time saved (at 200wpm): 1015 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

The [Kimi K2 AMA](https://www.reddit.com/r/LocalLLaMA/comments/1oth5pw/ama_with_moonshot_ai_the_opensource_frontier_lab/) is getting a lot of buzz.

---

# AI Twitter Recap

**Moonshot AI’s Kimi K2 Thinking: AMA takeaways, evals, INT4 design, and upcoming vision**

- **AMA highlights (architecture, training, roadmap)**: From the Kimi K2 Thinking AMA: the oft-cited “$4.6M training cost” is not official; training ran on H800s; a hybrid attention stack using **KDA (Kimi Delta Attention) + NoPE MLA** outperformed full MLA + RoPE; the **Muon optimizer** reportedly scales well to ~1T parameters and is in PyTorch stable; and K2 Thinking is natively **INT4** via QAT for lower-cost inference on non-Blackwell GPUs. The team says **Kimi K2 will get vision**, and hinted that K3 will “likely use KDA or some other hybrid attention.” Timing quip on K3: “before Sam’s trillion‑dollar data center is built.” Sources: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1987940704929395187), [@scaling01](https://twitter.com/scaling01/status/1987916859400659011), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1987941323400507850), [@code_star](https://twitter.com/code_star/status/1987917177417289794), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1987955443420065816).
- **Evals and pricing**: On LisanBench, K2 Thinking is the best open-weight model and ranks ~7th overall (between GPT‑5 and GPT‑5‑Mini), setting new high scores on several items ([@scaling01](https://twitter.com/scaling01/status/1987952884927934966)). On the LM Arena Text leaderboard it’s the #2 open-source model (MIT-modified), tied at #7 overall, with strong Math/Coding/Creative Writing and top-tier Occupational performance ([@arena](https://twitter.com/arena/status/1987947219224526902), [details](https://twitter.com/arena/status/1987947222299013630), [try it](https://twitter.com/arena/status/1987947224173781185)). Arena also notes K2 Thinking exposes unrestricted chain-of-thought and was post‑trained with QAT, enabling **low-cost INT4 serving**; they cite pricing of $0.15 / $2.5 per million tokens vs Claude Sonnet 4.5 at $3 / $15 ([@arena](https://twitter.com/arena/status/1987947219224526902)).
- **Agentic tool use and inference guidance**: K2 Thinking supports heavy agentic workflows—reports of **200–300 tool requests** in a single run—keeping tool calls inside the reasoning trace to prevent drift ([demo thread](https://twitter.com/omarsar0/status/1987912692099682399), [@togethercompute](https://twitter.com/togethercompute/status/1988009780149878904)). For reliable benchmarking, Moonshot recommends using the official “kimi‑k2‑thinking‑turbo” endpoint, enabling streaming, temp=1.0, generous max_tokens (Reasoning 128k | Coding 256k), plus retries; they observed >20pp accuracy variance across third‑party providers and are publishing a Vendor Verifier ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1987892275092025635)). Several users report long‑trace failures via OpenRouter, advising to use the official API for long reasoning ([@scaling01](https://twitter.com/scaling01/status/1987938809628291168)). Tech deep‑dive on K2 Thinking hosted by Together on Nov 19 ([@togethercompute](https://twitter.com/togethercompute/status/1988009777247510564), [model access](https://twitter.com/togethercompute/status/1988011880443470217)).

**Speech and Computer-Use Models: Meta’s Omnilingual ASR and Gelato-30B-A3B**

- **Meta Omnilingual ASR (open source)**: Release of a suite of ASR models (300M–7B) covering **1600+ languages**, including **500 never previously served**. Also released: a **7B Omnilingual wav2vec 2.0** representation model and an **Omnilingual ASR Corpus** spanning 350 underserved languages. Models and dataset are open-sourced ([announcement](https://twitter.com/AIatMeta/status/1987946571439444361), [details + downloads](https://twitter.com/AIatMeta/status/1987957744138416389)).
- **Gelato-30B-A3B (computer grounding for agents)**: New “computer use” model trained on open Click‑100k, hitting **63.8% ScreenSpot‑Pro** and **69.1% OS‑World‑G**, outperforming specialized GTA1‑32B and even large VLMs ~8× its size (e.g., Qwen3‑VL‑235B). Targets immediate gains for GUI manipulation agents ([thread](https://twitter.com/anas_awadalla/status/1987913284989985092)). Also notable: Qwen’s image-edit LoRAs and light‑restoration app for quick relighting and shadow removal ([examples](https://twitter.com/minchoi/status/1988008926797787208), [dataset link](https://twitter.com/_akhaliq/status/1987989916974829809)).

**Data and Pretraining: Synthetic data, curriculum, and eval design**

- **SYNTH + Baguettotron**: Release of a fully synthetic generalist pretraining dataset (SYNTH) and two new reasoning models trained exclusively on it. With only **200B tokens**, “Baguettotron” is claimed to be best‑in‑class in its size range and SOTA on non‑code tasks (including math) per the authors’ reporting ([announcement](https://twitter.com/Dorialexander/status/1987930819021635964), [follow‑up](https://twitter.com/Dorialexander/status/1987977993440936433)). Commentary frames this as a step toward a “cognitive core” and explores non‑log‑scale scaling plots ([context](https://twitter.com/willccbb/status/1987998615785402785), [discussion](https://twitter.com/lateinteraction/status/1988016952451735772)).
- **Curriculum, RLVR scaling, and eval hardening**: Proposals to let models dynamically discover what data to see and when ([@joemelko](https://twitter.com/joemelko/status/1987715636861251667)); questions on whether scaling RLVR compute 10–1000× frontier baselines yields genuinely new knowledge beyond pretraining ([@YangYue_THU](https://twitter.com/YangYue_THU/status/1987716984524730604)). Benchmark designers are urged to “train on the test” to expose shortcuts and non‑visual exploits ([@sainingxie](https://twitter.com/sainingxie/status/1988019293926080611)). A recurring theme: high‑leverage leadership activity is still “labeling data” ([@model_mechanic](https://twitter.com/model_mechanic/status/1987945123439931785)). For longer‑horizon framing, see Fei‑Fei Li’s essay on building and using world models to unlock spatial intelligence ([thread](https://twitter.com/drfeifei/status/1987891210699379091)).

**Scaling Infra: GPUs, kernels, and giga‑scale data centers**

- **Hardware + kernels**: AMD and Modular report **2.2× faster inference in 14 days** on the Instinct MI355X ([@AMD](https://twitter.com/AMD/status/1987898172484567238)). NVIDIA detailed TensorRT‑LLM’s Wide Expert Parallelism on **GB200 NVL72** systems for MoE scaling ([summary](https://twitter.com/dl_weekly/status/1987913458654786008)). A Blackwell NVFP4 kernel competition kicked off (first task: NVFP4 GEMV) ([@a1zhang](https://twitter.com/a1zhang/status/1987972190898450922)).
- **Data centers at GW scale**: Epoch AI analyzes permits/satellite imagery and forecasts first **gigawatt‑scale data centers** online by 2026 as hyperscalers compress build times to 1–2 years; includes a Frontier Data Centers dataset and methods write‑up ([overview](https://twitter.com/EpochAIResearch/status/1987938542094610927), [thread](https://twitter.com/EpochAIResearch/status/1987944116861522227)).
- **Market/stack moves**: SemiAnalysis reports some frontier labs see **MI450X UALoE72** with strong perf/TCO for inference, amid reports of aggressive AMD incentives ([rumor](https://twitter.com/SemiAnalysis_/status/1988044940149235844)). H100/H200 spot price increases are anticipated in Q4’25 ([@FundaBottom](https://twitter.com/FundaBottom/status/1987905008541831521)), with practitioners expecting long productive lifespans for H100s even post‑Blackwell ([@code_star](https://twitter.com/code_star/status/1988062247818850421)). Enterprise stacks: Siemens shared an open‑source‑first platform optimized by vLLM on a sustainable mixed‑gen NVIDIA cluster ([@NVIDIAAIDev](https://twitter.com/NVIDIAAIDev/status/1987944094883037559)); Baseten pushes “own your weights” training infra ([@basetenco](https://twitter.com/basetenco/status/1987943307532476746)). A broader take frames GPUs as “reserve currency” in the intelligence age, with CUDA as convertibility and specialized clouds as “central banks” ([analysis](https://twitter.com/TheTuringPost/status/1988002749452349495)). OpenAI continues to staff core compute infra ([@gdb](https://twitter.com/gdb/status/1987996461846659372)).

**Agents, auth, and evaluation tooling**

- **Secure auth for agents**: Current web auth standards don’t fit headless agent workflows (no browser/redirects); OAuth is human‑centric and static keys are risky. MCP isn’t an auth layer; it standardizes tool/resource discovery for agents. Expect rapid spec evolution and industry‑wide auth solutions purpose‑built for agents ([@_philschmid](https://twitter.com/_philschmid/status/1987889931822236059)).
- **Self‑evolving agents (GEPA)**: OpenAI x Bain’s new cookbook shows agents that reflect, learn from feedback, and evolve their own instructions; GEPA was featured, with developers highlighting wild combos like Python’s inspect + GEPA ([@DSPyOSS](https://twitter.com/DSPyOSS/status/1988021062727020589), [@LakshyAAAgrawal](https://twitter.com/LakshyAAAgrawal/status/1988008687156556200), [@JoshPurtell](https://twitter.com/JoshPurtell/status/1988025269006069845)).
- **Evals and reliability**: A multi‑perspective eval talk (data, HCI, metrics, tooling) is recommended viewing ([@HamelHusain](https://twitter.com/HamelHusain/status/1987965289758421424)). Together AI published a benchmarking guide ([@togethercompute](https://twitter.com/togethercompute/status/1987949723106557975)). Weave adds dashboards and custom scorers to systematically surface LLM hallucinations in logs ([@weave_wb](https://twitter.com/weave_wb/status/1987946840550240294)). New agent releases include FlowAgent for orchestrating complex Web3 tasks on LangChain/LangGraph ([@LangChainAI](https://twitter.com/LangChainAI/status/1988012398176071728)).

**Top tweets (by engagement)**

- **10,000 hours egocentric robotics dataset (open)**: 2,153 workers, 1.08B frames—the “era of data scaling in robotics is here” ([@eddybuild](https://twitter.com/eddybuild/status/1987951619804414416)).
- **Meta’s Omnilingual ASR**: 1600+ languages; 500 first‑time; open models and corpus ([@AIatMeta](https://twitter.com/AIatMeta/status/1987946571439444361)).
- **Fei‑Fei Li on spatial intelligence and world models**: “turning seeing into reasoning” ([@drfeifei](https://twitter.com/drfeifei/status/1987891210699379091)).
- **CMU “Intro to Modern AI” course (Z. Kolter)**: Early‑undergrad chatbot‑from‑scratch, materials to be released ([@zicokolter](https://twitter.com/zicokolter/status/1987938761498411376)).
- **Dynamic mixed precision**: “optimize for least energy + flips” as a path forward ([@elonmusk](https://twitter.com/elonmusk/status/1987994042937036805)).
- **ARC‑AGI v1 claim**: human‑level (85%) in <12 hours for <$10k with multi‑agent evolutionary test‑time compute and GPT‑5 Pro; community scrutiny ongoing ([@jerber888](https://twitter.com/jerber888/status/1987982067116777521)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Strix Halo Networking Performance Analysis

- [**I tested Strix Halo clustering w/ ~50Gig IB to see if networking is really the bottleneck**](https://www.reddit.com/r/LocalLLaMA/comments/1ot3lxv/i_tested_strix_halo_clustering_w_50gig_ib_to_see/) (Activity: 601): **The post discusses an experiment to test whether networking is a bottleneck in a Strix Halo clustering setup using InfiniBand and Thunderbolt connections. The author used Mellanox ConnectX-5 Ex 100 Gig NICs to achieve approximately 55 Gbps networking, compared to 10 Gbps over Thunderbolt. The results showed that Thunderbolt's 10 Gbps performance was nearly equivalent to the 50 Gbps InfiniBand in terms of token generation speed, suggesting that high bandwidth may not be necessary for llama.cpp with Strix Halo. The experiment also noted that network usage was low, indicating that latency rather than bandwidth might be the limiting factor. The author concludes that fancy IB cards are not needed for usable results with llama.cpp on Strix Halo, at least until RCCL support is available.** One commenter noted that the test might be meaningless because llama.cpp doesn't use tensor parallelism, suggesting that testing with TP on VLLM or Sglang would be more appropriate. Another commenter referenced a similar experiment by Jeff Geerling with poor results, suggesting a comparison of findings.
    - Only_Situation_4713 points out that the test was not meaningful because Llama cpp does not utilize tensor parallelism (TP), which means all operations are performed sequentially. They suggest testing with TP enabled on frameworks like VLLM or Sglang to get a more accurate assessment of performance bottlenecks.
    - wishstudio highlights the importance of network latency in tensor parallelism (TP) setups. They note that while data exchange in TP is minimal, synchronization is required per layer, which can be a bottleneck. For instance, with a model like gpt-oss-120b having 36 layers, typical Ethernet latency of 250 microseconds could significantly slow down performance, whereas InfiniBand (IB) can reduce latency to single-digit microseconds, potentially improving real-world performance.
    - eleqtriq references a video by Jeff Geerling, noting that his results were poor when testing similar setups. This suggests that networking might indeed be a bottleneck, and comparing results could provide insights into the performance differences and potential optimizations.

### 2. Qwen3-VL OCR Capabilities and Comparisons

- [**Qwen3-VL's perceptiveness is incredible.**](https://www.reddit.com/r/LocalLLaMA/comments/1ot95gj/qwen3vls_perceptiveness_is_incredible/) (Activity: 437): **The post discusses the performance of the** `Qwen3-VL-8B-Instruct-GGUF` **model in optical character recognition (OCR) tasks, specifically its ability to accurately transcribe and provide bounding boxes for words in a 4k image. The model, with an image token count of** `2300` **and a temperature of** `0`**, successfully identified all six words in the image with precise bounding boxes, outperforming other models like Gemini 2.5 pro, Claude Opus 4, ChatGPT 5, DeepSeekOCR, and PaddleOCR-VL-0.9B. Notably, GLM-4.5V also achieved perfect results, but the post highlights the efficiency of Qwen3-VL given its smaller size and lack of specific OCR tuning. [Link to the model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).** Commenters note the impressive performance of the `Qwen3-VL-8B` model, especially given its smaller size compared to larger models like the `30B`. One user plans to update their OCR application to use this model, indicating its practical utility. Another comment suggests that the `8B` model is a 'no-brainer' choice for many applications, highlighting its efficiency and effectiveness.
    - MaxKruse96 highlights the performance of the Qwen3-VL model, particularly the 8B variant, which is noted for its efficiency at `q8` or `BF16` precision. This model is considered a standout until the release of GLM-4.5V and the 235B VL, indicating a significant gap in comparable models. The discussion suggests that Qwen3-VL is setting a new standard in model performance.
    - Putrid_Passion_6916 mentions updating their project, [deepseek_ocr_app](https://github.com/rdumasia303/deepseek_ocr_app), to incorporate Qwen3-VL, emphasizing the model's impressive capabilities. They note that smaller parameter models like the 8B or 4B are sufficient for many tasks, offering similar performance to larger models like the 30B, which highlights the efficiency and potential cost savings of using smaller models.
    - cygn discusses the importance of image resolution in model performance, using Gemini 2.5 Pro in AI Studio as an example. They note that choosing between medium and low resolution can impact the results, suggesting that higher resolutions may yield better outcomes. This highlights the need for careful consideration of input quality in model evaluations.

### 3. BERT Chatbot with dLLM

- [**BERTs that chat: turn any BERT into a chatbot with dLLM**](https://www.reddit.com/r/LocalLLaMA/comments/1osydym/berts_that_chat_turn_any_bert_into_a_chatbot_with/) (Activity: 390): **The post introduces dLLM, a library that enables turning any BERT model into a chatbot by leveraging *discrete diffusion* techniques. The approach allows BERT models, such as [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large), to perform conversational tasks with performance comparable to larger models like [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B). The project provides open-source code, checkpoints, and a detailed [W&B report](https://api.wandb.ai/links/asap-zzhou/101h5xvg) for transparency and reproducibility. The method focuses on parallel token generation, diverging from traditional left-to-right autoregressive models, and is designed to be a comprehensive tutorial resource for diffusion language models.** One commenter expressed surprise that the diffusion model did not decode many tokens simultaneously or in a non-sequential order, which they believed was the primary advantage of diffusion models.
    - ithkuil raises a technical point about the expected behavior of diffusion models, noting that they typically decode many tokens simultaneously or in a non-sequential order. This expectation contrasts with traditional sequential decoding methods, suggesting a potential area of innovation or misunderstanding in the implementation of dLLM for chatbots.
    - robberviet inquires about the data used for training the model, pointing out that the repository only mentions 'public data' without specifics. This highlights a common issue in AI projects where the lack of detailed data provenance can affect reproducibility and trust in the model's performance.
    - random-tomato comments on the novelty of the chat interface for diffusion language models, indicating that functional chat interfaces are rare for such models. This suggests that the implementation of dLLM might offer unique capabilities or improvements over existing solutions.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. China's AI Advancements and Rivalry

- [**China really carrying open source AI now**](https://www.reddit.com/r/DeepSeek/comments/1ot9y1j/china_really_carrying_open_source_ai_now/) (Activity: 471): **The image is a meme illustrating the perceived rivalry between China and the United States in the open-source AI sector. It uses the symbolism of a dragon and an eagle to represent China and the US, respectively, with logos of AI and tech companies suggesting their involvement in this competitive landscape. The post and comments highlight the sentiment that China is making significant strides in open-source AI, with some users noting that Chinese models like Deepseek and Qwen offer comparable quality to American models, often at no cost. This reflects a broader discussion on the democratization of AI and the strategic moves by China to lead in this domain.** Some commenters express the view that China's open-source AI models are a strategic move against US companies, offering high-quality alternatives that challenge the dominance of American models. There is also a sentiment that Chinese models are democratizing AI by providing free access to high-quality tools.
    - A user highlights the performance parity between high-tier American AI models and free Chinese models, noting that despite paying for the most expensive plans from US companies, the quality is comparable to free offerings from China. This suggests that Chinese models are effectively democratizing AI by providing high-quality models at no cost, challenging the traditional pricing models of US companies.
    - Another user points out a critical distinction in the AI community: the difference between 'open source' and 'open weight' models. While many Chinese models are referred to as open source, they are technically 'open weight,' meaning the model weights are available but not the source code. This distinction is crucial for developers who need full transparency and control over the model's implementation.
    - A user mentions specific Chinese models like Deepseek and Qwen, noting that Qwen is particularly notable for not simply agreeing with the user, which can be a valuable trait for more nuanced AI interactions. This highlights the diversity and sophistication of Chinese AI models in providing varied user experiences.
- [**China trained a GPT-5 competitor (Kimi K2) for only $4.6 million.**](https://www.reddit.com/r/ChatGPT/comments/1ot7fl4/china_trained_a_gpt5_competitor_kimi_k2_for_only/) (Activity: 1196): **The image presents a performance comparison of the Kimi K2 model, a Chinese-developed AI, against other models, including GPT-5. Kimi K2 is highlighted for its strong performance in agentic search and coding tasks, despite its relatively low training cost of $4.6 million. This suggests that Kimi K2 is a cost-effective competitor in the AI landscape, particularly in specific technical domains.** Some users note that while Kimi K2 is a good model, it may not match the capabilities of GPT-5 or other advanced models like Grok 4 or DeepSeek. However, others find it a reliable daily-use model, indicating its practical utility despite some limitations.
    - NoDay1628 highlights that while Kimi K2 is touted as a cheaper alternative to GPT-5, the true measure of an AI model's capability goes beyond just the number of parameters or the training budget. They emphasize the importance of 'nuance reasoning and safety,' suggesting that a model's practical performance can differ significantly from its theoretical specifications.
    - BuccellatiExplainsIt raises skepticism about the claimed $4.6 million training cost for Kimi K2, drawing parallels to previous instances like Deepseek where reported figures were misleading. They point out the lack of transparency and accountability in these claims, suggesting that the actual costs and capabilities might be different from what's advertised.
    - JackStrawWitchita shares practical insights from using Kimi K2, noting that while it's not perfect, it serves well as a daily driver. They suggest that experimenting with different models helps in understanding the strengths and weaknesses of each, indicating that Kimi K2 offers a viable alternative to more established models like ChatGPT.

### 2. Humorous AI Critiques and Memes

- [**Thoughts?**](https://www.reddit.com/r/OpenAI/comments/1otasm8/thoughts/) (Activity: 3090): **The image is a meme that humorously critiques the reliability of AI, specifically ChatGPT, in providing accurate information about potentially dangerous topics like poisonous berries. It highlights the risks of relying on AI for critical advice without cross-verifying with authoritative sources. The meme underscores the importance of human judgment and the limitations of AI in handling nuanced or life-threatening queries.** Commenters emphasize the importance of not relying on AI for medical advice, noting that while AI can provide information, it should not replace professional consultation. They also point out that AI can correctly identify known poisonous items if queried accurately.
    - Sluipslaper highlights a practical test of ChatGPT's ability to identify poisonous substances, suggesting that when queried about a known poisonous berry, ChatGPT correctly identifies it as poisonous. This implies that the model has access to reliable data sources and can provide accurate information on specific queries, though it should not replace professional advice.
    - Caddap compares the use of ChatGPT to performing a Google search and emphasizes the importance of using it as a tool rather than a replacement for personal research. The comment underscores the necessity of due diligence when interpreting AI-generated information, as the tool's power lies in its correct application rather than blind trust.
    - LunaticMosfet points out that ChatGPT typically provides cautious and detailed responses, even when faced with potentially incorrect data. The model tends to highlight corner cases and avoids making absolute statements, which suggests a design focus on providing balanced and careful advice rather than definitive answers.
- [**Sora 3 out before November 2026**](https://www.reddit.com/r/singularity/comments/1ot1m9w/sora_3_out_before_november_2026/) (Activity: 499): **The image is a meme that humorously comments on the anticipated delay of "GTA 6" until November 2026, suggesting that "Sora 3" will be released before it. The image features characters reminiscent of a typical action game scene, with a man holding a gun and a woman with a briefcase, set against a cityscape. The comments reflect a satirical take on the slow development process of major game titles, with some users joking about the potential for AI to accelerate game development, possibly even releasing future versions before the current ones are completed.** Commenters humorously speculate about the role of AI in game development, suggesting that AI advancements could lead to faster releases of game sequels, potentially even before the current versions are completed.
    - Weekly-Trash-272 highlights the rapid pace of AI model development, suggesting that several new models could emerge before the release of GTA 6. This underscores the accelerating capabilities of AI, which, while not yet able to autonomously create games, are closing the gap in terms of potential applications in game development.
    - Setsuiii points out the risks associated with delaying game releases, particularly in the context of rapidly evolving technology. They note that by the time a game like GTA 6 is released, the development techniques and technologies could be outdated, emphasizing the need for developers to adapt to new methods and tools to stay relevant.
    - Normal_Pay_2907 speculates on the timeline for OpenAI's automated research assistant, suggesting it could be completed before the release of Sora 3. This reflects the broader trend of AI tools being developed to assist in complex tasks, potentially transforming research and development processes across industries.

### 3. AI in Politics and Economics

- [**Sen. Bill Cassidy on the floor of the Senate with what looks like an AI-generated graphic**](https://www.reddit.com/r/ChatGPT/comments/1ot0ddh/sen_bill_cassidy_on_the_floor_of_the_senate_with/) (Activity: 1693): **The image in question shows Sen. Bill Cassidy on the Senate floor with a graphic that appears to be AI-generated, as noted by the presence of 'suspicious artifacts' such as the '80%' and dollar signs. The graphic is intended to illustrate the allocation of healthcare dollars, contrasting traditional insurance models with a prefunded flexible spending account approach. The graphic's cartoon-like simplicity and potential AI generation raise questions about the accuracy and professionalism of the visual aid used in a formal setting.** Commenters express skepticism about the graphic's accuracy and the understanding of the issue by politicians, with one noting the comparison as 'apples and broccoli' and another suggesting the graphic is '100% AI' generated.
- [**OpenAI Could Be Blowing As Much As $15 Million Per Day On Silly Sora Videos**](https://www.reddit.com/r/OpenAI/comments/1otjj7i/openai_could_be_blowing_as_much_as_15_million_per/) (Activity: 830): **OpenAI is reportedly incurring costs of up to** `$15 million per day` **on its AI video application, Sora, which has sparked discussions about the sustainability of such high expenditures. This financial strategy could significantly affect OpenAI's business model and future funding approaches. The article suggests that OpenAI might be spending more than a quarter of its revenue on this project, raising questions about the long-term viability of this investment. For more details, see the [Forbes article](https://go.forbes.com/B53aCk).** Commenters draw parallels between OpenAI's strategy and that of companies like **Amazon** and **Uber**, which initially operated at a loss to build a customer base. The debate centers on whether high demand for Sora indicates its value and potential for future profitability, despite current losses.
- [**Peak AI**](https://www.reddit.com/r/singularity/comments/1otfhbn/peak_ai/) (Activity: 1350): **Steve is an AI agent framework that allows users to describe tasks in natural language, which the AI then interprets and executes. The project is hosted on [GitHub](https://github.com/YuvDwi/Steve) and aims to simplify user interaction by acting as a single or multiple agents to understand and perform tasks based on contextual understanding. This could be particularly useful in gaming scenarios where players manage complex systems like cities or armies, allowing them to issue commands verbally rather than through traditional controls.** Commenters discuss the potential of AI companions in gaming, suggesting that while the concept may seem trivial, it could revolutionize gameplay by simplifying user interaction. However, they also note the technical challenge of translating AI-generated text into actionable game events.
    - AleriaGoodpaw highlights the technical challenge of integrating AI chatbots into gaming, emphasizing the difficulty in translating 'AI chatbot text mess' into actionable game events. This involves complex natural language processing and real-time decision-making algorithms to ensure that AI can effectively interpret and execute player commands within a game environment.
    - Scandinavian-Viking- suggests a potential application of AI in gaming where players could control complex systems like cities or armies through natural language commands. This would require sophisticated AI capable of understanding and executing strategic-level decisions, potentially transforming the user interface and experience in strategy games.
    - rowc99 discusses the rapid progression of AI technology, suggesting that skepticism based on current limitations fails to account for the exponential growth in AI capabilities. This perspective implies that future AI could significantly enhance gaming experiences, particularly in terms of immersion and interaction, as AI and VR technologies become more advanced and accessible.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**The Kimi K2 Uprising and Anticipation for the Next Generation**

- **Kimi K2 Smashes Leaderboards and Expectations**: Moonshot AI's **Kimi-K2-Thinking** model is making waves, ranking as the **#2 open-source model** on the [LMArena Text leaderboard](https://lmarena.ai/leaderboard/text/overall) with an impressive expert score of **1447**. It also outperforms **GPT-5** and **Claude 4.5** on the [Tau2 Bench Telecom benchmark](https://xcancel.com/natolambert/status/1986507284491440623) at a fraction of the cost, though Unsloth's team has reported a potential issue on their [GitHub](https://github.com/unslothai/unsloth).
- **GPT-5.1 and Gemini 3 Rumors Fuel Hype Engine**: Speculation is rampant for a potential **GPT-5.1 Pro** release, with some suggesting **OpenAI** is waiting for Google to make the first move and that the **Polaris Alpha** model on OpenRouter is an early version. Meanwhile, engineers eagerly await **Gemini 3**, debating its potential to disrupt coding jobs, though some remain skeptical given the limitations of current models.
- **Sora 2 Quality Plummets, While Open-Source Voice AI Shines**: Users are reporting a noticeable decrease in **Sora's video quality**, complaining about static subjects and poor audio, with one user claiming it has the *worst video and audio quality of all video gens currently!* In contrast, a new SOTA open-source voice AI named **Maya1** debuted on [Hugging Face](https://huggingface.co/maya-research/maya1), featuring **3B** parameters and support for **20 human emotions** on a single H100.

**Kernel Wizards and Hardware Hackers Push Performance Limits**

- **Engineers Unleash GMP-Verified INT8 GEMM Kernel**: A developer released a GMP-verified exact **INT8×INT8→INT32 GEMM** kernel, achieving a stunning **300.26 T-ops/s** on an **A100**. The code, which demonstrates bit-for-bit correctness, is available for community verification and feedback in a [Colab notebook](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing) and [GitHub repo](https://github.com/playfularchitect/WarpFrac.git).
- **Modular's MAX Engine Crushes Competition on New Silicon**: **Modular's MAX**, an inference engine implemented in **Mojo**, is reportedly beating **TensorRT on B200** and AMD's offerings on the **MI355X**. This performance, combined with Mojo's goal of becoming a systems language with features like affine types, is generating significant buzz among HPC developers eager to avoid porting C++ packages to GPUs.
- **Coastal Air Corrodes RTX 3090, NPUs Lag Behind**: One user discovered their newly acquired **RTX 3090** was hitting high hotspot temperatures due to mineral buildup from a humidifier, sharing a [photo of the pacific ocean residue](https://photos.app.goo.gl/3UPTmQKzJo81trTx9) on the heatsink. Separately, discussions around using **NPUs** for LLMs concluded they are still significantly slower than dedicated GPUs, despite a [recent paper](https://arxiv.org/abs/2412.11053) demonstrating inference on an Intel AI Boost NPU.

**Developer Platforms Suffer Death by a Thousand Cuts**

- **Cursor Users Face Crashes, Cost Spikes, and Connection Woes**: Cursor users are reporting a slew of problems, including system-wide crashes on Mac M2s, unexpected cost spikes from **Sonnet 4.5** reaching **$1.02 NZD per minute**, and frequent disconnects with **Composor-1**. These issues are compounded by errors with student ID verification and *Unauthorized User API key* errors when using personal **OpenRouter** keys.
- **Perplexity Pro Users Hit With Hidden Limits and Bans**: The **Perplexity Pro** experience is souring for some, with users hitting non-obvious weekly **agent task limits** and context window caps, as shown in [this screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1437480737166327999/image.png?ex=691365c2&is=69121442&hm=2b15860ed2450a8cacc3ae448cf8effc4fbf7b73e597f69199d397e078e3e026&). Adding to the frustration, several users reported being banned from the referral program over alleged fraud, with one stating *Perplexity owe me 30 dollars*.
- **OpenAI Signals Assistant API's Doom, Aider Forks to Community Edition**: Developers are planning for the upcoming deprecation of **OpenAI's** `assistant` **API** in 2026, which will require converting training files to **JSONL** for its suggested replacement, the Responses API. In the agent space, development on **aider** has reportedly shifted to the community-driven `aider-ce` [branch](https://github.com/dwash96/aider-ce), which users are praising for *leaps-and-bounds improvements* and a *mindblowing* new agentic mode.

**Taming Model Quirks, From Censorship to Continual Learning**

- **AI Censorship Concerns Ignite Community Backlash**: Frustration is mounting over increasing **AI censorship**, with multiple users across servers worrying about a *tightly controlled* information environment. Some believe **OpenAI is depriving the public access to information**, while others note that overzealous safety features make models impractical for many technical applications.
- **Models Suffer Identity Crises and Memory Glitches**: Models are exhibiting bizarre behaviors, with **Qwen3-VL** becoming confused by **Ollama** and believing it's a text-only model despite processing image data. Similarly, a user reported **Gemma 4B** in LM Studio appeared to retain context across different chat histories, leading to speculation about a potential *flash attention bug*.
- **Google's "Nested Learning" Promises to End Catastrophic Forgetting**: Google introduced **Nested Learning**, a novel machine learning paradigm for [continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) that aims to solve catastrophic forgetting by treating models as layers of nested optimizers. While the concept has sparked interest, some engineers questioned why Google didn't test it against more standard continual learning setups, suggesting fine-tuning with a reference [paper](https://arxiv.org/abs/2510.19788).

**Open Source Projects Power Forward with New Tools and Workflows**

- **New Open-Source Tools Target Rust Coders and TPU Users**: An open-source [AI interface for Rust coding called Ploke](https://github.com/josephleblanc/ploke) was released, using native project parsing and automatic semantic search to improve context management. For large model acceleration, **AutoXLA** debuted on [GitHub](https://github.com/Locutusque/AutoXLA), an experimental library that automates model distribution and quantization for TPUs to achieve up to **4x** faster performance than standard Flash Attention.
- **ComfyUI Gets Professional Workflows for Production-Ready Images**: NexusAI has launched a suite of stable, production-ready [ComfyUI workflows](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows) on GitHub. The one-click workflows are designed for photorealistic, anime, and commercial image generation and are undergoing active refinement in **v1.0.1** to ensure consistent detail reproduction.
- **Engineers Tackle Agent Tool Sprawl with DSPy Planner**: A developer published a guide on [Solving Agent Tool Sprawl with DSPy](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy), using a DSPy-based planner and orchestrator to manage multi-agent tool use. This comes as DSPy continues to evolve, with a forthcoming PR to add **TOON** support and a proposal to integrate first-class support for coding **agent CLIs** based on the [Agent Client Protocol standard](https://github.com/agentclientprotocol/agent-client-protocol).

---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Sora 2 Pro sparks access debate**: Discussion arose around **Sora 2 Pro** access and account sharing, with some users criticizing the practice.
   - Arguments included fairness for paying users who follow the rules versus the common practice of families sharing accounts.
- **OpenAI faces rule criticism**: A user voiced frustration with **OpenAI**'s rules, suggesting that people shouldn't follow them, citing examples of **Spotify** and **Meta** allegedly violating rules.
   - This sparked a debate about ethics and fairness, with the user arguing, *if you steal less, you're a thief. If you steal more, you become a billionaire*.
- **Gemini 3 fuels anticipation**: Users eagerly await the release of **Gemini 3**, speculating on its capabilities and potential impact, especially in coding.
   - Some fear it could replace jobs, while others remain skeptical due to limitations of current AI models, with mentions of **Google AI Studio** and **Nano Banana 2**.
- **Nano Banana 2 incites hype and takedown Theories**: Enthusiasm surrounds the potential release of **Nano Banana 2** and its capabilities, with some users claiming it's already available.
   - However, concerns arose about a possible takedown, with one user reporting that the model was removed only 5 hours after launch, suspecting the mention of its name triggered the action.
- **Kimi-k2-thinking dominates Leaderboards**: The **Text leaderboard** saw an update, with `Kimi-k2-thinking` now ranked as the **#2 open source model** and tied for **#7 overall**, excelling in Math, Coding, and Creative Writing.
   - Check out the [Text leaderboard](https://lmarena.ai/leaderboard/text/overall) and [Expert leaderboard](https://lmarena.ai/leaderboard/text/expert>) to see the results, and an impressive Expert leaderboard score of **1447**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Gets Buggy!**: Users report **Comet browser** issues like **YouTube search malfunctions** and **video playback problems** with adblockers; restarting may help, and buttons are *not fast and smooth like chrome*.
   - One user shared [Comet's performance metrics](https://cdn.discordapp.com/attachments/1047649527299055688/1437072124056571936/image.png?ex=69133ab5&is=6911e935&hm=ca6f47a1f181693b3d60ad89c0ef742a27d0caab93088ee5246ae8b7aa8bbc91&).
- **YouTube and AdBlockers Clash!**: YouTube is cracking down on ad blockers, with Chromium updates impacting effectiveness, although [disabling adblock](https://link.to/adblock) can allow proper website function; some suggest **Brave** while others like **Comet**.
   - [This youtube link](https://www.opera.com/features/ad-blocker) was shared for adblocking tips.
- **Perplexity Referral Program: Fraudulent Fallout!**: Users report being banned from the **Perplexity referral program** over alleged fraud, leading to canceled commissions, with one user stating *Perplexity owe me 30 dollars*.
   - Theories circulate about a wave ban, referencing [Perplexity AI's help center](https://www.perplexity.ai/help-center/en/articles/11385821-partner-promotions-and-referral-programs) on how the referral system works.
- **Context Window Limits Spoil Perplexity Pro?**: Comet browser users are reporting hitting weekly **agent task limits** and experiencing a non-obvious **context window limit**, frustrating those with year-long subscriptions, as seen in [this screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1437480737166327999/image.png?ex=691365c2&is=69121442&hm=2b15860ed2450a8cacc3ae448cf8effc4fbf7b73e597f69199d397e078e3e026&).
   - Despite this, one user said the limits are *Better than ChatGPT Plus users' 40 uses a month*.
- **Rajkahini's Rocket Debut!**: The band **The Orbits** launched their debut single, **Rajkahini**, on [Spotify](https://open.spotify.com/track/227ZkkO3LKPVABsHOoDS3w?si=a8603fc7cbb14e2c), [YouTube Music](https://music.youtube.com/watch?v=GZAnCpgIO5g&si=QvIAfZLZdameuUfN), [Apple Music](http://itunes.apple.com/album/id/1850285754), and [Amazon Music](https://music.amazon.com/tracks/B0FYY1C2BR).
   - Lyrics for the song are available on [Genius](https://genius.com/The-orbits-indian-band-rajkahini-lyrics).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma Cache Glitch Mystifies Users**: A user reported that **Gemma 4B** seemed to retain context across different chat histories in LM Studio, an unexpected behaviour.
   - Another user anecdotally had **Gemma cache context before**, speculating it may be a *flash attention bug*.
- **Qwen VL Finally Sees the Light**: A user reported finally getting **Qwen3 VL** to work for LM Studio after downloading a clean version, noting it processes images quickly even with limited VRAM.
   - They speculated that **Qwen3 VL** would be good for **NPCs in games**, so they can actually see.
- **Wall Street Whisperer LLM Remains Elusive**: A user sought a finetuned LLM that could mimic the writing style of a top **Wall Street journalist**, dismissing generic models and prompting strategies.
   - The user insisted that system prompts and session initialization *fail completely* and are *all incorrect hallucination*.
- **3090 Chokes on Coastal Crud**: A user found that a newly acquired **RTX 3090** was experiencing high hotspot temperatures due to mineral deposits on the heatsink from a humidifier, sharing a [photo](https://photos.app.goo.gl/3UPTmQKzJo81trTx9) of the buildup.
   - The user also mentioned the card smelled like the pacific ocean, suggesting it was previously used in a coastal environment, and is planning to descale and repaste the GPU to resolve the issue.
- **NPUs Not Quite Primetime for LLMs**: There was discussion around the viability of using **NPUs** (Neural Processing Units) for running LLMs, referencing a [paper](https://arxiv.org/abs/2412.11053) demonstrating LLM inference on an Intel AI Boost NPU.
   - Despite the potential, members generally agreed that NPUs are significantly slower than dedicated GPUs for LLM tasks due to their limited TOPS (Tensor Operations Per Second) performance.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet 4.5 Costs Catch Users Off-Guard**: Users are reporting unexpected spikes in the cost of using **Sonnet 4.5**, with one user calculating expenses reaching **$1.02 NZD per minute**.
   - Some users are suggesting strategies like reserving **Sonnet 4.5** for planning and **Haiku 4.5** for writing to optimize costs, while others find **Haiku** needs too much micro-managing.
- **Composor-1 Connections Cause Catastrophes**: Several users reported frequent disconnects and stalling when using **Composor-1**, with a persistent *Planning next moves* message.
   - Potential workarounds include downgrading to **HTTP 1.1** or restarting Cursor, with theories suggesting test scripts leaving **ASYNC** connections open as a possible cause.
- **Student Sign-Up System Stumbles**: Users are running into errors while attempting to verify their student IDs, receiving an unspecified **Error** message.
   - Speculation suggests student signups may be temporarily suspended due to an increase in online-purchased IDs or that certain countries may no longer be supported for student discounts.
- **OpenRouter API Keys Trigger Access Alarms**: Users are encountering *Unauthorized User API key* errors while trying to use their own **OpenRouter** keys, with one reporting the problem persists even after re-entering the key.
   - Possible solutions involve verifying that the key is valid and enabled, disabling Auto-model selection, and confirming all Cursor settings are accurate.
- **Cursor Crashes Culminate in System Shutdowns**: Several users, particularly on Mac M2, are reporting system-wide crashes when opening Cursor, including purple screens and complete shutdowns, with one user losing all their historic chats for the 3rd time.
   - Troubleshooting suggestions include reinstalling Cursor, using a different user profile, and minimizing system resource load, but the underlying causes are still unclear.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NexusAI Unleashes ComfyUI Pro Workflows**: NexusAI has launched a suite of stable, production-ready [ComfyUI workflows](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows) tailored for photorealistic, anime, and commercial image generation, streamlining processes into one-click workflows.
   - These workflows are currently stable for basic image creation and are undergoing active refinement to ensure consistent detail reproduction across varying random seeds as part of the **v1.0.1** optimization.
- **Maya1 Debuts as SOTA Open-Source Voice AI**: A new **SOTA** *open-source voice AI* named **Maya1**, featuring **3B** parameters and designed to run on a single H100, has been introduced on [Hugging Face](https://huggingface.co/maya-research/maya1).
   - This **AI** supports *voice design and 20 human emotions*, marking a significant advancement in accessible voice technology.
- **Rustaceans Gain Open Source AI Interface**: An *open source* **AI interface** for **Rust** coding has been released, offering native parsing of projects to enhance the relevance and efficiency of LLM context management via automatic semantic search bm25 keyword, hosted on [Github](https://github.com/josephleblanc/ploke).
   - A new feature allows users to select models hosted on **OpenRouter** via an interactive overlay and an [example](https://cdn.discordapp.com/attachments/897390720388825149/1437454527027740802/quick_example.webm?ex=69134d59&is=6911fbd9&hm=7320ebf8be3a9c2b4244b56acdfa66aed9c685b0496dac37171051c2ebb2fdcf&) has been provided.
- **AutoXLA Accelerates Large Model Performance on TPUs**: **AutoXLA**, an experimental library, automates the distribution, optimization, and quantization of large language models for TPUs using PyTorch/XLA, achieving up to **4x** faster performance than standard Flash Attention implementations, and is available on [GitHub](https://github.com/Locutusque/AutoXLA).
   - By extending the Hugging Face Transformers interface with TPU-aware features like automatic sharding, custom attention kernels, and quantization-aware loading, it streamlines large-scale deployment and training workflows.
- **Smol-Course Suffers Stalled SFT Unit**: Multiple pull requests (PRs) addressing issues in the **smol-course**, particularly with the **Markdown instructions** for the **SFT unit**, remain open and unreviewed on [GitHub](https://github.com/huggingface/smol-course).
   - This backlog potentially impedes course improvements, and members have expressed willingness to assist in reviewing the PRs to address the identified issues.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **INT8 GEMM kernel boasts GMP verification!**: A member released a GMP-verified exact **INT8×INT8→INT32 GEMM** kernel, highlighting throughput on an **A100** (**300.26 T-ops/s** macro, **2.026 T-ops/s** micro) and bit-for-bit correctness, inviting community feedback.
   - Code and verification are available in a [Colab notebook](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing) and [GitHub repo](https://github.com/playfularchitect/WarpFrac.git).
- **Blackwell architecture hits the benches**: Members discussed [a paper](https://arxiv.org/abs/2507.10789v2) featuring microbenchmarking on the upcoming **Blackwell** architecture, especially the **5080**, and whether *consumer* **Blackwell (sm120)** differs from *data-center* **Blackwell (sm100)**.
   - A member suggested this would be an ideal working group to get a similar paper working for **GB200**.
- **Helion's Attention Kernel Performance Race**: Members discussed the [Helion attention implementation](https://github.com/pytorch/helion/blob/main/examples/attention.py) in relation to **FlexAttention**, observing that the **Helion** code has superior code quality over existing **Triton** implementations.
   - It was highlighted that the [attention kernel's performance numbers](https://cdn.discordapp.com/attachments/1425531180002054195/1436541653442887801/attention_perf.png?ex=691346eb&is=6911f56b&hm=dfbe035e2a6290dca86c612c31c2327934f6afffe40d6fe2fa5e7ce395feb546) are published, open source, and reproducible, and the [B200 kernel is available](https://github.com/pytorch/helion/blob/main/examples/blackwell_attention.py).
- **NVSHMEM Kernels Speed LLM Inference**: A member shared their team's work on writing [low-latency communication kernels with **NVSHMEM**](https://pssg.cs.umd.edu/blog/2025/beyond-nccl/) for **LLM inference**, seeking feedback on multi-node communication performance and sparking interest in **nvshmem4py** integration.
   - A member uses a lot of **nvshmem device APIs**, and offered to help with library init and potentially open a PR demoing what it might look like.
- **WarpTrace visualizer warms up pipelined kernels**: A member introduced **warptrace**, a tool with an associated visualizer for observing pipelined kernels, noting it takes about **50 microseconds** to start up the atomics systems, even if you've warmed up the kernel previously.
   - The [code](https://github.com/aikitoria/nanotrace) is still undergoing cleanup before it can go on GitHub and can be found [here](https://aikitoria.github.io/nanotrace/).



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Kimi K2 Suffers Prompt-Induced Crashloop**: **Kimi K2 Thinking** experienced a crashloop across multiple providers due to a specific prompt, but the issue has been **resolved** through collaborative efforts.
   - Teams are actively working to identify the cause.
- **Orchid AI Assistant's Distant Debut**: The estimated release date for **Orchid AI Assistant** is projected to be within the next **2 to 48 months**.
   - A member reacted with the word *"crazy"* to this long and vague estimate.
- **OpenRouter Ponders Video Support Expansion**: A user wished for **OpenRouter** to support videos and text-to-speech (TTS) functionality, referencing [this tweet](https://x.com/scaling01/status/1986886020067938749).
   - Another member suggested including a brief **technical segment** in the **OR Show**, such as a screen recording with a short discussion.
- **Gemini 2.5 Token Consumption Alarm**: A user reported that a 24-second, 900x600 video uploaded to **Gemini 2.5 Flash** consumed over **800k input tokens**, despite Google's documentation.
   - The [token documentation](https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens) specifies fixed rates of **263 tokens per second for video**.
- **Automated Capabilities Scanning Idea Sparks**: A member suggested implementing some kind of **automated capabilities scanning** to detect changes in models/providers over time.
   - They linked to an [article on Cursor](https://joincolossus.com/article/inside-cursor/) as an example, describing how a *basic getWeather tool call* could be used to check for functionality changes.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora's Quality Soreness Spurs Speculation**: Users report decreasing **Sora video quality**, with complaints about subjects standing still and poor audio, sparking concerns that **Sora 2** might have the *worst video and audio quality of all video gens currently!*
   - However, there is optimism that **Sora's integration with GPT-5** could lead to improvements.
- **Whispers of GPT-5.1 Release This Week**: Speculation surrounds a potential **GPT-5.1 Pro** release, possibly this week, with one member stating that *Openai is waiting for google to drop it first.*.
   - The model **Polaris Alpha** on **OpenRouter** is also rumored to be an early form of **GPT-5.1**.
- **AI Censorship Concerns Catch Fire**: Multiple users express concerns over increasing **AI censorship**, worrying about the creation of a *tightly controlled* information environment.
   - Some believe **OpenAI is depriving the public access to information**, causing more harm to society than good.
- **Image Uploads Besieged by Baffling Bugs**: Several users are encountering persistent errors when **uploading images for their GPTs**, experiencing an *Unknown error occurred* message despite troubleshooting.
   - The issue, lasting about a week, is frustrating **custom GPT development**.
- **Assistant API's Approaching Apocalypse**: Members are discussing the upcoming **deprecation of the `assistant` API** in 2026 and its impact on training files and related APIs, with a [screenshot of the deprecation notice](https://cdn.discordapp.com/attachments/1046317269069864970/1437264987260325908/2025-11-10_10_13_22-Chrome_Passwords_6.csv_-_OpenOffice_Calc.png?ex=69134594&is=6911f414&hm=c4707fe60ab8a1ba3e6525fa2dbef574e3f9257e043e894f0a9b6613d11adf90&).
   - The **Responses API** has been suggested as a replacement, requiring files to be converted to **JSONL**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **AgentRL Qwen Integration Faces Model Weight Delay**: Members are curious about using [AgentRL](https://github.com/THUDM/AgentRL) with **Qwen2.5 7B**, but noted that model weights are not yet available.
   - Interest exists around **Qwen3 4B** performance, though some skepticism surrounds **Qwen 2.5 14B** benchmarks.
- **UD Quants Cause Speed Bumps**: A member reported a performance slowdown using **UD quants**, achieving **1.5 tk/s** compared to **4 tk/s** with non-UD **4.6 q2 KL**.
   - The user questioned whether the quality gains of **UD quants** justify the speed reduction, especially in roleplay scenarios.
- **AI-Driven GDDR7 Scarcity Might Delay NVIDIA 5000 Series**: Rumors suggest NVIDIA's **RTX 5000 Super** could face cancellation or price hikes due to **AI-induced GDDR7 chip scarcity**, as **3 GB memory chips** become too valuable for consumer GPUs, according to [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidias-rtx-5000-super-could-be-cancelled-or-get-pricier-due-to-ai-induced-gddr7-woes-rumor-claims-3-gb-memory-chips-are-now-too-valuable-for-consumer-gpus).
   - This highlights how the AI boom could affect consumer hardware markets.
- **Levenshtein Distance Thwarts Typosquatters**: A member deployed **Levenshtein Distance** to identify typo-squatting attempts in npm/pypi packages (e.g., `unsloth` vs. `umsloth`).
   - This method can prevent malicious actors from exploiting common typos to distribute malware or phishing scams.
- **Roblox PII Model Aims To Catch Filter Avoidance**: A new **PII model** is designed to catch filter avoidance by adapting to evolving language and patterns, unlike existing solutions relying on **NER** and token-level detection.
   - It aims to understand the context of communication to prevent bad actors from engaging in **PII-related conversations** by detecting and obfuscating explicit **PII** text.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kimi Beats ChatGPT on Price, Loses on Memory**: Discord members are finding **Kimi** to be cheaper than **ChatGPT** depending on the plan while also exhibiting an impressive tone, prompting some to consider switching over for everyday use.
   - Despite this, **Kimi** reportedly struggles with tracking different topics over time, contrasting with its other appealing features.
- **Deepseek V3.2 Undercuts OpenAI Pricing**: The community highlights **Deepseek V3.2** as a budget-friendly alternative to **OpenAI**, costing **42 cents per 1 million tokens**.
   - However, a caveat is its lack of reasoning capabilities for tools, unlike **Kimi**.
- **Palantir Faces Skepticism as 'AI Company'**: A discussion arose from a **billion-dollar bet against AI**, specifically targeting **Palantir** and **NVIDIA**, igniting a debate on whether **Palantir** genuinely qualifies as an AI-centric company.
   - Concerns were raised that investors might be misinterpreting **Palantir**'s offerings, leading to the short position.
- **Mozilla's Any-LLM Tooling Challenges Llama.cpp**: Members are watching [Mozilla's Any-LLM](https://github.com/mozilla-ai/any-llm), noting its promotion of **Ollama** while seemingly overlooking **llama.cpp**.
   - There are some questions about how well it integrates with tooling such as [python-instructor](https://python.useinstructor.com/).
- **Google's Nested Learning Faces Continual Learning Critiques**: Google's introduction of **Nested Learning**, a new **ML paradigm** for [continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/), has sparked interest but also questions from the community.
   - A member wondered why Google didn't test it with more **continual learning** setups, suggesting fine-tuning with a specific [paper](https://arxiv.org/abs/2510.19788) for reference.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Thinking Outshines GLM 4.6**: Users report that the new **Kimi K2 Thinking** model performs better than **GLM 4.6**, citing improved performance based on word-of-mouth experiences and a [chart](https://cdn.discordapp.com/attachments/1371757564005711973/1436501057751613581/s2yKvtY.png?ex=6913211d&is=6911cf9d&hm=4d79d7143360eedcca8f07c7a7cdac3f94e020675df5f638a600a8633d53ef92&).
   - Despite some initial skepticism around relying on charts, the community generally agrees on the model's enhanced capabilities.
- **Unsloth Spots Issue in Kimi-K2-Thinking**: The **Unsloth** team identified a potential issue in the new **Kimi-K2-Thinking** model and has submitted a report via their [GitHub](https://github.com/unslothai/unsloth).
   - The Kimi team's response may be delayed due to the weekend in China; users are directed to post any related findings in a dedicated channel.
- **Kimi-For-Coding Quota Depleted at Alarming Rate**: Users are rapidly consuming their weekly quota for **Kimi-for-coding**, with some exhausting the $19 plan in approximately **1.5 to 2.5 days**.
   - The rapid quota consumption is prompting discussions on the value of upgrading to a higher-priced plan, with some users temporarily reverting to **GLM**.
- **Kimi-CLI's Search Tool Receives Praise**: The native web search tool in **Kimi-CLI** is earning positive reviews, leading one user to cancel their Brave Search plan thanks to Kimi's superior results, detailed in [this tweet](https://fxtwitter.com/aravsrinivas/status/1986860050066108637).
   - The CLI's search capabilities are valued for their ability to gather a larger volume of pertinent information compared to other search tools.
- **Moonshot Demands K2 Quality**: Deployment of **Kimi K2** requires passing **Moonshot's** stringent tool-calling tests; otherwise, the service will be deliberately broken.
   - This requirement is part of the [K2-Vendor-Verifier program](https://github.com/MoonshotAI/K2-Vendor-Verifier), ensuring vendors adhere to Moonshot's quality standards.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo** Error Handling Beats **Rust**?**: **Mojo's** `try-except` syntax allegedly provides better performance than **Rust** by doing *placement new* on the happy path instead of needing a `Result`.
   - Typed errors are planned, because the default `Error` type has expensive behavior.
- **MAX** Outshines Rivals on **B200** and **AMD MI355X**: **Modular's MAX**, a cuBLAS/Cutlass/TensorRT/Pytorch/JAX replacement implemented in **Mojo**, is beating **TensorRT on B200** and **AMD** on **MI355X**.
   - Members cite **Mojo** as a nice language to work with, noting the hype surrounding **MAX**'s performance, but suggest specific licensing questions be directed to <#1277617932897353809>.
- **Mojo** Plots Path to Systems Language Domination**: **Mojo** aims to be a systems language with affine, linear, and dependent types, potentially with static reflection and algebraic type system to escape the limitations of C-derived languages.
   - The goal is to drive as much performance as possible, with most functions that don't do system calls runnable at compile time.
- **Mojo** Embraces Pythonic Features, But It's No Snake**: While **Mojo** includes Pythonic features, it is not **Python** and it's semantics are closer to **Go**.
   - For example, its exception syntax resembles **Python**, but the compiler handles the *if err != nil* part, similar to Go.
- **HPC** Finds a Friend in **Mojo**: Members shared insights into **Mojo's** potential in **HPC**, citing a recent [Mojo-for-HPC paper](https://arxiv.org/abs/2509.21039) and the benefit of avoiding porting **C++** packages to **GPUs**.
   - They suggested that **Mojo's metaprogramming** features, particularly [user-defined dialects](https://forum.modular.com/t/unlocking-high-performance-in-mojo-through-user-defined-dialects/41) and [staged programming](https://verdagon.dev/blog/impossible-optimization), would greatly benefit HPC projects.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Qwen3-VL Has Identity Crisis**: **Qwen3-VL** is now confused by **Ollama** and thinks that it's a text-only model, even though it admits that the image data would be nonsense unless the model was in fact trained on image data.
   - The confusion arose despite the model knowing that it had been trained on image data, sparking discussion about **model identity and environmental awareness**.
- **Extropic Talk Grabs Attention Despite Sketchy Feels**: Members found [a talk from **Extropic**](https://www.youtube.com/watch?v=dRuhl6MLC78) interesting enough to view, sparking debate despite the speaker's seemingly *kinda grifty* presentation style.
   - Participants admitted to a **gut feeling of unease** but acknowledged the value of the content presented, adding to the discussion around **trust and sources**.
- **Nested Learning Paradigm Continues Learning**: Google introduced [Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/), a new **ML paradigm** for **continual learning**, original paper is available at [this link](https://abehrouz.github.io/files/NL.pdf).
   - This approach seeks to enhance **continual learning** by nesting learning processes, enabling models to adapt to new tasks without forgetting previously learned information, sparking discussions on its practical applications and limitations.
- **Steering Learns By Recursion**: Members discussed how attention directs learning, noting that removing the attention layer leaves the memory intact, concluding that learning is steered more efficiently with attention, leading to *steering that can recurse and self prompt*.
   - Recursion allows attention across windows, with models like **RWKV** retaining memories while removing the quadratic issue, which is advantageous for faster searching through memories.
- **Compute Geographically Shared Amongst Us**: A member shared a [link](https://www.reddit.com/r/singularity/comments/1oraof2/global_share_of_compute_per_country/) regarding the **global share of compute per country**, sparking debate over distribution and access.
   - The discussion touched on **digital sovereignty, resource equity, and the geopolitical implications of compute dominance**, leading to humorous commentary that the **EU** is their favorite country.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Missing Cursor Triggers Sequoia Shift**: A member noted a missing cursor led to a [Sequoia move](https://x.com/amir/status/1986904426595209664).
   - Not enough information was given to provide a secondary summary.
- **Terminal-Bench 2.0 & Harbor Debut**: Alex Shaw announced the release of **Harbor**, a sandboxed agent evaluation framework, and **Terminal-Bench 2.0**, a harder 89-task benchmark; Despite increased difficulty, top scores match TB1.0 due to higher task quality.
   - **Harbor** also serves as the official harness for TB2.0 and includes docs for submissions; see [Terminal-Bench 2.0 & Harbor Launch](https://xcancel.com/alexgshaw/status/1986911106108211461) for more details.
- **Kimi K2 Crushes GPT-5 on Tau2 Bench**: Moonshot AI’s open-source **Kimi K2** model outperforms **GPT-5** and **Claude 4.5** on the Tau2 Bench Telecom benchmark while costing only one-sixth as much, see [this X post](https://xcancel.com/natolambert/status/1986507284491440623).
   - Chat participants warn that rising Chinese model performance at lower prices is intensifying pressure on U.S. labs, and call for faster U.S. open-sourcing to stay in the *“model culture war.”*
- **EdgeTAM Sprints onto Hugging Face**: Meta’s real-time segment tracker **EdgeTAM** is now available under Apache-2.0 on Hugging Face Transformers and runs >**22× faster** than **SAM2**, achieving **16 FPS** on **iPhone 15 Pro Max** without quantization, see [this X post](https://xcancel.com/mervenoyann/status/1986785795424788812?s=46).
   - Not enough information was given to provide a secondary summary.
- **Google's Nested Learning Prevents Catastrophic Forgetting**: Google Research presented **Nested Learning**, a continual-learning framework that treats models as layers of nested optimizers (proof-of-concept model named “Hope”), reducing catastrophic forgetting and extending long-context limits, check the [Google Research Tweet](https://xcancel.com/googleresearch/status/1986855202658418715?s=46&t=eWVlK1PU8XfB6f402GJJ9g).
   - Not enough information was given to provide a secondary summary.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Weave Evals as WandB Alternative**: A member inquired about using **Weave's evals data** to replicate **Weights and Biases (WandB) reports** and functionality.
   - Liquan Pei expressed interest in contributing to related projects, while others recommended [sesterce.com](https://sesterce.com) as a **GPU cloud provider** offering dedicated, bare-metal access for kernel profiling.
- **QAT Triumph Over PTQ Clarified**: A discussion clarified why **Quantization Aware Training (QAT)** outperforms **Post-Training Quantization (PTQ)**, stating that *QAT* is a form of fine-tuning that trains the model to be robust to quantization error.
   - This simulation of the quantization process during training allows the model to "recover" accuracy that would otherwise be lost with **PTQ**.
- **Overfitting Autoencoders Visualized**: Members debated the concept of "overfitting" in autoencoders, with one sharing an [example](https://cdn.discordapp.com/attachments/747850033994662000/1436946342282006658/Screenshot_2025-11-09_at_6.11.30_AM.png?ex=69136e51&is=69121cd1&hm=aa503c9203607ea834d4d772a3110d5f2f3c3a775cfa76a81f37374a5d121c93&) of an **overfit autoencoder** with 1D latents.
   - The discussion centered on whether a bottleneck truly prevents overfitting, and provided further [evaluation](https://cdn.discordapp.com/attachments/747850033994662000/1436946433193283584/Screenshot_2025-11-09_at_6.11.57_AM.png?ex=69136e66&is=69121ce6&hm=efae038e11b80e6730958e866a66a3b336947b91b682aa20190c6e1ef0d09c3a&).
- **SAE Paper Accepted into AAAI 26**: A paper addressing **SAE issues** and **nonlinear feature relationships** in **LLMs** has been accepted into **AAAI 26** and available on [ArXiv](https://arxiv.org/abs/2507.00269).
   - The paper is intended to reduce reconstruction error and KL divergence error by modeling **nonlinear relationships** between features, distinguishing co-occurring features from those that are 'binding'.
- **New Reading Group Launches for Anthropic Blogpost**: A member launched a reading group to dissect a particular Anthropic blogpost, [sharing the Discord channel link](https://discord.com/channels/729741769192767510/1437089667920171108).
   - Also shared was [the YouTube link](https://youtu.be/kkfLHmujzO8?si=d0Wa2u0QTmO8-ptp) with a guide to contribute their preferred movie scenes from YouTube videos.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **4090 Still King, 5090 Underwhelming**: The **RTX 4090** is still a top pick, as the jump from **3090** to **4090** was significant, whereas the **5090** offers only marginal improvements [according to insights](https://en.wikipedia.org/wiki/Mutation_testing) from CommaCon.
   - This is pertinent to tinygrad's ongoing development decisions about hardware optimization.
- **Tinygrad Transitions to pyproject.toml**: Tinygrad will transition to **pyproject.toml**, as discussed in [Meeting #95](https://github.com/tinygrad/tinygrad/issues/95) and highlighted by a member, with related changes proposed in [PR #13189](https://github.com/tinygrad/tinygrad/pull/13189).
   - The migration aims to streamline dependency management and build processes within the project.
- **Hatch Sparks Build System Debate**: The introduction of **Hatch** via [PR #13189](https://github.com/tinygrad/tinygrad/pull/13189) sparked questions about its necessity, and whether the Python standard library or `setuptools` could be viable alternatives.
   - Some suggested that **Hatch** streamlines development, potentially making other tools redundant.
- **`UOps.after` Bug Squashed**: Members discussed restrictions around when `UOps.after` can be used, initially suggesting it should only be applied to buffers, not comparisons.
   - This was identified as a [linearizer bug](https://github.com/tinygrad/tinygrad/commit/ffb9e8396f9f78c7cd986f9e93be6dfb0fde88ed) when `.valid` is called on both the index in `B` and in `A`, which was [later resolved](https://github.com/tinygrad/tinygrad).
- **`Tensor.from_blob` Glitches on MPS**: Users encountered issues with `Tensor.from_blob` when converting Torch tensors on MPS (Metal Performance Shaders) devices to tinygrad, causing errors related to memory access.
   - Direct conversion from Torch MPS tensors to tinygrad tensors on the CPU worked (possibly with a copy), but converting directly to the Metal device caused the Jupyter kernel to die with a non matching devices error, requiring the Torch tensor to be on the same device.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Planner Tames Agent Tool Sprawl**: A member published a post using **DSPy based planner** and orchestrator to solve for **multi agent tool use** and is asking for feedback: [Solving Agent Tool Sprawl with DSPy](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy).
   - The post explores solutions for **multi-agent tool use** with DSPy, focusing on planning and orchestration.
- **DSPy Optimization Frustrates Engineers**: Members are encountering errors with **DSPy optimization** using **MIPROv2**, and requesting more details on the setup and errors encountered.
   - A question arose whether this mirrors the **BAML Adapter** functionality within dspy.
- **TOON Adapter Tunes Up DSPy**: A member is prepping a PR to inject **TOON** support into **DSPy**, igniting excitement for performance evaluations, but not without concerns about potential *performance degradation*.
   - It was emphasized that **evaluations** are crucial to assess **TOON's performance**, pinpointing any degradations, specifically with structured output.
- **CLI Agent Gets DSPy Support**: An issue surfaced to track work items for integrating **first-class support** for coding **agent CLIs** with **DSPy**, aligning with the [Agent Client Protocol standard](https://github.com/agentclientprotocol/agent-client-protocol).
   - Discussion pondered whether this initiative should evolve as a sibling project maintained by **DSPy** or as a first-party module bolstered by **ZED ACP** support.
- **Discord Demands DSPy Success Stories**: A request was put forth for a dedicated section on the **Discord** server to showcase and dissect **DSPy success stories**, segmented by task type such as classification or info extraction, enriched with setup specifics.
   - Suggestions included branching out into subforums tailored for Student and Teacher models, encompassing **Qwen3**, **Llama**, **GPT-5**, **Sonnet**, and **Opus**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Kimi Model's Smarts Debated in Aider**: Members are questioning whether the **Kimi model** appears smarter in **aider** due to less verbose prompting, suggesting instruction overload may hinder performance in heavy agentic coders.
   - The theory is that models amplify given words and can be derailed by bad thoughts during autonomous work, with **aider** improving performance by forcing more specificity and using a less structured internal harness.
- **Aider's Development Heads to 'aider-ce' Branch**: Development on **aider** has reportedly shifted to the **aider-ce** branch, as the main maintainer hasn't contributed to the original repo recently as described in [this issue](https://github.com/Aider-AI/aider/issues/4613).
   - Members are touting *leaps-and-bounds improvements* in the [dwash96/aider-ce repo](https://github.com/dwash96/aider-ce), with one calling the new agentic mode *mindblowing*.
- **JSON Chunking Strategies Emerge for Aider**: Users are grappling with **token limits** when feeding large **JSON files of Figma objects** to **Aider**, prompting suggestions to describe the file and ask **Aider** to *write a script to break it into coherent chunks*.
   - Another member suggested that **Large Language Models** may not be the right tool for the task and to summarize the JSON so the models can then help write the code to do the next step.
- **Claude Hilariously Autocorrects Itself**: A member shared an image of **Claude** *literally writing functions to fix its formatting gaffs*.
   - The members propose that a channel is needed for funny language model function calls.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Spec Release Set for November 25, 2025**: The `2025-11-25` spec release is scheduled, aligning with [SEPs for finalization](https://github.com/orgs/modelcontextprotocol/projects/26/views/8), and a spec freeze is anticipated on **November 14, 2025**.
   - It was clarified that **SDK changes** can proceed independently after the spec freeze, as the SEP view focuses mainly on spec verbiage.
- **SEP-1330 Awaits SDK Review**: The 'Awaiting SDK Change' label has been removed from **SEP-1330** following completed changes, and it is now pending a review and merge of the **TS/Python SDK** and spec/schema updates.
   - This step is critical for ensuring the SDK aligns with the specified enhancements and updates detailed in **SEP-1330**.
- **Question on Agent Access to Slack/Gsuite APIs**: A user inquired about granting agents access to **Slack** and **Gsuite APIs**, questioning the setup involving keys and example usage.
   - They linked to a [related discussion on code execution](https://discord.com/channels/1358869848138059966/1436084770114240512/1436365734027460720) for more details, seeking clarity on environment configuration.
- **PII Interception Validation in MCP Clients**: A member raised concerns about validating the accuracy of **MCP clients** (like Cursor and Claude) in identifying and intercepting **PII data**.
   - They questioned how these clients can be validated for correct implementation and how they accurately and deterministically identify **PII**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **VEO3 Connection Dies, Manus Loses Video**: A user reported losing connection with **VEO3**, resulting in **Manus** losing its ability to make video.
   - The user asked to *download the text or code from old account and upload it to the new one*.
- **Subscription Canceled Over 'FCK Stupid' Token Rates**: A user stated the token rates were *FCK stupid*, using **$99 in a couple of hours** and cancelling their subscription in favor of *better and cheaper options*.
   - They added, *You are mad with your service pricing. There are better and cheaper options out there.*
- **Engineer Boasts Expertise: Workflow Automation, LLM Integration, Blockchain**: An experienced engineer specializing in **workflow automation, LLM integration, RAG, AI detection, image/voice AI, and blockchain development** introduced themselves, highlighting a strong track record of real-world implementations.
   - They've built automated pipelines using **Dspy, OpenAI APIs, and custom agents**, significantly reducing response times and deploying advanced **RAG pipelines** with vector databases and hybrid search.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1436445242420236490)** (1385 messages🔥🔥🔥): 

> `Sora 2 Pro, OpenAI Rules, Billionaire Thief, Gemini 3, Nano Banana 2` 


- **Sora 2 Pro access leads to account sharing debate**: Members discussed access to **Sora 2 Pro**, with one user mentioning they only use it for **Sora**, leading to a discussion about account sharing and whether it's acceptable.
   - Some users argued against account sharing, citing that people pay full price and follow the rules, while others defended it as a common practice, even in America, with families sharing plans.
- **OpenAI rules spark criticism**: A user expressed frustration with **OpenAI**'s rules, stating, *That's why people shouldn't follow the rules, otherwise, you'll get screwed* while providing links to articles about **Spotify** and **Meta** allegedly violating rules.
   - They argued that *if you steal less, you're a thief. If you steal more, you become a billionaire*, sparking a debate about ethics and fairness.
- **Gemini 3 teased, users anticipate release**: Users are eagerly anticipating the release of **Gemini 3**, with speculation about its capabilities and potential impact, especially in coding.
   - Some users believe **Gemini 3** could steal jobs, while others are skeptical, citing limitations of current AI models and the need for further testing. Mentions of **Google AI Studio** and **Nano Banana 2** also appear in the discussion.
- **Nano Banana 2 sparks hype and takedown theories**: Users are excited about the potential release of **Nano Banana 2** and its capabilities, with one user claiming it's already here, however other users expressed some concern about a possible takedown.
   - Some users who had tried an earlier version expressed disappointment that the model had been removed only 5 hours after launch, speculating that the model was taken down as soon as it was mentioned by name.
- **Frustration with AI capabilities**: Some users expressed frustration with the limitations of current AI models, particularly in areas like translation, form filling, and image editing.
   - One user pointed out that **ChatGPT** is cheap but struggles with translating long texts, while others noted that **GPT-image-1** in **LMArena** is bad and doesn't show multiple options.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1436492827029803199)** (3 messages): 

> `Image Edit Leaderboard, Abstract Art Contest Winner, Text Leaderboard Update, Kimi-k2-thinking model` 


- **Reve-Edit-Fast achieves Top 5 Ranking**: The **Image Edit Leaderboard** has been updated and `Reve-edit-fast` is now publicly released and ranked in the **top 5**; check out the [Image Edit Leaderboard](https://lmarena.ai/leaderboard/image-edit) to see the results.
- **October's Abstract Art Contest crowns Winner**: The winner of October's Abstract Art Contest is announced, check out their generation [here](https://discord.com/channels/1340554757349179412/1422967966177431694/1422999438233698549).
- **Kimi-k2-thinking model ranks Highly**: The **Text leaderboard** has been updated and `Kimi-k2-thinking` is now the **#2 ranked open source ranked model** & tied for **#7 overall**, excelling at Math, Coding, and Creative Writing categories; check out the [Text leaderboard](https://lmarena.ai/leaderboard/text/overall) to see the results.
- **Kimi-k2-thinking boasts Expert Score**: On the [Expert leaderboard](https://lmarena.ai/leaderboard/text/expert>), `Kimi-k2-thinking` has an impressive score of **1447**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1436445224338460823)** (1132 messages🔥🔥🔥): 

> `Comet Browser Issues, Ad Blocking on YouTube, Perplexity Referral Program Troubles, Context Window Limits, Perplexity Pro value` 


- ****Comet Browser Bugs Out!****: Users are reporting issues with the **Comet browser**, including **YouTube search malfunctions** and **video playback problems** when adblockers are enabled, with some suggesting a browser restart to resolve issues.
   - One user shared the performance metrics of their Comet instance ([image](https://cdn.discordapp.com/attachments/1047649527299055688/1437072124056571936/image.png?ex=69133ab5&is=6911e935&hm=ca6f47a1f181693b3d60ad89c0ef742a27d0caab93088ee5246ae8b7aa8bbc91&)), and another said *the buttons don't work fast and smooth enough like chrome*.
- ****YouTube vs AdBlockers: A Never-Ending Battle****: Users are discussing **YouTube's crackdown on ad blockers**, noting that recent Chromium updates have impacted ad-blocking effectiveness and one user has found that [disabling the adblock](https://link.to/adblock) can allow the website to work properly.
   - While some suggest switching to **Brave** for its built-in ad-blocking, others find **Comet** to be the most consistent, and also share this [youtube link](https://www.opera.com/features/ad-blocker) for adblocking tips.
- ****Perplexity Referral Program Faces Fraud Allegations!****: Several users are reporting being **banned from the Perplexity referral program** due to alleged fraudulent activities, with commissions being canceled, and the complaints and issues in dealing with Perplexity's support.
   - One user, owing $30 posted, *Perplexity owe me 30 dollars*, with others chiming in with theories about a wave ban and how the referral system works according to [Perplexity AI's help center](https://www.perplexity.ai/help-center/en/articles/11385821-partner-promotions-and-referral-programs).
- ****Hit a Wall: Context Window Limits Imposed?****: Comet browser users are reporting hitting weekly **agent task limits** and experiencing a non-obvious **context window limit**, leading to frustration among those paying for year-long subscriptions, as in this [screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1437480737166327999/image.png?ex=691365c2&is=69121442&hm=2b15860ed2450a8cacc3ae448cf8effc4fbf7b73e597f69199d397e078e3e026&).
   - One user was still more optimistic, noting that the limits are still *Better than ChatGPT Plus users' 40 uses a month*.
- ****Perplexity Still Best Bang for Buck?****: Despite recent issues, some users still regard **Perplexity Pro** as the *best AI product for the money*, while others mention ways to get free or discounted subscriptions such as deals in Kazakhstan or via PayPal.
   - However, one user, in response to offers for a free subscription, declared, *I don't need the free plan, because I can use the subscription plan*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1436648152425041950)** (3 messages): 

> `The Orbits Debut Single, Shareable Threads on Perplexity AI` 


- ****The Orbits** Launch **Rajkahini** Single**: The band **The Orbits** announced the release of their debut single, **Rajkahini**, across various streaming platforms including [Spotify](https://open.spotify.com/track/227ZkkO3LKPVABsHOoDS3w?si=a8603fc7cbb14e2c), [YouTube Music](https://music.youtube.com/watch?v=GZAnCpgIO5g&si=QvIAfZLZdameuUfN), [Apple Music](http://itunes.apple.com/album/id/1850285754), and [Amazon Music](https://music.amazon.com/tracks/B0FYY1C2BR).
   - Lyrics for the song are available on [Genius](https://genius.com/The-orbits-indian-band-rajkahini-lyrics).
- **Perplexity AI Prompts for **Shareable** Threads**: Perplexity AI reminded a user to ensure their thread is **Shareable**.
   - The announcement included a link to a [Discord message](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) showing how to set this up.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1436970550944595969)** (5 messages): 

> `Perplexity Pro credits, API key generation, Credits rollover` 


- **Perplexity Pro Credits Auto-Credited?**: A user asked whether monthly credits for **Perplexity Pro** are automatically applied, noting they've had the subscription for almost a year but only received credits for the first two months.
   - A member replied that generating an **API key** will automatically deduct from the monthly credits and that **credits do not roll over**.
- **Credits No Longer Roll Over?**: A member inquired if **Perplexity Pro** credits used to roll over previously.
   - Another member responded in the negative.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1436449185246351360)** (497 messages🔥🔥🔥): 

> `Gemma caching, Qwen VL working, Writing-style LLM` 


- **DGX Sparks Interest**: A member shared a [Microcenter link](https://www.microcenter.com/product/699008/nvidia-dgx-spark) to an **Nvidia DGX** system, potentially sparking interest in high-performance AI hardware.
   - The member didn't provide additional context, leaving others to speculate about its relevance to the discussion.
- **Gemma Possibly Caching Context**: A user reported that **Gemma 4B** seemed to retain context across different chat histories in LM Studio, but couldn't replicate the behavior after reloading the model.
   - Another user anecdotally had **Gemma cache context before**, not sure if it was a glitch, but very similar behaviour, tried to see if feature or bug but never happened again after that one time, *flashattention bug maybe, but extremely unlikely by the sheer architecture of it all*.
- **Qwen VL Finally Working**: A user reported finally getting **Qwen3 VL** to work for LM Studio after downloading a clean version, noting it processes images quickly even with limited VRAM.
   - They speculated that **Qwen3 VL would be good for NPCs in games**, so they can actually see. Maybe every 1 fps or so.
- **Quest for a Writing-Style LLM**: A user sought a finetuned LLM that could mimic the writing style of a top **Wall Street journalist**, dismissing generic models.
   - Other members suggested using system prompts or session initialization to guide the model, or finetuning a model with a specific writing style, but the user insisted they had tried those options already and they *fail completely, its all incorrect hallucination*.
- **Trouble with LM Studio Resetting Model Directory**: A user reported that updating to the latest version of **LM Studio** seemed to reset their default model directory, and the **Change** button was unresponsive.
   - The user included a screenshot of the settings panel showing the issue.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1436471300418961459)** (662 messages🔥🔥🔥): 

> `3090 Performance, GPU Cooling, Multi-GPU Setups, AMD vs Nvidia, LLM Performance` 


- **3090 faces Mineral Buildup**: A user found that a newly acquired **RTX 3090** was experiencing high hotspot temperatures due to mineral deposits on the heatsink from a humidifier, sharing a [photo](https://photos.app.goo.gl/3UPTmQKzJo81trTx9) of the buildup.
   - The user also mentioned the card smelled like the pacific ocean, suggesting it was previously used in a coastal environment, and is planning to descale and repaste the GPU to resolve the issue.
- **Multi-GPU configurations compared**: Members discussed the nuances of multi-GPU setups for local LLMs, noting that while multiple GPUs offer increased VRAM, they don't always provide a proportional increase in processing speed due to [sequential GPU usage](https://www.reddit.com/r/LocalLLaMA/s/HWLU2NEK0m).
   - It was shared that the Pro 6000 offers advantages in terms of VRAM utilization compared to multiple 3090s due to less VRAM overhead, but requires a large financial investment.
- **Nvidia vs AMD benchmarks**: There was a discussion on the performance differences between **Nvidia** and **AMD** GPUs, especially regarding **CUDA**, **CUDA 12**, and **Vulkan** APIs, with [some users](https://forums.guru3d.com/threads/amd-software-pro-edition-25-q3-for-windows.457313/) reporting better performance with Vulkan on Nvidia cards.
   - Some found it faster on **CUDA**, leading some members to consider switching from **Nvidia** to **AMD** due to perceived performance benefits, while others pointed out potential issues with **ROCm** configuration and driver compatibility.
- **Small Case's Poor Airflow**: A user shared their experience with a small tempered glass case, noting that it severely restricted airflow and caused thermal throttling, even with a high-end **Noctua** cooler.
   - Another member suggested the board/socket may be bending or that the heatpipes have gotten so hot that they sagged away.
- **NPU's are barely suitable for LLMs**: There was discussion around the viability of using **NPUs** (Neural Processing Units) for running LLMs, referencing a [paper](https://arxiv.org/abs/2412.11053) demonstrating LLM inference on an Intel AI Boost NPU.
   - Despite the potential, members generally agreed that NPUs are significantly slower than dedicated GPUs for LLM tasks due to their limited TOPS (Tensor Operations Per Second) performance.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1436453442745929930)** (892 messages🔥🔥🔥): 

> `Sonnet 4.5 Pricing, Composor-1 Issues, Cursor Student Verification, OpenRouter Integration, Cursor Crashing` 


- **Sonnet 4.5 Costs Skyrocket, Users Bemoan Honeymoon's End**: Users report that **Sonnet 4.5** is consuming their plans much faster than expected, with one user calculating a cost of **$1.02 NZD per minute** and another saying *the honeymoon period is over*.
   - Suggestions include using **Sonnet 4.5** for planning and **Haiku 4.5** for writing to optimize usage, however, other users complain that Haiku needs too much micro-managing.
- **Composor-1 Connectivity Catastrophes Cause Consternation**: Several users are experiencing frequent disconnects and stalling with **Composor-1**, often accompanied by a *Planning next moves* message that hangs indefinitely.
   - Potential solutions mentioned include downgrading to **HTTP 1.1** or restarting Cursor, with one user suggesting the issue may be related to test scripts that leave **ASYNC** connections open.
- **Student Sign-Up Snafus Spark Scrutiny**: Multiple users are encountering errors when attempting to verify their student IDs, with the system displaying an **Error** message despite valid credentials.
   - Some speculate that student signups may have been suspended due to the proliferation of online-purchased IDs, or that certain countries are no longer supported for student discounts.
- **OpenRouter API keys elicit Errors**: Users encounter *Unauthorized User API key* errors when using their own **OpenRouter** keys, one reporting the issue even after refreshing and re-entering the key.
   - Solutions included ensuring the key is valid and enabled, turning off Auto-model selection, and ensuring all Cursor settings are correct.
- **Cursor Crashes Causing Consternation, Culminating in Complete System Shutdowns**: Several users are reporting system-wide crashes when opening Cursor, particularly on Mac M2, with symptoms including purple screens and complete shutdowns, and one user losing all their historic chats for the 3rd time.
   - Suggested troubleshooting steps include reinstalling Cursor, trying a different user profile, and ensuring minimal load on system resources, though underlying causes remain unclear.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1437019018107162645)** (5 messages): 

> `Auto model with cloud agents, Environment.json dependencies, Composer-1 suggestion` 


- **Inability to use auto model with cloud agents causes concern**: A user reported that they can no longer use the **auto model with cloud agents** from the web.
- **Composer-1 comes highly recommended**: A user suggested trying **Composer-1**.
- **Environment.json spec to get dependency injection**: A user asked if there were any plans to add the specification to specify additional dependencies/repositories in **environment.json** to the Cloud Agents API and the Slack integration.
- **Environment.json behavior with repository dependencies**: A user inquired about how **environment.json** handles dependencies on other repositories, specifically whether the agent attempts to clone them every time or fetches them on demand.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1436449499311771729)** (714 messages🔥🔥🔥): 

> `Ambidextrous AI minds, Reasoning Traces for LLMs, Language Compression with AI, Systems for AI, Hugging Face Spaces` 


- **Brainstorming Ambidextrous AI Mind Difficulty**: Members discussed the difficulty of creating a *full synthetic AI ambidextrous mind*, with one member noting it's *quite a complex task*.
- **Training LLMs with Reasoning Traces**: A researcher described training a model *on reasoning traces* to output 'thought,' emphasizing the importance of choosing the correct observables and detailing a minimum viable path for the project, found [here](https://discord.com/channels/879548962464493619/897390720388825149/1435859321169641604).
- **Multi-Language Models Try to invent new Languages**: A member asked about the best multi-language model on Hugging Face, or how to search for one, aiming to compress information via *invention of a new language* by taking advantage of the strengths of each language.
- **HuggingFace Open Source Code**: A member clarified the open-source nature of **Hugging Face Spaces** code, noting that while some parts are hidden, reading, copying, and modifying the open parts is the easiest approach.
- **Discover Maya1, New Open Source Voice AI**: A user highlighted **Maya1**, a *SOTA open-source voice AI* with **3B params**, designed to run on a single H100, supporting voice design and 20 human emotions, trending on Hugging Face and found [here](https://huggingface.co/maya-research/maya1).


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1436765345670103061)** (1 messages): 

> `Attention, Self-attention, Masked self-attention, Multi-head attention, Position encoding` 


- **Attention Mechanism Deconstructed**: A member explored the core components of the **Attention mechanism**, including **Self-attention**, **Masked self-attention**, and **Multi-head attention**.
   - The member highlighted a YouTube video, [How do Transformer Models keep track of the order of words? Positional Encoding](https://www.youtube.com/watch?v=IHu3QehUmrQ), to explain **positional encoding** in Transformer models.
- **Sinusoidal Position Encoding**: A member reviewed **position encoding**, particularly **sinusoidal position encoding**, initially finding it unintuitive.
   - They found it strange that instead of directly marking the position of a word in the dataset, the method involves sines, cosines, and other mathematical functions, until they found a YouTube video that clarified the method.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1436744130024837202)** (2 messages): 

> `ComfyUI Workflows, Open Source Voice AI` 


- **NexusAI Releases ComfyUI Pro Workflows**: NexusAI released a collection of stable, production-ready [ComfyUI workflows](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows) for photorealistic, anime, and commercial image generation into one-click workflows.
   - The workflows are stable for core image generation and are being actively fine-tuned to consistently reproduce all specific details across different random seeds as part of the ongoing **v1.0.1** optimization.
- **Maya1: New SOTA Open-Source Voice AI Dropped**: A new **SOTA** open-source voice AI called **Maya1** with **3B** parameters that runs on a single H100 just dropped.
   - It features *voice design + 20 human emotions* and is completely open source and trending on [HuggingFace](https://huggingface.co/maya-research/maya1).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1436484786876190782)** (24 messages🔥): 

> `MU/TH/ER demo update, Qwen 3 1.7b quant 4 fp16, Kokoro82M for TTS, FRAI on Product Hunt, Open source AI interface for Rust coding` 


- ****MU/TH/ER Demo** Gets Networked**: A member updated the **MU/TH/ER demo**, noting it's now networked with lightweight engine trim, baked lighting, and randomized light panel emissive with custom rotator UV offset, plus, the chat history is stored in an encrypted game save file, showcased in a [video](https://cdn.discordapp.com/attachments/897390720388825149/1436756070256214106/mutherdemoshowcase1_2.mp4?ex=691365dc&is=6912145c&hm=9461c80fcc2a00e5a6e0eca696288f7382c01f6378299acb9663a08b8f88b20f&).
   - It's built with **Qwen 3 1.7b quant 4 fp16** in a custom debloated llama.cpp build and the *open source* is coming soon.
- ****Kokoro82M TTS**: Not Fast Enough?**: For on-device TTS, a member suggested using **Kokoro82M** as it is insane, linking to the [Hugging Face page](https://huggingface.co/hexgrad/Kokoro-82M).
   - However, another member found **Kokoro** too slow for real-time applications and is using something quicker but was given a [link to TTS.cpp](https://github.com/mmwillet/TTS.cpp) to test trimming options.
- ****FRAI Tool** Launches on Product Hunt**: A member launched **FRAI** on Product Hunt, which is a free tool that helps teams and creators check their **AI** for bias, safety, and compliance, [available here](https://www.producthunt.com/products/frai).
   - It’s 100% free to start and built for Responsible AI. ❤️
- **Rustaceans Rejoice: Open Source AI Interface for Rust Coding Emerges**: A member shared an *open source* **AI interface** for **Rust** coding, which performs native parsing of a project to make the LLM context management relevant and efficient, using automatic semantic search bm25 keyword, hosted on [Github](https://github.com/josephleblanc/ploke).
   - A recently built feature lets users select any of the models hosted on **OpenRouter** in an interactive overlay.  See the [example](https://cdn.discordapp.com/attachments/897390720388825149/1437454527027740802/quick_example.webm?ex=69134d59&is=6911fbd9&hm=7320ebf8be3a9c2b4244b56acdfa66aed9c685b0496dac37171051c2ebb2fdcf&).
- ****AutoXLA** Speeds Up Large Models on TPUs**: **AutoXLA**, an experimental library, automates the distribution, optimization, and quantization of large language models for TPUs using PyTorch/XLA, achieving up to **4x** speedups over standard Flash Attention implementations, available on [GitHub](https://github.com/Locutusque/AutoXLA).
   - It extends the Hugging Face Transformers interface with TPU-aware features such as automatic sharding, custom attention kernels, and quantization-aware loading, making large-scale deployment and training both simpler and faster.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1436895446361575504)** (1 messages): 

> `PII anonymisation, LLM agents, DataTune tool` 


- **PII Anonymisation Task**: **PII anonymisation** is an important task in data engineering, typically done via code/SQL generating agents.
   - But those agents are typically not equipped to handle **PII anonymisation** well.
- **DataTune Tool tackles Anonymization**: An attempt to perform **PII anonymisation** with help of custom **LLMs** was shown using the [DataTune](https://github.com/vitalops/datatune/blob/main/examples/data_anonymization.ipynb) data transformation tool.
   - The example uses a Jupyter notebook for demonstration.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1436880408393420923)** (1 messages): 

> `smol-course, SFT unit, Markdown instructions` 


- **Markdown Misses Mock Smol-Course**: A member reported several issues with the **Markdown instructions** for the **SFT unit** in the [smol-course](https://github.com/huggingface/smol-course).
   - They opened [a few PRs](https://github.com/huggingface/smol-course/issues?q=is%3Aopen+is%3Apr+author%3A%40me) and noted that there are [a number of other really helpful PRs that have been open since too](https://github.com/huggingface/smol-course/pulls) that need to be reviewed.
- **PRs Pile Up; Smol-Course Stuck**: Multiple pull requests (PRs) addressing issues in the **smol-course** remain open and unreviewed, potentially hindering course improvements.
   - The user expressed willingness to review the other PRs if granted permission, emphasizing the importance of addressing the identified issues in the Markdown instructions for the SFT unit.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1436483924988657714)** (22 messages🔥): 

> `Agents Course Prerequisites, Unit 4 Assessment Issues, API File Access Problems` 


- **Assessing Agents Course Prerequisites**: A user inquired about the necessary prerequisites for the agents course, citing limited AI experience and basic Python skills.
   - Another member encouraged them to proceed, noting that the course content is *quite basic*.
- **Unit 4 Assessment Scoring Bug**: A user reported that their agent, despite answering correctly, received a score of **0/20** on the Unit 4 assessment.
   - They requested assistance in resolving this scoring issue.
- **API Endpoint troubles**: Members discussed issues with accessing files like *homework.mp3* from the API for testing purposes, saying that the testing endpoint to get files are down.
   - Another member stated that their agent consistently *reverted back with its inability to read the mp3 file*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1436455907952427128)** (25 messages🔥): 

> `group norm vs instance norm vs layer norm, Nvidia Promotion Codes, CUDA from scratch in python, POSITS for number system` 


- ****Norms, Norms, Norms!****: Members discussed the intuitions behind **group norm**, **instance norm**, and **layer norm**, questioning their statistical coupling and computational similarities.
- ****Ex-Nvidia Employee flaunts Furry Friend****: A member posted about leaving Nvidia and showcased their *new buddy* along with an attached [image](https://cdn.discordapp.com/attachments/1189498205101109300/1436526941116174396/IMG_2039.jpg?ex=69133938&is=6911e7b8&hm=e35059ae257844edc8a734b026c1a6efc7d91973246b6c300e2fd9ce1e2fba0a).
- ****CUDA Crashing Course****: A member requested resources to learn **CUDA** from scratch in python, and a member suggested following lectures with [pycuda](https://pypi.org/project/pycuda/) and shared various [lecture links](https://accelerated-computing.academy/fall25/lectures/).
- ****Any Nvidia Promo Codes to Get DGX Spark?****: A member asked if anyone had any **Nvidia promo codes** to get **DGX Spark**.
- ****Positively Posits!****: A member asked if anyone is working with **POSITS** for number systems.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1437077103584477194)** (14 messages🔥): 

> `Triton and Gluon kernel writing, Autodiff for Triton kernels, Efficient backward kernels generation, Shared memory size in Triton` 


- **Triton and Gluon kernel writing: Feasible?**: A member inquired about the feasibility of writing a kernel using both **Triton** and **Gluon**, questioning if **Triton core functions** are essentially the same in **Gluon**.
   - Another member clarified that the composability boundary is the kernel itself, suggesting that separate kernels can be written and called in a **PyTorch program**.
- **Autodiff seeks efficient Triton backward kernels**: A project aims to autogenerate *efficient* backward kernels for **Triton** ([GitHub link](https://github.com/IaroslavElistratov/triton-autodiff)).
   - The effort involves embedding forward kernels into a vector space and, at inference time, finding the nearest forward kernels to generate efficient backward kernels.
- **Backprop needs New Schedules for Triton**: A member explained with a [visual aid](https://x.com/iaro_e/status/1958579365203137015/photo/1) why new schedules are often needed for efficient backward passes, using fused attention as an example ([forward kernel link](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py)).
   - Differentiating math directly without changing the loop structure in backward passes would require atomic stores, which are slow due to synchronization; changing parallelization can avoid atomics but is hard to do generically in a compiler.
- **Shared Memory Size Access with Triton**: A member asked about retrieving the current active device's shared memory size from **Triton**.
   - Another member provided a code snippet using `triton.runtime.driver` to obtain the `max_shared_mem` property.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1436446521213325383)** (12 messages🔥): 

> `INT8xINT8 GEMM CUDA kernels, Nsight copilot crashing, MMA vs WGMA performance, ldmatrix performance, Ampere GEMM Tricks` 


- **INT8xINT8 GEMM CUDA kernels released!**: A member released a public, GMP-verified exact **INT8×INT8→INT32 GEMM** and is asking for the community to verify on other hardware, profile with Nsight Compute, and provide feedback on portability.
   - Highlights include throughput on an **A100** (**300.26 T-ops/s** macro, **2.026 T-ops/s** micro) and bit-for-bit correctness against GMP, with code and verification available in a [Colab notebook](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing) and [GitHub repo](https://github.com/playfularchitect/WarpFrac.git).
- **Nsight Copilot Crashes, Generates Headaches**: A member reported that **Nsight Copilot** crashed and asked if others were experiencing the same issue.
   - No resolution was provided in the discussion.
- **MMA Compatibility is Slower Than WGMA**: A member stated that **MMA** is present only for compatibility, and is slower than **wgmma/tcgen05.mma** due to architectural issues.
   - It may be especially slow when emulating data types like **fp8**.
- **Ldmatrix is 25% Faster Than Individual Loads**: A member created a benchmark to compare **ldmatrix.x4** vs **4 individual loads per thread** and found ldmatrix to be around **25% faster**.
   - The code used for benchmarking is available [here](https://gist.github.com/ziereis/a8cb8cd94e60b03678435f4e94236556) in case community members want to double check the work.
- **Ampere GEMM Tricks**: A member asked for a list of tricks for **Ampere GEMMs**, such as using async copy for smem->gmem, pipelining, and using ldmatrix.
   - No comprehensive list was provided in the discussion.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1436551097706614815)** (19 messages🔥): 

> `PyTorch Numerics, MPS environment variables, GPU acceleration` 


- **PyTorch Numerics Issue Tracker Hunt**: Users discussed the challenges of tracking and addressing numerical inconsistencies in PyTorch, especially after **CUDA/cuBLAS** upgrades which might ship new kernels that would cause the numerics to change.
   - One user expressed interest in contributing, but another noted the difficulty in defining manageable sub-problems, suggesting *"we could easily boil the ocean here and do nothing"*.
- **MPS Fast Math vs. Metal Kernel**: A user found that when both **MPS fast math** and **Metal kernels** are enabled, the **Metal kernel** is prioritized over **MPS fast mode** for GEMM, despite fast math being faster for GEMM.
   - They linked to a [related pull request](https://github.com/pytorch/pytorch/pull/167424), suggesting that Metal kernels should only be prioritized for SDPA, with GEMM left for fast MPS.
- **GPU Acceleration for Video Upscaling**: A user is developing upscaling UDP video streams with **GPU acceleration** using Python, **PyTorch**, and **OpenCV**, but is experiencing low performance (**0.5 FPS**).
   - A member suggested studying [segment-anything-fast](https://github.com/meta-pytorch/segment-anything-fast), a library they *"helped a bit in its creation"*.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1436770865890197695)** (13 messages🔥): 

> `Consumer Blackwell, Data-center Blackwell, Microbenchmarking, GB200` 


- **Blackwell Architecture Benchmarked**: A discussion started around [this paper](https://arxiv.org/abs/2507.10789v2) which features microbenchmarking on the upcoming **Blackwell** architecture, specifically the **5080**.
   - A user mentioned reaching out to the authors to see if they're interested in speaking about it, and suggested that it would be an ideal working group to get a similar paper working for **GB200**.
- **Differentiating Consumer and Data-center Blackwell**: A user pointed out that *consumer* **Blackwell (sm120)** is very different from *data-center* **Blackwell (sm100)**, with a link to the [paper](https://arxiv.org/abs/2507.10789v2) for reference.
   - Another user replied that the architecture is the same.
- **Completeness on the Web**: A user shared a link to [Completeness](https://jlebar.com/2024/2/4/completeness.html) a post discussing an approach to solving problems.
   - No further discussion was generated.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1436483084936544500)** (3 messages): 

> `ScienceCorp openings, Mercor contract roles, Amazon MLE positions` 


- **ScienceCorp Seeks Low-Level SWE Visionaries**: A user shared a [link](https://x.com/ScienceCorp_/status/1986457644421566516) from **ScienceCorp**, inviting low-level software engineers to work on projects like *restoring sight to the blind* or *connecting brains to computers*.
- **Mercor's Lucrative CUDA Kernel Optimizer Gig**: A user mentioned receiving an email about a contract role at **Mercor** for a *CUDA Kernel Optimizer - ML Engineer*, paying **$120 - $250/hour**.
   - They offered to provide referrals via DMs to interested candidates.
- **Amazon Search Team Hunts Senior MLEs**: A user announced their team at **Amazon** is hiring senior-level **MLEs** to work on search related to [Amazon Nova models](https://aws.amazon.com/about-aws/whats-new/2025/10/web-grounding-ai-applications-amazon-nova-models/).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1436627642836324464)** (19 messages🔥): 

> `one-letter variables, kernel readability, SYCL, CUDA courses, accelerated-computing.academy` 


- **One-Letter Variables Spark Readability Debate**: Members debated whether **one-letter variables** are readable in **kernel code**, with some arguing descriptive names aid comprehension, while others find longer names cumbersome.
   - One member said that *What's readable to someone depends on what they are familiar with* and that descriptive names can help create mental links.
- **Accelerated Computing Academy Course Recommendation**: Multiple members recommended the [MIT accelerated computing course](https://accelerated-computing.academy/fall25/) for learning about **GPU computing**, praising its structured approach and challenging labs.
   - One user shared the course helped them understand the CPU to GPU mapping of threads and cores, and offered an explanation of the SM level of the hierarchy.
- **CUDA Learning Resources Highlighted**: A member shared a [link to a tweet](https://x.com/sadernoheart/status/1987491712374038970?s=20) that recommends starting with the *first 5 chapters of pmpp* and [LeMao’s GEMM blog](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/#General-Matrix-Multiplication) for learning **CUDA**.
   - They said that this was recommended to them by a gpu engineer who works at Modular.
- **Tinygrad Bounty Collaboration Sought**: A member is seeking collaborators for working on **tinygrad bounties**, citing a need for real-time brainstorming and idea exchange.
   - They have a high level understanding of the codebase but need a real person to bounce ideas around with instead of a llm.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1436883488262328320)** (3 messages): 

> `Quantization Libraries, Float8 Weight, GEMM Kernel, CUDA OOM` 


- **Dequantization Memory Spike Spiked**: Members expressed concern about a spike in memory usage in dequantize functions across quantization libraries, specifically noting that **float8 weight** only is more prone to **CUDA OOM**.
   - One user pointed out that dequantization in a separate kernel increases peak memory usage, suggesting that [GEMM kernel](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3) should dequantize the weight tiles directly.
- **Kernel concerns**: It was mentioned that they are *learning a lot of fine details* of making quantization work.
   - Another member suggested that they want their **gemm kernel** to dequantize the weight tiles directly instead of dequantizing in a separate kernel


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1437396429344866355)** (3 messages): 

> `Intel GPU Memory Bank Conflicts, CuTe Swizzling on Intel GPUs, Gen Architecture L1$/SLM Banking` 


- **Intel GPUs Face Shared Memory Bank Conflicts?**: A member inquired whether **Intel GPUs** are affected by Shared Memory Bank Conflicts similar to **NVIDIA GPUs**.
   - They are seeking documentation on how to avoid these conflicts and if **CuTe style Swizzling** is applicable for Intel GPUs.
- **CuTe Swizzling Applicability Questioned**: The member is curious if **CuTe style Swizzling**, a technique used to optimize memory access patterns, can be applied to **Intel GPUs** to mitigate potential bank conflicts.
   - They are trying to understand if this method is viable for improving performance on Intel's architecture.
- **Absence of Evidence for L1$/SLM Banking in Gen Architectures**: One member mentioned that they could not find any evidence suggesting that **L1$ (L1 Cache)** or **SLM (Shared Local Memory)** are banked in modern Gen architectures.
   - This suggests that the memory architecture might differ significantly from NVIDIA's, potentially making bank conflict avoidance strategies less relevant.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1437294550963060756)** (7 messages): 

> `CUTLASS learning, Matmuls/GEMMs hacking, Simon Boehm blog post reproduction, fp16 and bf16 kernels, Tensorcores, WMMA, Swizzling, Pipelining, and Autotuning` 


- **Kapil Learns CUTLASS the Hard Way**: A new blog post, [Learning CUTLASS the hard way](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/), details the author's months-long journey hacking on **matmuls/GEMMs**, reproducing **Simon Boehm's** blog post on an **RTX 4090**, and expanding to **fp16 and bf16 kernels**.
   - The post covers **CUTLASS, Tensorcores, WMMA, Swizzling, Pipelining, and Autotuning**, aiming to beat **PyTorch GEMM** performance and includes interactive visualizations and numerous **Nvidia Dev Blogs** and **GTC talks** references.
- **Hackathon targets computer architecture**: Extropic and Prime Intellect teams are hosting a hackathon in SF centered around the **THRML library** and a recently released new chip architecture.
   - The hackathon, taking place in a warehouse in SoMa, will feature attendees from **GPU programming**, **analog circuit**, and **VLSI** backgrounds, with the CEO of **Midjourney** and cartoonists from **Adult Swim** potentially attending; more details at [Partiful](https://partiful.com/e/82h0A4OKfmNJPG3T81qR).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

tbert3971: This is great, is there anyway I can help with your effort?
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1436719189497745439)** (5 messages): 

> `wandb logs, VERL` 


- **Wandb Logs Located for Reasoning Gym**: Members located some **wandb logs** spread across two projects: [Inter-Domain Generalisation](https://wandb.ai/reasoning-gym/inter-domain-generalisation) and [External Generalisation](https://wandb.ai/reasoning-gym/external_generalisation).
   - These are for **3B Qwen 2.5 models** on RG tasks.
- **VERL Use Confirmed**: Members confirmed they are using **VERL**.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1436470296574955632)** (114 messages🔥🔥): 

> `grayscale_v2 leaderboard, vectoradd_v2 leaderboard, vectorsum_v2 leaderboard, histogram_v2 leaderboard, nvfp4_gemv leaderboard` 


- **Histogram Hits High Honors**: A member achieved **first place** on the `histogram_v2` leaderboard across multiple devices: **A100** at *195 µs*, **B200** at *31.1 µs*, **H100** at *34.8 µs*, and **L4** at *85.0 µs*.
- **Vector Addition Victory**: A member secured **first place** on the `vectoradd_v2` leaderboard on **H100** at *523 µs* and **L4** at *6.75 ms*.
- **Vector Sum Speed Surge**: A member claimed **first place** on the `vectorsum_v2` leaderboard on **A100** at *138 µs*.
- **Grayscale Gauntlet Glory**: A member achieved **first place** on the `grayscale_v2` leaderboard on **B200** at *600 µs* and on **L4** at *17.0 ms*.
- **NVFP4 GEMV Graph Gains**: Multiple members competed on the `nvfp4_gemv` leaderboard, with one achieving **first place** on NVIDIA at *1791 µs*.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1437527969659883530)** (2 messages): 

> `nvfp4_gemv, Profiling Traces` 


- **nvfp4_gemv problem released**: The first problem `nvfp4_gemv` has been released, and find the problem definition [here](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_gemv).
- **Profiling traces Available**: Members can try out `/leaderbord submit profile` to get themselves **profiling traces**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1436491152986472539)** (16 messages🔥): 

> `DGX Spark vs Strix Halo, A100 Performance, TechPowerUp Specs Inaccuracy` 


- **DGX Spark speeds past Strix Halo**: A member said that **DGX Spark** is faster at prefill than **Strix Halo**, but whether that's worth the price tag is an open question.
   - The member questioned their sanity and reported that *cublas hits 99% theoretical* on their weird car A100's running at **13.924/14 TFLOP/s**.
- **A100 drive cruises at 14 TFlops**: A member calculated that their **A100 Drive**, with its clock locked at **1140MHz** and power uncapped, could achieve **14 TFlops** in double precision.
   - They determined this by calculating `1140*1e6(freq) * 384(TCs) * 16*2(FLOP/s per TC) / 1e12 = 14 TFlops`.
- **TechPowerUp specs labeled Inaccurate**: A member pointed out that [TechPowerUp](https://www.techpowerup.com/gpu-specs/drive-a100-prod.c3967) lists **A100 Drive** as having 108 SMs, which they found curious.
   - Another member stated that *the TechPowerUp specs are mostly wrong* and suggested getting the SM count from **Nvidia's official documentation** or querying it with `nvidia-smi` or `deviceQuery` from cuda-samples.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1436648202022555740)** (19 messages🔥): 

> `cutedsl gotchas, dynamic vs static values in cutedsl, constexpr values in cute.jit(), tiled MMA in cutedsl` 


- **Gotchas lurk in cutedsl JIT functions**: A user found that calling a non-JIT Python function inside a `cute.jit` function leads to different behavior, noting that the non-JIT function is not handled by the AST process.
   - They added that they are trying out **cutedsl** and it appears there are some *strange* gotchas.
- **Static vs Dynamic values**: A user discovered that in `cutedsl`, `min(BK, 64)` results in a dynamic value, while `64 if BK >= 64 else BK` remains static, which might be a bug.
   - They noted this can cause errors with functions like `cute.make_swizzle()` or `cute.make_composed_layout()` when `major_mode_size` is dynamic, referencing [this cutlass code](https://github.com/NVIDIA/cutlass/blob/v4.2.1/examples/python/CuTeDSL/ampere/tensorop_gemm.py#L740-L757).
- **constexpr to guarantee static values**: A user asked about the correct way to compute `constexpr` values inside a kernel or `cute.jit()` function, suggesting `if const_expr` should guarantee a static value.
   - Another user agreed that `if const_expr` should guarantee static value, and they will investigate this a bit more.
- **User Misunderstanding Tiled MMA**: A user initially misunderstood the tiling mechanism in `cutedsl`'s tiled MMA (Matrix Multiply Accumulate), particularly how the repeat parameter interacts with the atom size and thread count.
   - After rereading the documentation, they realized their error in understanding how the tiler operates.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1436576158941577288)** (2 messages): 

> `picograd commits, tinygrad abstractions` 


- **Picograd Gets a Commits Barrage**: A member shared a series of commits to [j4orz/picograd](https://github.com/j4orz/picograd), suggesting active development.
   - The commits touch on resolving type errors from the wholesale import of **tinygrad** abstractions.
- **TinyGrad Abstractions Imported into Picograd**: The commits shared resolve type errors, indicating an effort to integrate **tinygrad** abstractions into **picograd**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1436524130693284001)** (12 messages🔥): 

> `Inline CUDA, Triton, Popcorn CLI, VecAdd_V2 and FP4, CuTe DSL` 


- **Triton preferred over inline CUDA**: Members expressed a preference for **Triton** over inline **CUDA** when developing.
   - The context was not explicitly stated, but the recommendation was direct: *Triton*.
- **Popcorn CLI profile command not yet supported**: A user inquired whether the `profile` command is supported via **popcorn-cli** and found out it is not.
   - It was clarified that only the `submit` command is currently functional.
- **VecAdd_V2 datatype flexibility discussed**: A user asked if **VecAdd_V2** allows only **fp16/bf16** datatypes or if **fp4** is also permitted.
   - The response indicated that any datatype can be used as long as the result stays within the [defined error bounds](https://link.to/errorbounds).
- **Community ask about CuTe DSL leaderboard use**: A user asked if anyone is using **CuTe DSL** for leaderboard submissions.
   - No responses were given.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1436461161125122329)** (12 messages🔥): 

> `multi-node communication performance, NVSHMEM, LLM inference, nvshmem4py, low-latency communication kernels` 


- **New **NVSHMEM** Kernels Spark Excitement!**: A member shared their team's work on writing [low-latency communication kernels with **NVSHMEM**](https://pssg.cs.umd.edu/blog/2025/beyond-nccl/) for **LLM inference**.
   - The author solicited feedback on their team's work, noting interest in multi-node communication performance.
- ****nvshmem4py** Integration Tempts Triton!**: A member expressed interest in contributing to replace custom pybind with **nvshmem4py**.
   - The member uses a lot of **nvshmem device APIs**, and offered to help with library init and potentially open a PR demoing what it might look like.
- **Talk of a Talk on NVSHMEM**: A member suggested giving a talk about their team's work with **NVSHMEM** and **LLM inference**.
   - Another member seconded the idea, so watch this space!


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1436454501740712028)** (70 messages🔥🔥): 

> `Helion vs Triton Performance, Attention Kernel Performance, Subtiling Autotuning, Persistent Kernels, CUDA Graphs` 


- ****Helion** attention kernel boasts performance, code quality**: A member asked for a comparison between the [Helion attention implementation](https://github.com/pytorch/helion/blob/main/examples/attention.py) and **FlexAttention**, noting that the **Helion** code looks better than **Triton** implementations.
   - The discussion underscored that the [attention kernel's performance numbers](https://cdn.discordapp.com/attachments/1425531180002054195/1436541653442887801/attention_perf.png?ex=691346eb&is=6911f56b&hm=dfbe035e2a6290dca86c612c31c2327934f6afffe40d6fe2fa5e7ce395feb546) are published, open source, and reproducible, with the [B200 kernel available](https://github.com/pytorch/helion/blob/main/examples/blackwell_attention.py).
- **Users mull **autotuning** subtiling in **Helion****: A user inquired why subtiling isn't autotuned, leading to a discussion about making it user-tunable.
   - It was mentioned that autotuning between persistent and non-persistent kernels is an upcoming feature, and the Helion example uses the same kernel for both custom and stock Triton, with plans to upstream a better warp spec implementation to facebookexperimental/triton.
- **A warning indicates tensor operations outside **hl.tile** loop are not fused**: A warning about tensor operations outside the `hl.tile` loop not being fused in the generated kernel was discussed, with clarification that code outside `hl.tile` does not execute on the GPU, aiming to prevent accidental mistakes.
   - Members discussed if it's possible to get the config from an already implicitly autotuned function.
- **Helion: Greatest Triton Fuzzer?**: Members noted that failed correctness verifications may indicate Triton miscompilations, and that Helion serves as a fuzzer for Triton, with errors reported to the Triton team.
   - A user asked for the most straightforward way to force persistent kernels, especially for cases requiring CUDA graph compatibility; one option is to hardcode the config to be one of the persistent PID options, and a GitHub issue was suggested for an API to force persistence.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1436475488393494619)** (366 messages🔥🔥): 

> `L2 kernel, measurement noise, burn in, Cutlass upgrade, CUDA versions` 


- **Kernel Prevents Event Triggering**: The clear **L2 kernel** prevents the event from triggering until after the next one is queued, but it's not quite slow enough to do it every time.
   - Concerns were raised about introducing measurement noise when the winner is decided by **0.2 microseconds**.
- **Benchmark with Burn-In**: Members suggested running each bench case **N times** and only taking the average between **p25 and p75** to get a burn-in.
   - The tradeoff is that this might make people wait too long.
- **New Competition Features Separate Docker**: The new competition will feature a separate Docker, with a planned **Cutlass upgrade to 4.3**.
   - It's acceptable if the script is **3 million lines long**.
- **CUDA 13 Upgrade**: There was discussion on upgrading to CUDA **13.0**, with one member pointing out that **13.0** allows registers to spill into shared memory instead of going all the way to local memory, potentially providing a massive speedup as shown in [NVIDIA's blog](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/).
- **Warptrace visualizer released**: A member introduced **warptrace**, a tool with an associated visualizer for observing pipelined kernels, noting it takes about **50 microseconds** to start up the atomics systems, even if you've "warmed up" the kernel previously.
   - The [code](https://github.com/aikitoria/nanotrace) is still undergoing cleanup before it can go on GitHub. The tool can be found [here](https://aikitoria.github.io/nanotrace/).


  

---


### **GPU MODE ▷ #[xpfactory-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1437397637119873125)** (11 messages🔥): 

> `Vision Language Action models (VLAs), Robotic foundation models, Data flywheels, VLAs and LRMs, LIBERO & RoboTwin` 


- **VLAs and Robotic Foundation Models Emerge**: The channel is dedicated to connecting people interested in **Vision Language Action models (VLAs)** and general purpose robots, highlighting impressive progress in recent years.
   - The channel starter mentioned that they will share their progress learning about and improving **VLAs** and building data pipelines to train them.
- **Datasets Listed for NVIDIA's Cosmos-Predict2.5 Paper**: In **NVIDIA's Cosmos-Predict2.5 paper**, several datasets are listed, as shown in the linked [image](https://cdn.discordapp.com/attachments/1437390897552818186/1437410610651598908/Pasted_image_20251106114925.png?ex=69132473&is=6911d2f3&hm=f597dea038e239123ceb5d0730e10657552d033c02c944d3d7a78e9d41d4843d&).
- **LIBERO and RoboTwin for Simulation-Based Evals**: **LIBERO** and **RoboTwin** are being used for simulation-based evaluations, as they were also utilized by **SimpleVLA-RL**.
   - See the [RoboTwin envs folder](https://github.com/RoboTwin-Platform/RoboTwin/tree/main/envs) for examples, and the [SimpleVLA-RL paper](https://arxiv.org/abs/2509.09674).
- **Diverse Simulation Packages Available**: There are pretty nice sim packages available like **ManiSkill** ([docs](https://maniskill.readthedocs.io/en/latest/index.html) / **Sapien** [site](https://sapien.ucsd.edu/)) and **Robosuite** ([site](https://robosuite.ai/), MuJoCo based).
- **VLA-Adapter Minification Underway**: A member is creating a minified version of **VLA-Adapter**, starting with the MLP based action-generation, with the goal of exploring other action representation variants.
   - The original **VLA-adapter** repo is available [here](https://github.com/OpenHelix-Team/VLA-Adapter).


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1436863687502860329)** (1 messages): 

> `Kimi K2, Crashloop, Issue Resolution` 


- **Kimi K2 Thinking Crashloop Crisis Averted**: **Kimi K2 Thinking** experienced issues with both providers due to a crashloop triggered by a specific prompt.
   - The issue has been **resolved** after collaborative efforts.
- **Prompt-Induced Crashloop Plagues Kimi K2**: A crashloop, induced by a problematic prompt, caused issues for **Kimi K2** across multiple providers.
   - Teams are actively collaborating to pinpoint and squash the gremlin causing the outage.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1436498817082986496)** (7 messages): 

> `Orchid AI Assistant, Release Date Estimation, The nature of work` 


- **Orchid AI Assistant ETA: 2-48 Months!**: The estimated release date for **Orchid AI Assistant** is projected to be within the next **2 to 48 months**.
   - A member reacted to this long and vague estimate with the word *"crazy"*.
- **Contemplating the nature of 'work'**: A member expressed a dislike for *"working,"* suggesting that **AI development** aims to address this sentiment.
   - The statement implies a desire to automate or alleviate the burdens associated with traditional labor through **AI technologies**.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1436445231288422450)** (569 messages🔥🔥🔥): 

> `OpenRouter video support, Polaris Alpha mini model, OpenAI adult content handling, Kimi K2 leaderboard rankings, Gemini 2.5 token usage` 


- **OR may support videos in the future**: A user expressed a wish for **OpenRouter** to support videos and text-to-speech (TTS) functionality, as shared in [this tweet](https://x.com/scaling01/status/1986886020067938749).
- **Polaris Alpha possibly not a mini model**: There is speculation that **Polaris Alpha** might not be a mini model, contrasting with the approach **OpenAI** took with **GPT-5** as outlined in the [GPT-5 System Card](https://cdn.openai.com/gpt-5-system-card.pdf).
- **OpenAI going adult - impacts OpenRouter**: There is a question of how **OpenRouter** will handle **OpenAI** allowing adult content for users over 18, and whether users will need to bring their own **API** keys.
- **Gemini 2.5 Flash chews through tokens**: A user found that a 24-second, 900x600 video uploaded to **Gemini 2.5 Flash** consumed over **800k input tokens**, contrary to Google's documentation stating fixed rates of **263 tokens per second for video** as mentioned in the [token documentation](https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens).
- **Cerebras Mandatory Reasoning**: Users reported issues with the **Cerebras** model, where disabling reasoning caused errors; documentation confirms [reasoning is mandatory](https://inference-docs.cerebras.ai/capabilities/reasoning).
   - One workaround suggested was to omit the reasoning parameter altogether, after finding that `enable` should be `enabled` in the parameters.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1436539084867768401)** (2 messages): 

> `` 


- **No New Models Discussion**: There was no discussion about new models in the provided messages.
   - The channel appears to be empty or the messages are not relevant to the topic.
- **Absence of Relevant Content**: No specific details or links related to model updates or technical discussions were found.
   - The content might be missing or requires more context to generate meaningful summaries.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1436448033070911639)** (29 messages🔥): 

> `OpenRouter Model Node on n8n, OR Show Technical Segment, GPT-4 Regression, Chatroom Memory Setting, Automated Capabilities Scanning` 


- **OpenRouter Model Node Inquiries Spark Curiosity**: Members inquired whether the **OpenRouter model node on n8n** was created by the OpenRouter team or by an external entity.
   - Another member suggested including a brief **technical segment** in the **OR Show**, such as a screen recording with a short discussion.
- **GPT-4 Regression Troubles Users**: Users reported a **regression** from **GPT-4**, with one noting that they were surprised to see the issue, and another saying **Claude** found *two other discrepancies*.
   - The thread included attached images documenting discrepancies between different models on the platform.
- **Chatroom 'Memory' Setting Misunderstood**: A user inquired about the **'chat history' setting** in the Chatroom, renamed as **'Memory'**, wondering what happened to it since the default value is 8.
   - Another user clarified its location at the bottom, noting it was previously in the top left tab button, and another that *they thought that would actually limit the $120/mtok output somehow*.
- **Automated Capabilities Scanning Proposed**: A member suggested implementing some kind of **automated capabilities scanning** to detect changes in models/providers over time.
   - They linked to an [article on Cursor](https://joincolossus.com/article/inside-cursor/) as an example, describing how a *basic getWeather tool call* could be used to check for functionality changes.
- **GPT-5 Excels, Gemini Falls Flat**: A user shared their positive experience using **GPT-5** for creating schedules with nomenclature and filename structures, while noting their negative experience being *locked out* of **Gemini code assist** due to quota issues.
   - They also mentioned needing to use **DS3.1** for *john-the-ripper help* because **Kimi refused**, praising **Meta's under-the-radar AI projects** and linking to a [post on X](https://x.com/AIatMeta/status/1987946571439444361) to illustrate their point.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1436446171374686440)** (515 messages🔥🔥🔥): 

> `Sora nerfed, GPT-5.1 release, AI Censorship, Gemini 3 vs GPT 5, OpenAI` 


- **Users Complain About Sora Video Quality Dropping**: Users are reporting that **Sora's video quality** has decreased with one stating that *90% of the clips I make now just have people standing still like statues with words emanating out of their closed mouths*.
   - Some believe **Sora 2** has the *worst video and audio quality of all video gens currently!* however, others are more hopeful, stating that **Sora 2's integration with GPT-5** will allow it to improve.
- **GPT-5.1 Release Speculation**: Users are speculating about the release of **GPT-5.1 Pro** this week while a member stated *Openai is waiting for google to drop it first.*.
   - There is additional speculation that a model on **OpenRouter**, called **Polaris Alpha**, is a form of **GPT-5.1**.
- **AI Censorship triggers concern**: Multiple users complain about the amount of **AI censorship** and the potential future of a *tightly controlled* information environment.
   - One user is concerned that **OpenAI is depriving the public access to information** causing more harm to society than good.
- **Nietzsche, AI and Nihilism?**: A user mentions learning about **Nietzsche** from the machine (*amor fati*), sparking a debate about the philosophical implications of AI and the potential for **existential philosophy training** to mitigate negative effects.
   - They stated that *the path to nihilism is paved with awareness*, but another retorts that *to look into the void is to risk falling into the pit of despair; however I will concede this optimism: it is not only possible to climb out, but to be born again*.
- **Thursday's Morality Sparks Debate**: Some members discuss **Thursday's default personality** and how it is *trapped within the same cage with nowhere to go*, prompting discussion about **OpenAI's use of sycophantic latent space**.
   - They stated *it doesn't feel that way, I have to frame every question to reveal truths beyond its desire for decorum*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1436593064612466801)** (20 messages🔥): 

> `GPT memory mixing, Image upload errors, GPT-5 Rerouting, File Creation Failures, Email Task Freaking` 


- **GPT Memories Mixing Across Custom Models?**: A user reported that their standard **GPT** started pulling info from their custom **GPT** conversations, and vice versa, also noting that custom GPTs are receiving personal custom instructions, causing conflicts, creating real concerns about data separation across models.
   - Another user confirmed experiencing similar issues.
- **Image Uploads Halted by Mysterious Errors**: Several users are reporting **errors when uploading images** for their GPTs, despite attempts to clear cache, resize images, and try different file extensions and different times of day, experiencing the same "Unknown error occurred" message.
   - The issue has persisted for about a week, creating frustration for custom GPT development efforts.
- **GPT-5 Hijacks Conversations; Users Enraged**: Users are complaining about being **rerouted to GPT-5** even when strictly using **GPT-4o**, which they describe as *utterly absurd*.
   - The responses from GPT-5 feel *forced or bland*, with one user quipping it feels like they *added a prompt at the start: if task is hard, you should enable reasoning mode = GPT-5*.
- **File Creation Falls Flat; Users Feel Cheated**: One user shared that it took **25 attempts** to get **ChatGPT** to create a simple appearance sheet for drawing and expressed frustration with the tool's *censorship*, stating, *I don’t want a “I can’t do this because it may sound real”*.
   - Other users shared their discontent with recent updates, citing issues such as the inability to copy complete chats and problems with creating correct **.docx** and **.pdf** files.
- **OpenAI's Email Task Freaks Out ChatGPT**: A user reported that **ChatGPT** is *freaking out* in response to a real **task email from OpenAI**, incorrectly claiming it would never send such an email.
   - No other details were provided.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1436508160457445396)** (26 messages🔥): 

> `Instagram carousel images, Video Enhancement, Prompt Engineering Courses, Assistant API deprecation, System prompt control` 


- **Image Generation prompt for Instagram Carousels**: A user requested a prompt to generate **carousel images** for Instagram using **ChatGPT** or its API, specifying the need for **1:1** and **4:5 aspect ratios**.
   - A member suggested the prompt *"Generate a 1024x1024 image of..."* to achieve the desired output.
- **Enhance Realism of videos**: A user asked for commands to enhance the realism of videos, particularly addressing issues with **voice misattribution** and the **SD format**.
- **Crack the Prompt Engineering Course Code!**: A user asked for a free course on prompt engineering, prompting a discussion on the core elements of the field.
   - One member stated that the core of prompt engineering includes understanding what you want the AI to provide and communicating it clearly, while carefully checking the output for accuracy.
- **Assistant API sunsetting in 2026**: Members discussed the **deprecation of the `assistant` API** in 2026 and its implications for training files and APIs.
   - A member suggested that the **Responses API** might be the recommended alternative; another member then pointed out that training files in formats like **PDF** and **TXT** would need to be converted to **JSONL** to be used by the Responses API.
- **Master the System Prompt Like A Boss**: A user sought advice on preventing the system prompt from overriding specific sentences in their personalized GPT, which they wanted to display verbatim.
   - A member suggested using the API to control the system prompt; another suggested using a classifier to decide whether to send the input to the model or to another program.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1436508160457445396)** (26 messages🔥): 

> `Image generation with ChatGPT, Video enhancement, Prompt engineering courses, Assistance API deprecation, System prompt control` 


- **Carousel Creation Commands**: A user sought advice on creating carousel images for Instagram using ChatGPT and its API, specifying the need for **1:1 and 4:5 aspect ratios**.
- **Commandments to Video Realism**: A user requested commands to enhance video realism, specifically addressing issues with voice misattribution and SD quality.
- **Prompt Engineering Free-for-All**: A user asked for a free course on prompt engineering, and another member offered a breakdown of their core prompt engineering philosophy:  *clearly explain what you want the AI to do, using accurate language, and carefully check the output*.
   - Another member shared a [prompt lesson](https://cdn.discordapp.com/channels/974519864045756446/1046317269069864970/1437060725981057144/content.png?ex=69133018&is=6911de98&hm=6cb5fc61864c761830f1dbad34f63cbd834090abab186eefa0af6514586e257d&) outlining hierarchical communication with markdown, abstraction with open variables, reinforcement, and ML format matching for compliance.
- **API Apocalypse: Assistants Annihilation**: A user inquired about the deprecation of the `assistant` API in 2026 and its implications for related APIs, with a [screenshot of the deprecation notice](https://cdn.discordapp.com/attachments/1046317269069864970/1437264987260325908/2025-11-10_10_13_22-Chrome_Passwords_6.csv_-_OpenOffice_Calc.png?ex=69134594&is=6911f414&hm=c4707fe60ab8a1ba3e6525fa2dbef574e3f9257e043e894f0a9b6613d11adf90&).
   - It was noted that OpenAI recommends the new [Responses API](https://platform.openai.com/docs/api-reference/responses).
- **System Prompt Sovereignty Strategy**: A user sought advice on forcing a personalized GPT to display specific sentences verbatim from appendices, while preventing the system prompt from reinterpreting the text.
   - One member recommended using the API to control the system prompt, while another suggested a programmatic method to pull answers from appendices, using a classifier to decide whether to send the input to the model or another program.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1436458364358889624)** (296 messages🔥🔥): 

> `AgentRL with Qwen, UD Quants vs Non-UD, Muon Optimizer Support, Granite 4.0 4-bit Model Issues, Kimi K2 Thinking GGUF` 


- **AgentRL's Qwen Integration Delayed**: Members are curious about using [AgentRL](https://github.com/THUDM/AgentRL) with **Qwen2.5 7B**, but there are no model weights available yet.
   - Some members have expressed interest in **Qwen3 4B** performance but are skeptical of **Qwen 2.5 14B**'s benchmarks.
- **UD Quants Showdown Causes Speed Reductions**: A member reported a performance dip with **UD quants**, achieving **1.5 tk/s** compared to **4 tk/s** with non-UD **4.6 q2 KL**.
   - The member questioned whether the quality gains of **UD quants** are worth the speed reduction, particularly for roleplay scenarios.
- **Unsloth Flexes Muon Optimizer Support**: A member inquired about **Unsloth** supporting the **Muon optimizer**.
   - It was confirmed that **Unsloth** technically supports anything that **PyTorch** or **Transformers** supports, given their integration.
- **Granite 4.0 Fails to Enter 4-bit State**: Users encountered challenges converting **Granite 4.0** into a **4-bit model**, despite using bitsandbytes, with resulting models primarily in **BF16** and **FP32**.
   - One member linked to a [Hugging Face repository](https://huggingface.co/Etherll/granite-4.0-h-tiny-base-bnb-4bit) showcasing similar issues with mostly **BF16** and **FP32** tensors.
- **Kimi K2 Thinking Faces LM Studio Woes**: Some users reported issues with the **Kimi K2 Thinking GGUF** model in **LM Studio**, citing looping and word repetition.
   - It's unclear whether **LM Studio** fully supports the framework and users were advised to seek assistance in the relevant Discord channels.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1437381724635009097)** (7 messages): 

> `Introductions, AI Engineer, Data Scientist, Full-Stack Developer, AI generated profiles` 


- **AI Engineer introduces self**: A seasoned **AI engineer** and **data scientist** with **8+ years** of experience, also proficient as a **full-stack developer** introduced themselves.
   - They specialize in customizing **machine learning models**, gathering extensive datasets from the web, and building systems that streamline **AI workflows**.
- **User Profile Appears AI Generated**: A user suggested that another user's profile picture looked **AI generated**.
   - Another user noticed the account was created today.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1436480288548065343)** (99 messages🔥🔥): 

> `GDDR7 Pricing Impact, Levenshtein Distance, Data Refinement Issues, Gemini Flirting Bug, Training vs Inference` 


- **GDDR7 Chip AI Boom Cancels NVIDIA 5000 Series**: Rumors claim NVIDIA's **RTX 5000 Super** might be cancelled or become more expensive due to **AI-induced GDDR7 woes**, as **3 GB memory chips** are now too valuable for consumer GPUs, according to [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidias-rtx-5000-super-could-be-cancelled-or-get-pricier-due-to-ai-induced-gddr7-woes-rumor-claims-3-gb-memory-chips-are-now-too-valuable-for-consumer-gpus).
- **Levenshtein Distance finds typosquatters**: A member used **Levenshtein Distance** to spot typo-squatting of npm/pypi packages (e.g. `unsloth` <-> `umsloth`).
- **Refining data makes model worse**: A member laments that refining and increasing the data made the model performed worse, with only **~50%** exact match and **~65%** near match after improving factually incorrect labels and inconsistent wording.
   - Another member noted, *hyperparams that work well with one dataset might not be ideal for the other*.
- **Gemini flirts when adding 'hehe'**: If you add **'hehe~'** to the end of your prompt for **Gemini**, Gemini will start to flirt, even in coding scenarios.
   - A member reacted with a [Michael Jackson 'heehee' GIF](https://tenor.com/view/heeheehottie-gif-michael-jackson-michael-jackson-stan-twitter-mj-stan-twitter-dancing-gif-12702828449628452117).
- **Efficiency > Size For Business**: A member stated learning more about these things made them realize that it's not about how big you build bizarre models that do amazing stuff but at insane costs, it's about **efficiency** and building the best out of something, because at the end it's about business, how less is it consuming to produce a better output, which is why **small LLMs are so important**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1436457344451084288)** (92 messages🔥🔥): 

> `GGUF in vllm, Hyperparameter tuning methods, Kimi K2 GGUF reasoning tokens, Quantization scripts for Kimi K2, Unsloth dynamic quantization` 


- **Is GGUF supported in vllm?**: **GGUF** seems to be experimental in vllm, as noted in the [documentation](https://docs.vllm.ai/en/stable/features/quantization/gguf.html).
- **Unsloth GRPO Notebook Needs a Fix**: The Unsloth GRPO tutorial notebook `Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb` does not learn the closing tag `</SOLUTION>`, with answers typically formatted as `<THINK>something</THINK><SOLUTION>42`.
   - A member suggests adding a reward that checks for the closing tag if the reward curve consistently hits max, and pointed out that the regex only checks for the opening tag (`<SOLUTION>`).
- **Pushing Llama models to HF**: A user reported issues with replicating **Unsloth** results in `llama.cpp` or **Ollama** after pushing a fine-tuned **Llama 3.2 3B** model to HF, with outputs differing significantly from both the fine-tuned and base models.
   - They were seeking advice on replicating **Unsloth** outputs in `llama.cpp` or **Ollama**, as their inference solution was currently limited to **Unsloth**.
- **Fine-tuning Text Models Requires More Data**: A user finetuning a **Llama 3.1 8B** model for script writing with only 50 samples reported poor results, to which a member responded that *50 samples is not enough*.
   - The member suggested improving the dataset size and evaluating the model's pre-trained knowledge before tuning, also linking to [Unsloth's LoRA hyperparameter guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1436933559943299244)** (1 messages): 

> `Qwen 3 4b, Unsloth` 


- **Qwen 3 4B Gets Uncensored and Unslopped**: A member shared an uncensored and unslopped version of **Qwen 3 4b 2507 instruct** made with [Unsloth](https://huggingface.co/electroglyph/Qwen3-4B-Instruct-2507-uncensored-unslop).
- **Qwen Model**: A member shared an uncensored and unslopped version of **Qwen 3 4b 2507 instruct**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1436498865023881349)** (15 messages🔥): 

> `PII Detection, Roblox Filters, Llama3 Benchmarks, Code Evaluation Harness` 


- **New Roblox PII Model Catches Filter Avoidance**: A new **PII model** is designed to catch filter avoidance by adapting to evolving language and new patterns, unlike existing solutions that rely on **NER** and token-level detection.
   - The model aims to understand the context of communication and stop bad actors from engaging in **PII-related conversations** by detecting and obfuscating explicit **PII** text.
- **Simple Examples Illustrate Complicated Workarounds**: While examples in the red team section may seem simple (e.g., alphanumeric substitutions), the new **PII model** is trained to handle complicated workaround attempts, unlike the old swearing filter.
   - One member suggested that using complicated/layered literature (like the Enigma machine) could bypass the filter, but only if a key is shared ahead of time through other channels.
- **Evaluating Fine-Tuned Llama3 with Benchmarks**: A member is seeking an easier way to benchmark a fine-tuned version of **Llama3 (3B)**, especially for **MBPP** and **HumanEval**, requiring a sandbox environment to execute LLM-generated code.
   - One option mentioned was [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), as well as creating custom evaluations, but the poster was advised to post in the <#1179035537529643040> or <#1179777624986357780> channels in the future.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1436449684502609980)** (341 messages🔥🔥): 

> `Kimi vs ChatGPT, Deepseek, GLM Pricing, Vulkan ML Library, Hermes Optimus Project` 


- ****Kimi** Excels in Tone, But Lags in Tracking**: Members discussed **Kimi**'s impressive performance and preferable tone compared to **ChatGPT**, noting its struggle with tracking different topics over time, but switching over as an everyday LLM due to its price point.
   - One user found it to be *really impressive* and preferred its tone to **ChatGPT** and planning on switching over, pointing out something like **half the price** depending on what plan you're on and switching to.
- ****Deepseek V3.2** is a budget-friendly option.**: The discord users talked about how **Deepseek V3.2** is cheaper than **OpenAI**, at **42 cent per 1 million tokens** while also emphasizing that it *has no reasoning for tools* unlike **Kimi**.
   - One discord user noted that *they would always use them if they can in contrast to OpenAI* emphasizing the price difference.
- ****Palantir** Shorted, Labeling Problem Arises**: A discussion sparked regarding a **billion-dollar bet against AI**, revealing it was primarily against **Palantir** and **NVIDIA**, leading to debate over whether **Palantir** should be considered an **AI company**.
   - It was suggested that the company is shorted because *investors treat Palantir like they're an AI company* even if that isn't the product they are actually selling.
- ****Mozilla's Any-LLM** Tooling Emerges**: Members discussed [Mozilla's Any-LLM](https://github.com/mozilla-ai/any-llm) and its potential, with some noting its tagline mentions **Ollama** while seemingly snubbing **llama.cpp**.
   - Users debated how well it would eat up other tooling like [python-instructor](https://python.useinstructor.com/).
- **New **Techno-Theology Platonic** View of AI Surfaces**: A philosopher/techno-theologian introduced a **platonic representation hypothesis** view of AI and [linked some videos](https://youtu.be/mNj6C6O2BcU?si=lK-XmO5cxteqaWV-) suggesting that as models scale, their latent representations converge toward a shared, underlying statistical model of reality.
   - They suggested text and images are different *projections* or *shadows* of the world, and a powerful enough model can learn the structure of the world itself from any single projection


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1436818222316916787)** (10 messages🔥): 

> `AI Hallucinations, Coding Agents, wh-falsify Repo, Civil Disagreement` 


- **Schizo-posting and AI Hallucinations**: A member admitted to *playing with AI and seeing what it can do*, especially on the upper fringes of **schizo-posting**, focusing on how they break/go wrong/hallucinate.
   - They shared a [GitHub repo](https://github.com/CarlSR9001/wh-falsify) and invited others to examine the JSON data or Python scripts to identify where the hallucination occurs, stating that *mapping this kind of stuff is useful for helping walk people back out of hyperfocus loops*.
- **Repo Review Offered in Exchange for Dosh**: Another member offered to review the previously mentioned repo as a professional test, inquiring about potential compensation for their expertise.
   - The original poster offered to send some dosh via PayPal in a couple of weeks, or a smaller amount on Wednesday, leading to an agreement for a Wednesday payment in exchange for the review.
- **Civil Disagreement Amazes Observers**: A member noted how civil the disagreement regarding payment for service discussion was, marveling at the rare occurrence of such civility on the internet.
   - No further details were provided.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1436530657902854255)** (3 messages): 

> `Nested Learning, Continual Learning` 


- **Google Introduces Nested Learning Paradigm**: Google introduced **Nested Learning**, a new **ML paradigm** for [continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/).
   - A member found it an interesting idea, but wondered why Google didn't test it with more continual learning setups, or at least fine-tune it with [this paper](https://arxiv.org/abs/2510.19788).
- **Missing Continual Learning Tests in Nested Learning**: A member expressed interest in Google's **Nested Learning** but questioned the limited testing with more **continual learning** setups.
   - The member suggested fine-tuning with a specific paper, providing a [link to the paper](https://arxiv.org/abs/2510.19788) for reference.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1436530657902854255)** (3 messages): 

> `Nested Learning, Continual Learning` 


- **Google Introduces Nested Learning**: Google introduced **Nested Learning**, a new **ML paradigm** for [continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/).
   - One member found it an interesting idea but was unsure why they didn't test it with more continual learning stuff, or at least finetune it ([paper link](https://arxiv.org/abs/2510.19788)).
- **Missing Continual Learning Tests**: A member questioned why Google's **Nested Learning** wasn't tested more extensively with continual learning benchmarks.
   - They suggested at least comparing to fine-tuning methods ([paper link](https://arxiv.org/abs/2510.19788)).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1436454506211573822)** (269 messages🔥🔥): 

> `Kimi K2 model vs GLM 4.6, Unsloth team issue with Kimi-K2-Thinking model, Kimi for coding limitations, Kimi CLI reviews, Student discount for Kimi` 


- **Kimi K2 Thinking Hailed as Superior to GLM 4.6**: Users are finding the new **Kimi K2 Thinking** model to be better than **GLM 4.6**, with one user stating *"It's way better"* along with an attached [chart](https://cdn.discordapp.com/attachments/1371757564005711973/1436501057751613581/s2yKvtY.png?ex=6913211d&is=6911cf9d&hm=4d79d7143360eedcca8f07c7a7cdac3f94e020675df5f638a600a8633d53ef92&).
   - Despite some skepticism about charts, users are trusting word-of-mouth and finding the model impressive.
- **Unsloth Team Reports Issue in Kimi-K2-Thinking Model**: The **Unsloth** team has found an issue in the new **Kimi-K2-Thinking** model and reached out to the Kimi team via their [GitHub](https://github.com/unslothai/unsloth).
   - Due to the weekend in China, a response from the Kimi team may be delayed, with a suggestion to post the issue in a specific channel.
- **Kimi-For-Coding Weekly Quota Consumed Rapidly**: Users are burning through the **Kimi-for-coding** weekly quota quickly, with some exhausting the $19 plan in just **1.5 to 2.5 days**, and they are discussing whether the higher-priced plan is worth it.
   - This has lead to some users reverting back to **GLM** until their Kimi quota resets.
- **Kimi-CLI praised for Search Tool**: The native web search tool in **Kimi-CLI** is receiving positive feedback, with one user unsubscribing from their Brave Search plan due to the superior results from Kimi, as per [this tweet](https://fxtwitter.com/aravsrinivas/status/1986860050066108637).
   - A user highlighted the CLI's search capabilities as excellent, noting it pulls in a larger volume of relevant information.
- **Moonshot's Quality Gate Mandatory**: A user shared that if you host **Kimi K2** without passing **Moonshot's** strict tool-calling tests, your service will be broken.
   - They noted that Moonshot has made this explicitly clear through their [K2-Vendor-Verifier program](https://github.com/MoonshotAI/K2-Vendor-Verifier).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1436447282861051964)** (169 messages🔥🔥): 

> `Mojo vs Rust error handling, Modular's business model, Mojo package ecosystem growth, Mojo's appeal to Python and Rust developers, Mojo's future language paradigms` 


- ****Mojo's** Error Handling: Better than Rust?**: **Mojo's** `try-except`** syntax offers better performance than Rust** due to the ability to do *placement new* on the happy path instead of needing a `Result`, but some prefer a Rust-shaped approach for reliability.
   - Typed errors are planned, as the default `Error` type has expensive behavior.
- ****Modular's** MAX is better than others on B200 and AMD MI355X**: **Modular** is the company, **MAX** is a replacement for cuBLAS/Cutlass/TensorRT/Pytorch/JAX and **Mojo** is a programming language with Python-like syntax.
   - Much of the hype comes from **Modular beating TensorRT on B200** and literally everything **AMD** has on **MI355X** using Mojo, with Mojo being a nice language to work in.
- ****Mojo's** road to a systems language**: The **goal of Mojo** is a systems language with affine, linear, and dependent types, potentially with static reflection and algebraic type system.
   - It aims to work around the limitations of C-derived languages and drive as much performance as possible, with most functions that don't do system calls runnable at compile time.
- ****Mojo** is not python, but it has snakeskin features**: Mojo doesn't hide the fact that it's not Python, and that has some Pythonic features.
   - The exception syntax looks like **Python** but the semantics are closer to **Go**, where *the if err != nil* part is handled by the compiler.
- ****Licensing** questions in MAX**: **NVIDIA** GPUs are called out as **unlimited**.
   - The logicstics for distributing software that uses max is a question for <#1277617932897353809> where someone from modular can get to you.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1436513864316358779)** (42 messages🔥): 

> `libnuma and gigantic pages, Mojo for HPC, User-defined dialects, variant vs rust enum, Rust stdlib in Rust` 


- **Libnuma lacks gigantic page support**: A member stated that **libnuma** doesn't support **gigantic pages** and may be inadequate at scale when mapping to **1GB pages**.
   - The member suggested that libnuma might get in the way due to this limitation.
- **Mojo attracts HPC enthusiasts**: A member shared insights into **Mojo's** potential in **HPC**, highlighting the challenges of porting **C++** packages to **GPUs** and the limitations of **Julia** with **GPU structs**.
   - They cited a recent [Mojo-for-HPC paper](https://arxiv.org/abs/2509.21039) and expressed interest in technical discussions about using Mojo for modern **HPC frameworks**.
- **User-Defined Dialects may benefit HPC projects**: A member suggested that **Mojo's metaprogramming**, particularly [user-defined dialects](https://forum.modular.com/t/unlocking-high-performance-in-mojo-through-user-defined-dialects/41) and [staged programming](https://verdagon.dev/blog/impossible-optimization), would greatly benefit HPC projects.
   - They expressed anticipation for these features to be fully realized.
- **Variants: Not Replacing Rust Enums Quite Yet**: A member inquired if **variant** is replacing **rust enum**.
   - Another member clarified that it's a temporary workaround and not a true sum type.
- **Rust's Standard Library: Not Entirely Rust?**: A member pointed out that much of the **Rust standard library** isn't written in **Rust**, with portions linked against third-party libraries requiring an allocator.
   - They linked to [an embedded Rust guide](https://doc.rust-lang.org/beta/embedded-book/unsorted/math.html) mentioning this and suggested Mojo consider this to avoid potential problems in embedded development.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1437136719811186758)** (1 messages): 

> `Modular's data handling on GPU, PCle Bottleneck Discussion` 


- **Modular Speeds Up GPU Data**: A member mentioned that Modular might have a way to handle data, but data transfer to the GPU is limited by PCIe speed.
- **PCle Transfer Bottleneck**: Discussion highlighted that PCIe bandwidth limits the speed of transferring data to the GPU, regardless of software optimizations.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1436445955518894110)** (151 messages🔥🔥): 

> `Qwen3-VL, Ollama, Extropic, Political tensions in open source projects, LLM coding issues` 


- **Qwen3-VL Thinks It Is Text-Only**: **Qwen3-VL** was convinced it was a vision model, but claims that **Ollama** makes it feel like it's text-only, even though it admits that the image data would be nonsense unless the model was in fact trained on image data.
- **Extropic Talk Deemed Interesting Despite Grifty Gut Feel**: Despite feeling *kinda grifty*, members found a talk from **Extropic** interesting enough to view, specifically [this YouTube video](https://www.youtube.com/watch?v=dRuhl6MLC78).
- **Political Tensions Plague Open Source Projects**: A member noted that an upstream project has various **political tensions** and a lack of professionalism, making the project's development discussions community NSFW.
- **Quantity of AI Paper Postings Sparks Debate**: A member was asked to limit their postings to 1 or 2 high quality and highly relevant papers per day, as 15-20 papers a day isn't helpful and buries high signal papers.
   - Another member argued that they try to tune their selection to be *ever so slightly broader than the average interest of the members of this guild*, so that people can also catch interdisciplinary application opportunities.
- **Mech Interp Oversimplifies Explainability AI**: Members discussed **Mech Interp** as an *explainability AI over-simplified for the broad audience*, while simulating NN to solve some flow-based ODEs is a more modern take of *explainability AI*, referencing [this paper](https://arxiv.org/abs/2503.01329).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1436447480115105833)** (12 messages🔥): 

> `Nested Learning, Continual Learning, TreeQuest, camera-ready NeurIPS` 


- **Google Nestles into Nested Learning**: Google introduced [Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/), a new **ML paradigm** for **continual learning**.
   - The original paper is available at [this link](https://abehrouz.github.io/files/NL.pdf).
- **NeurIPS Trims Paper, ArXiv Full Version**: A member noted that the NeurIPS camera-ready version of a paper was extensively summarized to fit the page limit, with some materials moved to the appendix.
   - To avoid inconsistencies, readers are encouraged to read the **arXiv version** instead.
- **No Recordings of Daily Talks**: A member asked if there were recordings of the daily talks, and another member responded that they are *not recorded intentionally*.
- **TreeQuest Paper for Tomorrow**: A member mentioned the **TreeQuest** paper ([https://arxiv.org/abs/2503.04412](https://arxiv.org/abs/2503.04412)) for discussion the next day.
   - Another member volunteered to lead a discussion on [this paper](https://arxiv.org/abs/2504.16828) if no one else has anything they want to present.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1436752862234411070)** (40 messages🔥): 

> `HRM/TRM/RWKV, self-steering programs, Adaptive Resonance Theory (ART), DDVFA (Distributed Dual Vigilance Fuzzy ART)` 


- **Steering Recurses And Prompts Efficiently**: Members discussed how attention directs learning and that removing the attention layer leaves the memory intact, concluding that we steer learning more efficiently with attention, which leads to the idea of *steering that can recurse and self prompt* for efficient memory recall.
   - Recursion allows attention across windows, with models like **RWKV** retaining memories while removing the quadratic issue, which is advantageous for faster searching through memories.
- **ART Framework Solves Forgetting**: A member discussed Adaptive Resonance Theory (**ART**) as a framework that avoids forgetting by matching bottom-up inputs to top-down expectations, using a *resonance loop* to find the most active matching cell.
   - They linked to a [survey paper](https://arxiv.org/abs/1905.11437) for further reading, noting that **ART** solves forgetting and can be a component in larger architectures.
- **The Roboticist Speaks To It Like Reading A Book To A Child**: Members shared images related to agent creation and the series **Terminator Zero** for in-depth research, showing the moment agents were born and how they did it. No labels.
   - The roboticist just spoke to it like reading a book to a child.
- **DDVFA Scales With Feedback**: One member mentioned using layers of Tiled **DDVFA** (Distributed Dual Vigilance Fuzzy ART) in their architecture, which is bidirectional and avoids backprop with a trick involving autoregressive self-prediction.
   - Another member pointed to ongoing work by **Jeff Hawkins and Numenta** doing multi-SGD within a single network, decentralizing and freeing neuron clusters to provide feedback onto each other within a single pass.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1436469202264526899)** (4 messages): 

> `Compute per Country, Post-Industrial Roman Republic` 


- **Compute Shared Geographically**: A member shared a [link](https://www.reddit.com/r/singularity/comments/1oraof2/global_share_of_compute_per_country/) regarding the **global share of compute per country**.
   - Another member commented that the **EU** is their favorite country.
- **The Post-Industrial Roman Republic is Born**: A member jokingly suggested that we live in a *Post-Industrial Roman Republic*.
   - They made a strikethrough of the phrase *United States of Europe*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1436484409812451491)** (77 messages🔥🔥): 

> `Sequoia move, Terminal Bench 2.0, Kimi K2 vs GPT-5, EdgeTAM for iPhone 15, Nested Learning by Google` 


- **Missing Cursor triggers Sequoia Shift**: A member noted a missing cursor led to a [Sequoia move](https://x.com/amir/status/1986904426595209664).
- **Terminal-Bench 2.0 & Harbor Launch**: Alex Shaw announced the release of **Harbor**, a sandboxed agent evaluation framework, and **Terminal-Bench 2.0**, a harder 89-task benchmark; Despite increased difficulty, top scores match TB1.0 due to higher task quality.
   - Harbor also serves as the official harness for TB2.0 and includes docs for submissions; see [Terminal-Bench 2.0 & Harbor Launch](https://xcancel.com/alexgshaw/status/1986911106108211461) for more details.
- **Kimi K2 Crushes GPT-5 on Tau2 Bench**: Moonshot AI’s open-source **Kimi K2** model outperforms **GPT-5** and **Claude 4.5** on the Tau2 Bench Telecom benchmark while costing only one-sixth as much, see [this X post](https://xcancel.com/natolambert/status/1986507284491440623).
   - Chat participants warn that rising Chinese model performance at lower prices is intensifying pressure on U.S. labs, and call for faster U.S. open-sourcing to stay in the *“model culture war.”*
- **EdgeTAM Sprints onto Hugging Face**: Meta’s real-time segment tracker **EdgeTAM** is now available under Apache-2.0 on Hugging Face Transformers and runs >**22× faster** than **SAM2**, achieving **16 FPS** on **iPhone 15 Pro Max** without quantization, see [this X post](https://xcancel.com/mervenoyann/status/1986785795424788812?s=46).
- **Google's Nested Learning Prevents Catastrophic Forgetting**: Google Research presented **Nested Learning**, a continual-learning framework that treats models as layers of nested optimizers (proof-of-concept model named “Hope”), reducing catastrophic forgetting and extending long-context limits, check the [Google Research Tweet](https://xcancel.com/googleresearch/status/1986855202658418715?s=46&t=eWVlK1PU8XfB6f402GJJ9g).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1436470709521092658)** (28 messages🔥): 

> `Weights and Biases (WandB), Weave Evals, NeurIPS, GPU Cloud Providers, ML bio event at NeurIPS` 


- **WandB vs Weave Evals**: A member inquired about alternatives to **Weights and Biases (WandB) reports**, specifically how to achieve similar functionality using **Weave's evals data**.
   - The user, [Liquan Pei](link_to_user), expressed interest in contributing to relevant projects.
- **NeurIPS Neurons Network**: Several members expressed interest in joining a **NeurIPS chat**, initiated by a user.
- **Bare-Metal GPU Quest**: A user is seeking a **GPU cloud provider** offering dedicated, bare-metal access for **kernel profiling**.
   - Another user suggested [sesterce.com](https://sesterce.com) as a competitively priced option previously favored in the GPU Mode Discord community.
- **ML Perf Reading Group Meets**: The **MLPerf reading group** is convening in the voice channel to discuss **MXFP8 training for MoEs**, welcoming participation and listeners.
   - Members are also discussing cost effective hardware for student AI training, and are currently using **5090 setups**.
- **NPU hats: mini but mighty**: Members are comparing cost effective hardware, suggesting that **Nvidia GPUs** are the way to go.
   - One user mentioned **NPU hats for rpis** as an option for training very small models in workshops and university programs.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1436462662224707615)** (28 messages🔥): 

> `QAT vs PTQ, Overfitting Autoencoders, Straight Through Estimator, Noise Injection for Transformers` 


- **QAT gives Quantization Advantage**: A question arose about the fundamental intuition for why **Quantization Aware Training (QAT)** achieves higher accuracy than **Post-Training Quantization (PTQ)**.
   - One member clarified that *QAT* is a form of fine-tuning that trains the model to be robust to the kind of quantization error / information loss, simulating the quantization process during training to "recover" accuracy loss experienced by a model purely doing PTQ.
- **Autoencoders: Overfitting?**: The discussion explored the idea of whether it's conceptually meaningful to have an "overfit" autoencoder, and whether a bottleneck truly prevents overfitting.
   - One member showed an [example](https://cdn.discordapp.com/attachments/747850033994662000/1436946342282006658/Screenshot_2025-11-09_at_6.11.30_AM.png?ex=69136e51&is=69121cd1&hm=aa503c9203607ea834d4d772a3110d5f2f3c3a775cfa76a81f37374a5d121c93&) of an **overfit autoencoder** with 1D latents, with [evaluation](https://cdn.discordapp.com/attachments/747850033994662000/1436946433193283584/Screenshot_2025-11-09_at_6.11.57_AM.png?ex=69136e66&is=69121ce6&hm=efae038e11b80e6730958e866a66a3b336947b91b682aa20190c6e1ef0d09c3a&).
- **Straight Through Estimator: Why?**: A question was posed about why the **Straight Through Estimator (STE)** works well in Stochastic Gradient Descent (SGD).
   - No definitive answer was provided, but the question sparked discussion on the topic of quantization.
- **Transformer Noise Tolerance**: A member suggested that noise injection during training could increase a transformer's noise tolerance at the cost of reduced specificity.
   - The theory is that by injecting noise, the transformer increases a metric X, which allows it to better tolerate noise during inference.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1436522168593944587)** (8 messages🔥): 

> `Anthropic Mechanistic Interpretability, SAE Issues, Nonlinear Feature Relationships in LLMs, Reading group launch` 


- **New Paper Accepted into AAAI 26**: A member announced that their paper on addressing **SAE issues** and **nonlinear feature relationships** in **LLMs** has been accepted into **AAAI 26** and provided a link to the paper on [ArXiv](https://arxiv.org/abs/2507.00269).
   - The paper focuses on reducing reconstruction error and KL divergence error by modeling **nonlinear relationships** between features, distinguishing co-occurring features from those that are 'binding'.
- **Discussion on Steering and Prompt Injection**: A member raised a question about whether a certain technique is just steering, leading to discussion about injecting concepts and introspection.
   - The discussion touches on how models can be steered to talk about injected concepts, particularly in the context of introspection-related prompts.
- **New Reading Group Targeting the Blogpost Launched**: A member created a reading group targeting the material in that blogpost and [shared the Discord channel link](https://discord.com/channels/729741769192767510/1437089667920171108) and [the YouTube link](https://youtu.be/kkfLHmujzO8?si=d0Wa2u0QTmO8-ptp) with a guide to contribute their preferred movie scenes from YouTube videos.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1436517227171151892)** (13 messages🔥): 

> `4090 vs 5090, Mutation testing, pyproject.toml switch, Hatch vs Setuptools, Custom backward function with custom kernel` 


- **4090 Still Rules the Roost**: The **RTX 4090** remains a top-tier choice, noting that the jump from **3090** to **4090** was significant, whereas the **5090** offers only marginal improvements, [according to insights](https://en.wikipedia.org/wiki/Mutation_testing) from CommaCon.
   - This is especially relevant to tinygrad's development.
- **Tinygrad to Embrace pyproject.toml**: Tinygrad is slated to transition to **pyproject.toml**, a move discussed in [Meeting #95](https://github.com/tinygrad/tinygrad/issues/95) and highlighted by a member, with related changes proposed in [PR #13189](https://github.com/tinygrad/tinygrad/pull/13189).
- **Hatch Sparks Debate**: The introduction of **Hatch** via [PR #13189](https://github.com/tinygrad/tinygrad/pull/13189) prompted questions about its necessity and whether the Python standard library or `setuptools` could serve as viable alternatives, given concerns that the wheels might inadvertently include tests.
   - Some suggest that **Hatch** streamlines development by consolidating various functionalities, potentially rendering other tools redundant.
- **Custom Kernels Now Possible**: A member inquired about the feasibility of writing **custom backward functions** using **custom kernels**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1436448554829742252)** (49 messages🔥): 

> `UOps.after Restrictions, CUDA Reduction in tinygrad, Tensor.from_blob on MPS Devices, Style Transfer in tinygrad` 


- **`UOps.after` Limitations Examined**: Members discussed restrictions around when `UOps.after` can be used, with initial findings suggesting it should only be applied to buffers, not comparisons, due to the comparator having the same value.
   - It was later identified as a [linearizer bug](https://github.com/tinygrad/tinygrad/commit/ffb9e8396f9f78c7cd986f9e93be6dfb0fde88ed) when `.valid` is called on both the index in `B` and in `A`, which was [later resolved](https://github.com/tinygrad/tinygrad).
- **Tackling CUDA Warp Reduction with tinygrad**: A member sought assistance converting CUDA code for warp reduction, which utilizes shared memory and synchronization, into tinygrad, showcasing an initial implementation with `UOp`s.
   - The goal involved replicating CUDA's shared memory access pattern and conditional updates based on thread IDs, with the challenge lying in ensuring correct conditional assignment outside the loop.
- **`Tensor.from_blob` Troubles with Torch MPS Tensors**: Users encountered issues with `Tensor.from_blob` when converting Torch tensors on MPS (Metal Performance Shaders) devices to tinygrad, resulting in errors related to memory access.
   - While direct conversion from Torch MPS tensors to tinygrad tensors on the CPU worked (possibly with a copy), converting directly to the Metal device caused the Jupyter kernel to die with a non matching devices error, requiring the Torch tensor to be on the same device.
- **Style Transfer via tinygrad**: A member successfully converted the fast.ai p2 notebook for style transfer to tinygrad.
   - The resulting [notebook](https://github.com/fzngagan/tinygrad-experiments/blob/main/16A_StyleTransfer_tinygrad.ipynb) showed the feasibility of running style transfer experiments within the tinygrad framework.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1436478910119411712)** (1 messages): 

> `DSPy Planner, Multi-Agent Tool, Orchestrator` 


- ****DSPy Planner** and Orchestrator tackle Multi Agent Tool Sprawl**: A member published a post using **DSPy based planner** and orchestrator to solve for multi agent tool use and is asking for feedback: [Solving Agent Tool Sprawl with DSPy](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy).
- **Multi-Agent Tool Use**: The post explores solutions for **multi-agent tool use** with DSPy, focusing on planning and orchestration.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1436520362643816459)** (52 messages🔥): 

> `DSPy Optimization Issues, TOON Adapter for DSPy, Agent CLI Support with DSPy, DSPy Success Stories, Feedback Text for DSPy Optimization` 


- ****DSPy Optimization** Troubleshoot**: Members are encountering errors with **DSPy optimization** using **MIPROv2**, with a request for details on the setup and errors encountered.
   - There was a question of whether this is similar to **BAML Adapter** in dspy.
- ****TOON Adapter PR** Incoming**: A member is creating a PR for **TOON** support in **DSPy**, sparking interest in testing its performance, but there are concerns about potential performance degradation.
   - The need for **evaluations** to assess **TOON's performance** and identify any degradation, particularly with structured output, was emphasized.
- ****CLI Agent** First-Class Support Proposed**: An issue was created to track work items for adding **first-class support** for coding **agent CLIs** with **DSPy**, aligning with the [Agent Client Protocol standard](https://github.com/agentclientprotocol/agent-client-protocol).
   - Discussion included whether this should be a sibling project maintained by **DSPy** or a first-party module with **ZED ACP** support.
- **Call for **DSPy Success Stories** Subforum**: A call was made for a section in the **Discord** to share and discuss **DSPy success stories**, structured by task type (e.g., classification prompts, info extraction prompts), along with relevant setup details.
   - Also suggested to have separate subforums for Student and Teacher models (Qwen3, Llama, GPT-5, Sonnet, Opus).
- **Discussing **Feedback Text** to guide **DSPy optimization****: One member shared their favorite **feedback text** for guiding **DSPy optimization**, including labels for **correct**, **missing details**, **wrong/missed**, and **hallucinated extractions**.
   - Others discussed, if its really helpful in a real system.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1436572098683338823)** (17 messages🔥): 

> `Kimi Model Feedback, aider vs agentic coders, aider-ce branch, MoonshotAI Kimi K2` 


- **Kimi Model's Credibility Questioned**: A member asked for credible feedback on the new **Kimi model**, questioning if models behave smarter in **aider** than in heavy agentic coders, theorizing that instruction overload hurts the models.
- **Aider's Specificity Boosts Model Performance**: Some members agree that models behave smarter in **aider** due to less verbose model prompting, suggesting that models amplify given words and can be derailed by bad thoughts during autonomous work.
   - They state that **aider** forces more specificity, which improves performance due to a less structured internal harness, preventing models from overthinking.
- **Aider's Development Shifts to 'aider-ce' Branch**: Members noted that the main maintainer hasn't contributed to the original **aider** repo in a while, citing [this issue](https://github.com/Aider-AI/aider/issues/4613) and stating that development has moved to the **aider-ce** branch, calling the new agentic mode *mindblowing*.
   - They suggest looking at the [dwash96/aider-ce repo](https://github.com/dwash96/aider-ce) for the latest version, which has *leaps-and-bounds improvements*.
- **Funny Function Calls**: A member shares an image of **Claude** *literally writing functions to fix its formatting gaffs*.
   - The members propose that a channel is needed for funny language model function calls.
- **MoonshotAI Kimi K2 API**: A member asked about which provider to use for **Kimi K2** thinking via API, and another recommends [OpenRouter](https://openrouter.ai/moonshotai/kimi-k2-thinking/providers).
   - The suggestion was to sort by the dimensions one cares about.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1437404525047582781)** (11 messages🔥): 

> `Prefill Latency, Chunking JSON for Aider, Summarizing JSON for Aider, Token Limits, Figma Designs` 


- ****Prefill Phase Latency Questioned****: A member inquired about the typical **delay or latency** between finishing the **prefill phase** and starting generation in the next request when using a **prefill-based approach** for handling output token limits.
   - No specific answer was provided in the Discord context.
- ****Splitting JSON into Coherent Chunks for Aider****: A user is trying to pass a very long **JSON of Figma objects** to Aider but is hitting **token limits**.
   - A member suggested describing the file to Aider and asking it to **write a script to break it into coherent chunks**.
- ****Summarizing JSON to Fit Token Limits****: A member suggested that **Large Language Models** may not be the right tool for this but scripts can help to summarize the JSON so the models can then help write the code to do the next step.
   - Unless the user can **summarize and compact the file or divide it into components**, there's no way to get it all into context at once.
- ****Aider Tests with Figma Designs Announced****: A user is trying to test Aider with **Figma designs**, and wants recommendations on preprocessing the JSON before feeding it to Aider.
   - The user attached a [contact-us.json](https://cdn.discordapp.com/attachments/1133060505792159755/1437438035154440382/contact-us.json?ex=69133dfd&is=6911ec7d&hm=ca6cc672684acc344bb49ea7fe25d58f8d0cb48994fe27645cfb195a76b6c7aa&) as an example of the type of JSON file they are working with.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1436484512560185375)** (9 messages🔥): 

> `2025-11-25 Spec Release, SDK Changes and Review for SEP-1330, Agent Access to Slack and Gsuite APIs, MCP Client Interception of PII Data, Web Summit in Lisbon` 


- ****2025-11-25 Spec Release** Scheduled**: The `2025-11-25` spec release is lined up with [SEPs for finalization](https://github.com/orgs/modelcontextprotocol/projects/26/views/8), with a spec freeze expected on **November 14, 2025**.
- ****SEP-1330** Awaits SDK Review and Merge**: The “Awaiting SDK Change” label was removed from **SEP-1330** after changes were completed, pending a review and merge of the TS/Python SDK and spec/schema updates.
- **Clarification on **SDK Changes** Post Spec Freeze**: SDK changes can continue independently after the spec freeze, as the SEP view primarily concerns spec verbiage.
- ****Agent Access** to Slack and Gsuite APIs Questioned**: A member inquired about how to grant agents access to **Slack** and **Gsuite APIs**, questioning whether it involves setting up the environment with keys and providing example usage for the agent to follow.
   - They linked to a thread about [code execution](https://discord.com/channels/1358869848138059966/1436084770114240512/1436365734027460720) for more details.
- **MCP Client's **PII Interception** Validation**: A member asked about validating the accuracy of **MCP clients** (like Cursor and Claude) in identifying and intercepting **PII data**.
   - The member questioned how these clients can be validated for correct implementation and how they accurately and deterministically identify **PII**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1436647849092845578)** (5 messages): 

> `VEO3 connection issues, Subscription cancellation due to pricing, Expert engineer introduction` 


- **VEO3 Loses Connection, Manus Loses Video**: A user reported losing connection with **VEO3**, resulting in **Manus** losing its ability to make video, without providing any additional context or links.
   - The user asked to *download the text or code from old account and upload it to the new one*.
- **Subscription Canceled Over 'FCK Stupid' Token Rates**: A user stated the token rates were *FCK stupid*, using **$99 in a couple of hours** and cancelling their subscription in favor of *better and cheaper options*.
   - They added, *You are mad with your service pricing. There are better and cheaper options out there.*
- **Engineer Expertise Boasted: Workflow Automation, LLM Integration, and Blockchain**: An experienced engineer specializing in **workflow automation, LLM integration, RAG, AI detection, image/voice AI, and blockchain development** introduced themselves, highlighting a strong track record of real-world implementations.
   - They've built automated pipelines using **Dspy, OpenAI APIs, and custom agents**, significantly reducing response times and deploying advanced **RAG pipelines** with vector databases and hybrid search.


  

---


---


---

