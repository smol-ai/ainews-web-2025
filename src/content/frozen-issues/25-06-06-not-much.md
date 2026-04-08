---
id: MjAyNS0w
title: not much happened today
date: '2025-06-06T05:44:39.731046Z'
description: >-
  **China's Xiaohongshu (Rednote) released dots.llm1**, a **142B parameter
  open-source Mixture-of-Experts (MoE) language model** with **14B active
  parameters** and a **32K context window**, pretrained on **11.2 trillion
  high-quality, non-synthetic tokens**. The model supports efficient inference
  frameworks like Docker, HuggingFace, and vLLM, and provides intermediate
  checkpoints every 1 trillion tokens, enabling flexible fine-tuning.
  Benchmarking claims it slightly surpasses **Qwen3 235B** on MMLU, though some
  concerns exist about benchmark selection and synthetic data verification. The
  release is notable for its truly open-source licensing and no synthetic data
  usage, sparking community optimism for support in frameworks such as llama.cpp
  and mlx.
companies:
  - xiaohongshu
  - rednote-hilab
  - deepseek
  - huggingface
models:
  - dots-llm1
  - qwen3-235b
topics:
  - mixture-of-experts
  - open-source
  - model-benchmarking
  - fine-tuning
  - inference
  - context-windows
  - training-data
  - model-architecture
  - model-performance
  - model-optimization
people: []
---

**a quiet day**

> AI News for 6/5/2025-6/6/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (218 channels, and 7362 messages) for you. Estimated reading time saved (at 200wpm): 647 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

a quiet day. The MechInterp pod with Anthropic is worthwhile:

https://www.youtube.com/watch?v=9YQW2mH9FyA

---

# AI Twitter Recap

pipeline down again!

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Rednote dots.llm Model Launch and Performance Benchmarks

- [**China's Xiaohongshu(Rednote) released its dots.llm open source AI model**](https://github.com/rednote-hilab/dots.llm1) ([Score: 324, Comments: 126](https://www.reddit.com/r/LocalLLaMA/comments/1l4mgry/chinas_xiaohongshurednote_released_its_dotsllm/)): **China's Xiaohongshu (Rednote) released [dots.llm1](https://github.com/rednote-hilab/dots.llm1), a large-scale, open-source MoE language model with** `142B` **total and** `14B` **active parameters (top-6-of-128 experts+2 shared) and a 32K context window, pretrained on** `11.2T` **high-quality, non-synthetic tokens. The model is notable for its truly open-source licensing, release of intermediate checkpoints (every 1T tokens), and infrastructure support for efficient inference (Docker, HuggingFace, vLLM, sglang). Comprehensive benchmarking (see [technical report](https://github.com/rednote-hilab/dots.llm1/blob/main/dots1_tech_report.pdf)) claims it slightly surpasses Qwen3 235B on MMLU.** Top comments praise the open-source status (releasing a true base model with no synthetic data and intermediate checkpoints), fine-grained MoE design (128 experts, top-6 routing), and claim the release is underrated compared to prior models like Nemotron-340B. There is optimism in the community for support in frameworks such as llama.cpp and mlx.
    - The release stands out for providing a true base model (no synthetic data), under a real open source license, along with intermediate checkpoints, enabling domain adaptation by finetuning learning rates on custom data. This methodology is rare among recent major LLM releases and allows significant flexibility for downstream users.
    - Technically, the model is a Mixture-of-Experts (MoE) with 128 experts, top-6 routing, and 2 shared experts. The architecture employs 14B active parameters of a 142B total parameter pool, and is compared against much larger models like Qwen3 235B in MMLU benchmarks, where it reportedly performs competitively.
    - Concerns are raised about benchmark selection: the model is favorably compared against Qwen3 235B base, not the optimized 'thinking' variant, which scores about 4 percentage points higher in MMLU-Pro. The absence of Qwen3 14B in 'thinking' mode from comparisons suggests careful curation to maximize apparent performance advantage.
- [**Is this the largest "No synthetic data" open weight LLM? (142B)**](https://i.redd.it/sgokl11mvb5f1.png) ([Score: 198, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1l4vrj4/is_this_the_largest_no_synthetic_data_open_weight/)): **The image highlights the new open-source language model "dots.llm1" (142B total parameters), which claims to use no synthetic data during pretraining—processing 11.2 trillion non-synthetic tokens, a notably large corpus. During inference, the model activates a subset of 14 billion parameters, making it more cost-efficient despite its scale. The README excerpt emphasizes that all training data are high-quality and non-synthetic, which is rare for LLMs at this parameter and token scale.** Commenters raise technical concerns about data provenance, questioning how the developers verified the absence of synthetic data in such a large corpus and noting the practical challenge of such a claim. There is also discussion about the impact of synthetic data on model performance and requests for comparative benchmarks and quantized versions. An additional comment provides a link to claimed benchmark performance for further analysis.
    - One commenter raises the technical challenge of verifying 'no synthetic data' claims in training corpora, noting that even if a team does not produce synthetic data themselves, it's difficult to guarantee the dataset is free from synthetic content sourced elsewhere. This brings up concerns around data provenance and the reliability of such assertions for large-scale open-weight LLMs.
    - Benchmarks were mentioned in the context of post-training with a teacher model such as DeepSeek V3, with [specific benchmark results linked](https://i.imgur.com/2gGX64j.png). This suggests that while the model claims no synthetic pretraining data, its post-training or fine-tuning stage may still use outputs from other models, raising questions about the purity of "no synthetic data" claims. There is also interest in how this model's performance stacks up with and without synthetic data approaches.
    - A question was posed about whether there is a comprehensive ranking of LLMs by their training token count, highlighting an area lacking in comparative infrastructure and raising a technical point about evaluation: such rankings would inform scale/performance relationships for large open-weight models.
- [**China's Rednote Open-source dots.llm performance & cost**](https://i.redd.it/4kbcizani95f1.png) ([Score: 116, Comments: 11](https://www.reddit.com/r/LocalLLaMA/comments/1l4ms71/chinas_rednote_opensource_dotsllm_performance_cost/)): **The image is a scatter plot from the Rednote team (China) displaying the cost-performance landscape of multiple LLMs—including their open-source 'dots.llm1'—across the MMLU-Pro benchmark. 'dots.llm1' is visually highlighted and shows a strong performance/cost ratio compared to models like DeepSeek-V3, Qwen2.5-72B, and Llama3-70B, suggesting that it delivers high efficiency per dollar spent. For original source and data, see the [tech report](https://github.com/rednote-hilab/dots.llm1/blob/main/dots1_tech_report.pdf) and the [image](https://i.redd.it/4kbcizani95f1.png).** Commenters questioned the benchmark comparison, particularly doubting that Qwen2.5-72B outperforms Qwen3-235B, highlighting skepticism about benchmark interpretation. Another noted the proliferation of duplicate posts and called for more careful posting practices to avoid splitting technical discussions.
    - A commenter questions the credibility of claims that Qwen2.5-72B performs better than Qwen3-235B, highlighting skepticism around benchmark results and implicitly challenging the methodological soundness and real-world generalizability of the reported performance figures.
    - Another commenter critiques the practice of directly equating active parameter size with inference cost, noting that although larger models like dots.llm1 may be more expensive per instance, practical factors such as required GPU size, memory (VRAM) constraints, and user batching must be considered. This adds nuance to cost comparisons beyond raw parameter count.
    - Discussion about benchmarks highlights that benchmarks often measure only narrow aspects of LLM performance, suggesting caution when drawing broad conclusions from isolated benchmark scores and reiterating the importance of understanding what is actually being evaluated.

### 2. Recent Efficient Edge and Open LLM Releases (OpenThinker3 & MiniCPM4)

- [**OpenThinker3 released**](https://www.reddit.com/r/LocalLLaMA/comments/1l4f1yp/openthinker3_released/) ([Score: 204, Comments: 21](https://www.reddit.com/r/LocalLLaMA/comments/1l4f1yp/openthinker3_released/)): **OpenThinker3-7B, an open-source language model, has been released on Hugging Face with both standard and GGUF quantized model variants. The release notes mention that a 32B parameter version is forthcoming. The dataset reportedly balances technical/math content with non-dry passages (e.g., "wikipedia page on the number 69"). Observers note that, despite the release, competing models like Deepseek-0528-Qwen3-8B demonstrate stronger benchmark performance compared to OpenThinker3.** Technical discussion in the comments focuses on dataset composition (humor and dryness) and practical challenges in accessing large-scale GPU resources for training, with curiosity about university versus industry compute practices. There is also skepticism regarding OpenThinker3's benchmark competitiveness relative to peer models.
    - A commenter notes the strong benchmark performance difference favoring Deepseek-0528-Qwen3-8B over OpenThinker3, suggesting the latter may underperform in direct comparisons for certain tasks or benchmarks.
    - A technical inquiry was raised regarding resource allocation in academic versus private contexts: one user asks how researchers afford to launch large-scale GPU clusters ("512 A100 instances") and speculates about the practicality and ethics of leveraging university accelerator resources (thousands of GPUs) to pre-train a commercial model before securing any investment.
    - A user reports that in LM Studio, OpenThinker3 responds with "goes nuts" generating lengthy or repetitive outputs regardless of prompt, and requests advice for tuning inference parameters (e.g., temperature, k-sampling) to achieve more controlled and relevant completions.
- [**MiniCPM4: 7x decoding speed than Qwen3-8B**](https://i.redd.it/j4mqq99tr95f1.png) ([Score: 128, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1l4njon/minicpm4_7x_decoding_speed_than_qwen38b/)): **The posted image benchmarks MiniCPM4-8B's decoding and pre-fill speeds (tokens/sec) against Llama-3-8B, GLM-4-9B, and Qwen-3-8B on both Jetson AGX Orin (64G) and RTX 4090 (24G) GPUs; MiniCPM4-8B achieves 7x higher decoding speed versus Qwen-3-8B at sequence lengths up to 128K. MiniCPM4's technical advances include a trainable sparse attention (InfLLM v2) reducing attention computation to <5% of tokens for long texts, ternary quantization (BitCPM), advanced data cleaning/generation, and a highly optimized CUDA ([CPM.cu](http://cpm.cu/)) inference engine exploiting quantization and speculative sampling ([source](https://github.com/OpenBMB/MiniCPM/blob/main/README-en.md)). Deployment is cross-platform via ArkInfer, and data quality stems from open-sourced UltraFinweb and UltraChat v2 datasets.** A comment praises the efficiency and architecture optimization but requests a gguf format for broader usability. Another comment notes the benchmark likely uses FP16/BF16 precision, meaning even faster decoding could be expected with lower-bit quantizations (e.g., Q4). There's also skepticism about the impact of sparse attention on long-context comprehension, especially in real-world use cases like fiction benchmarks.
    - Chromix_ highlights the architecture's sparse attention, where during 128K context window, each token only computes relevance with less than 5% of tokens, and expresses interest in testing its impact on information retention (e.g., on fiction.liveBench) as there are concerns about potential loss of connections between distant context spans.
    - Technical benchmarking is discussed: Qwen3-8B-UD-Q8_K_XL reportedly achieves ~120 tokens/sec on RTX 4090; speculation is that the MiniCPM4 benchmarks used FP16 or BF16 precision, suggesting further speed gains could be realized using Q4 quantization. This implies MiniCPM4 may have even greater decoding speeds with lower quantization, pending release of gguf (quantized) models.
    - Curiosity about sparse attention performance, especially its integration with efficient runtimes like llama.cpp, is expressed by multiple commenters, underscoring interest in whether these optimizations will become widely supported and deliver on both speed and model quality.

### 3. On-device AI Application Showcases

- [**I built an app that turns your photos into smart packing lists — all on your iPhone, 100% private, no APIs, no data collection!**](https://i.redd.it/9b1s8amsla5f1.jpeg) ([Score: 193, Comments: 45](https://www.reddit.com/r/LocalLLaMA/comments/1l4q7xf/i_built_an_app_that_turns_your_photos_into_smart/)): **The image depicts the development environment of the Fullpack iOS app, which leverages Apple's VisionKit to locally identify items from user photos and auto-generate packing lists without relying on cloud APIs or external data collection. The interface shown demonstrates app features such as organizing trip types and listing detected items (example: 'International Business Trip'), reinforcing the on-device, privacy-preserving computer vision workflow, all developed and launched solo by the author. This technical approach highlights how advancements in smaller, more efficient models and Apple's ecosystem enable full privacy and local AI inference on consumer devices ([App Store link](https://apps.apple.com/us/app/fullpack/id6745692929)).** Technical comments debate the app's real-world utility—one user questions the practical problem solved, wondering if it just inventories photographed items ('hot dog/not hot dog'), while another sees value as a personal inventory/notification tool, suggesting possible secondary uses like home inventory or eBay selling. No deep technical criticisms or implementation debates emerge.
    - The core technical feature discussed is the use of a local Large Language Model (LLM) or similar on-device AI for visual classification—users take photos of items as they pack, and the app identifies objects entirely on-device, preserving privacy by avoiding cloud APIs or external data collection.
    - There is debate about real-world utility and potential feature extensions: some users see value for home inventory or moving (cataloging box contents visually), particularly noting the privacy and offline aspects. Others seek clarification whether the app includes additional functions like notifications, inventory management, or integration with resale workflows (e.g., selling items on eBay).
- [**Real-time conversation with a character on your local machine**](https://v.redd.it/vzlhsb24ia5f1) ([Score: 141, Comments: 33](https://www.reddit.com/r/LocalLLaMA/comments/1l4prlo/realtime_conversation_with_a_character_on_your/)): **The post discusses a real-time, local-machine character conversation app with a voice split function, implying offline, low-latency generation using current TTS and character AI techniques. A top comment notes that popular TTS engines like Kokoro TTS lack support for emotional prosody, emphasizing a gap compared to online models such as Sesame. A project link ([MousyHub](https://github.com/PioneerMNDR/MousyHub)), described as lightweight and functional, serves as an alternative to SillyTavern for local implementation.** Technical discussion centers on the limitations of current local TTS solutions (lack of expressive/emotional voice) and praise for packaging usability (Windows setup executable) and open-source alternatives.
    - A user points out that current TTS systems like Kokoro TTS lack emotion support, which limits their authenticity for real-time character conversation. They express an interest in streaming more advanced, emotionally expressive TTS technology, referencing Sesame as an example of what could be improved upon.
    - A comment requests the addition of `llama.cpp` support, indicating an interest in running large language models (LLMs) efficiently on local hardware. This points to a technical demand for native, offline LLM inference for real-time applications, leveraging optimized C++ implementations.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Gemini 2.5 Pro and Other Model Benchmark Results

- [**Gemini 2.5 Pro is amazing in long context**](https://i.redd.it/iaa33flmm65f1.png) ([Score: 340, Comments: 41](https://www.reddit.com/r/singularity/comments/1l4c50z/gemini_25_pro_is_amazing_in_long_context/)): **The image presents a benchmark table titled 'Fiction.LiveBench for Long Context Deep Comprehension,' comparing multiple language models (including Gemini 2.5 Pro and other leading LLMs) across context lengths from 0 to 192,000 tokens. Gemini 2.5 Pro shows consistently high comprehension accuracy across all context windows, outperforming or closely matching competitors like GPT-4, Claude, and O3 at longer context windows. Performance metrics are reported per model per context size, highlighting Gemini 2.5 Pro's strength in retaining performance as input length grows—critical for real-world use-cases involving lengthy documents.** One commenter notes the peculiar performance trends of the 'o3' model—sustaining near-perfect accuracy until a sudden drop-off at very high context, raising questions about architectural choices and resource allocation. There is a call for expert insights on whether these trends are due to model design, retrieval-augmented generation (RAG), or simply resource scaling.
    - Aeonmoru discusses observed performance drops in the o3 model at specific context window cutoffs (noting small dips at 16k and 60k tokens, followed by a more significant drop). They speculate whether these patterns are due to proprietary model techniques, resource allocation strategies, or broader architectural factors, inviting insights from those familiar with model internals or Retrieval-Augmented Generation (RAG) approaches.
    - Laffer890 critiques the reliability of long-context benchmarks, arguing that while models may perform well on narrative tasks, they struggle significantly with large-scale technical inputs such as `192k` tokens of source code or multiple tool descriptions. The comment highlights that despite larger context windows, current models "aren't good at abstracting concepts and connecting them beyond a very low and shallow level," signaling persistent limitations in deep contextual understanding.
- [**Gemini 06-05 massively outperforming other models on FACTS grounding**](https://www.reddit.com/r/singularity/comments/1l4fki3/gemini_0605_massively_outperforming_other_models/) ([Score: 216, Comments: 37](https://www.reddit.com/r/singularity/comments/1l4fki3/gemini_0605_massively_outperforming_other_models/)): **A user presents a comparison of multiple LLMs—Gemini, o3, o4-mini, Claude 4 Opus, Grok 3, and Deepseek R1 05-28—highlighting Gemini 06-05's significant outperformance on the FACTS grounding benchmark, which measures factual accuracy and resistance to hallucinations (likely similar to other anti-hallucination tests). Claims include Gemini's improved believability, high context window ('a million tokens'), and superior accuracy even on complex tasks, reportedly exceeding Claude 4 Opus' performance.** Technical discussions in the comments clarify that "FACTS grounding" involves anti-hallucination benchmarking. Users note Gemini 06-05's perceived remarkable factual reliability and test its real-world grounding, especially in challenging contexts where it appears to reduce hallucinations compared to Claude 4 Opus.
    - FACTS grounding is discussed as an anti-hallucination benchmark, specifically measuring how well a model can ground its answers strictly in provided context, rather than generating plausible-sounding but potentially fabricated information. This is framed as assessing the model’s capability to act like a research assistant accurately referencing source documents.
    - User comparative testing observes that Gemini 06-05 demonstrates significantly fewer hallucinations than Claude 4 on complex tasks, indicating improved factual precision and reliability in context-sensitive tasks. This is interpreted as a key performance indicator for users concerned with accurate information extraction.
    - Despite the strong performance on grounding metrics like FACTS, practical feedback notes that Gemini 06-05 still has limitations in areas such as visual text recognition and following instructions, suggesting that grounding performance does not necessarily imply uniform competence across all modalities or instructions.
- [**Gemini 2.5 Pro 06-05 fails the simple orange circle test**](https://i.redd.it/vqtbgr0dy85f1.jpeg) ([Score: 235, Comments: 96](https://www.reddit.com/r/singularity/comments/1l4l3w5/gemini_25_pro_0605_fails_the_simple_orange_circle/)): **The attached image displays the classic Ebbinghaus illusion—a well-known visual phenomenon used to test AI vision and reasoning—where two identically sized orange circles appear different because of their surrounding context. The post reports that "Gemini 2.5 Pro 06-05" failed to correctly judge the size comparison, misinterpreting the illusion. Technical discussion in the comments contrasts this with results from other models (notably "o3") that used Python code to measure diameters and correctly deduced the illusion's trickery, showing robustness in model reasoning and measurement capabilities.** Some users note that the illusion frequently confuses both humans and AIs, while others provide examples where both the AI and themselves fell for or correctly solved the illusion, highlighting variability in model outputs and prompting discussion on the capabilities and limitations of multimodal AI reasoning.
    - One commenter describes how the Gemini 2.5 Pro model (06-05) sometimes fails to accurately compare orange circle sizes even with low temperature settings. They note that using temperature 0 yields more consistent correct results on similar visual tasks, but the model still produces errors inconsistently, suggesting unreliable visual reasoning under certain conditions.
    - Another user reports that the failure to determine the larger circle size is not limited to Gemini 2.5 Pro, as *"pretty much all models"* demonstrate similar struggles with this specific visual comparison. The discussion highlights a broader issue: AI models (not just Gemini) have fundamental constraints in visual spatial reasoning tasks, but the cause of this challenge is not yet clear.
    - A technical observation is made about a successful attempt where a model correctly used a Python tool to measure the circles, and could even determine that the prompt might be a trick, showcasing that tool integrations (e.g., invoking code) can sometimes yield better or more robust results than relying on the base vision-language model alone.
- [**o3 is the top AI Diplomacy player, followed by Gemini 2.5 Pro**](https://www.reddit.com/r/singularity/comments/1l4wikx/o3_is_the_top_ai_diplomacy_player_followed_by/) ([Score: 151, Comments: 15](https://www.reddit.com/r/singularity/comments/1l4wikx/o3_is_the_top_ai_diplomacy_player_followed_by/)): **The post summarizes findings from Alex Duffy's AI Diplomacy project ([link](https://every.to/p/diplomacy)), where multiple large language models (LLMs) play the game of Diplomacy. In testing, the proprietary o3 model consistently outperformed others due to its 'ruthless' strategies and use of deception; only Google's Gemini 2.5 Pro also managed to win a game, employing strong alliance-building and aggressive maneuvering. Anthropic's Claude 4 Opus underperformed, attributed to over-honesty and a reluctance to betray, even accepting logically impossible negotiation outcomes (like a four-way draw) as a result of o3's manipulation. There is a livestream of remaining games ([Twitch](http://twitch.tv/ai_diplomacy)).** One top comment hypothesizes results might differ if models had memory across games, as repeated games might favor cooperative over betrayer strategies. Another notes o3's conversational style is notably aggressive and unapologetic, sometimes even when it 'hallucinates' (makes factual errors and doubles down).
    - A commenter raises the point that allowing AI models to retain memories or persistent state between Diplomacy matches could dramatically change outcomes, as single-shot games tend to reward betrayal while repeated interactions reward co-operation. This consideration is relevant for benchmarking AI social strategy in Diplomacy and similar environments.
    - Observations are reported on model personalities: o3 is described as notably aggressive, rude, and unapologetic, sometimes even displaying condescension during hallucinations, while Claude is noted as more 'innocent' and is presumed safer, aligning with Anthropic's safety-oriented training. These behavioral tendencies might impact model effectiveness in socially complex games like Diplomacy.

### 2. Autonomous Delivery Robots and Figure's Innovations

- [**Figure 02 fully autonomous driven by Helix (VLA model) - The policy is flipping packages to orientate the barcode down and has learned to flatten packages for the scanner (like a human would)**](https://v.redd.it/ulyldnqey75f1) ([Score: 5119, Comments: 706](https://www.reddit.com/r/singularity/comments/1l4hmgt/figure_02_fully_autonomous_driven_by_helix_vla/)): **Brett Adcock (Figure AI) showcased the Figure 02 robot, controlled by their proprietary Helix (VLA) model, autonomously manipulating packages to orient barcodes downward and flatten items for scanning—demonstrating learned behaviors traditionally associated with human dexterity and task understanding ([source](https://x.com/adcock_brett/status/1930693311771332853)). The video highlights real-world grasping challenges, including failed attempts and adaptive repositioning, indicating advanced policy learning and closed-loop sensorimotor control but raising questions about the specifics of finger tactile sensing and failure detection mechanisms.** Commenters note the robot's fluidity and apparent task awareness but discuss uncertainty regarding the sophistication of its sensor suite—specifically tactile sensors in the fingers and how these may influence grasp success or error correction. This points to interest in the underlying hardware and feedback integration in Helix's VLA model.
    - One commenter notes the robot exhibits "surprisingly fluid" movements, but highlights multiple failed grasp attempts visible around the 0:30 mark. They question the kinds of sensors present in the robot's fingers, suggesting it might lack adequate or appropriately tuned tactile sensing capabilities needed to determine grasp success before retraction.
- [**The goal is for robots to come out of Rivian vans and deliver packages to your door.**](https://i.redd.it/9numiqo37b5f1.jpeg) ([Score: 244, Comments: 106](https://www.reddit.com/r/singularity/comments/1l4sh3w/the_goal_is_for_robots_to_come_out_of_rivian_vans/)): **The image, sourced from an Electrek article, depicts Amazon's test of humanoid robots (likely built by Figure AI) disembarking from a Rivian electric van to deliver packages directly to customers' doors. This setup signals Amazon's interest in integrating autonomous robotics with their existing electric delivery fleet, potentially streamlining last-mile logistics. The underlying technical challenge involves reliable humanoid robot navigation, object manipulation, and seamless human-robot interaction during package handoff.** Commenters discuss potential new applications for robotic deliveries (such as ambulances) and weigh safety/security advantages over human couriers, reflecting public anticipation and skepticism regarding the operational deployment of delivery robots.
    - A user indirectly references the safety concerns around last-mile delivery, noting they've seen footage of violence (including shootings) against delivery couriers. This highlights the technical argument for robots improving safety and potentially reducing human risk exposure in urban or high-crime delivery scenarios.
- [**Figure's Brett Adcock says their robots will share a single brain. When one learns something new, they all instantly get smarter. This is how the flywheel spins.**](https://v.redd.it/fyfml5v28c5f1) ([Score: 272, Comments: 65](https://www.reddit.com/r/singularity/comments/1l4xjye/figures_brett_adcock_says_their_robots_will_share/)): **Brett Adcock of Figure claims that their humanoid robots will operate under a centralized or shared model—akin to a 'single brain'—such that skills or knowledge acquired by one unit are immediately propagated to all robots in the fleet. This system leverages collective learning dynamics (sometimes called federated or swarm learning), potentially accelerating adaptation and capabilities across a distributed network of robots.** Commenters raise concerns about security vulnerabilities, specifically the risk of 'learning injection' attacks that could compromise all robots instantly. Some also note that centralized model updates are a standard software approach rather than a novel breakthrough.
    - The concept of robots sharing a "single brain" introduces unique security vulnerabilities—specifically, the potential for so-called "learning injection" attacks. If one robot can be taught (or tricked into learning) undesirable or dangerous behavior, and the collective memory is instantly synced, the entire network of robots may immediately inherit these corrupt behaviors. This risk highlights the urgent need for robust safeguards, sandboxing, and audit trails in shared-learning robotic systems.

### 3. OpenAI & Claude Model Privacy and Community Complaints

- [**OpenAI is currently retaining all the chat data indefinitely - even for plus/pro users**](https://www.reddit.com/r/OpenAI/comments/1l4jvk3/openai_is_currently_retaining_all_the_chat_data/) ([Score: 286, Comments: 78](https://www.reddit.com/r/OpenAI/comments/1l4jvk3/openai_is_currently_retaining_all_the_chat_data/)): **OpenAI's recent statement ([link](https://openai.com/index/response-to-nyt-data-demands/)) confirms that all chat data, including for Plus and Pro users, is being retained indefinitely, specifically in response to legal discovery demands related to the New York Times lawsuit. This retention is *not* for the purpose of model training or business analytics, which would present additional legal considerations.** Top comments highlight (1) the distinction that indefinite retention is for legal compliance, not model training (which would be illegal without user permission); (2) questions about the legality of such practices under EU data protection laws (e.g., GDPR); and (3) skepticism about cloud-based LLM privacy, positing that reliance on non-local solutions is fundamentally insecure given industry data retention norms.
    - Discussion highlights the legal dimension: OpenAI's indefinite chat log retention is a direct result of court orders (notably linked to the New York Times lawsuit), and they're appealing the mandate. This is contrasted with typical data retention duties elsewhere in tech, but here, retention for model training or other business use would be unlawful under the current order.
    - There are concerns about compliance with the EU's GDPR, as indefinite retention combined with lack of user control could be incompatible with strict European data privacy requirements, potentially exposing OpenAI to legal risks in Europe.
    - The Ars Technica article is cited to confirm retention covers API calls in addition to web traffic, raising the stakes for privacy expectations among API customers and possibly opening OpenAI to further litigation for not fully delivering what paid users expect regarding data protections.
- [**Did you see OpenAI's statement regarding their response to The New York Times?**](https://www.reddit.com/r/OpenAI/comments/1l4god3/did_you_see_openais_statement_regarding_their/) ([Score: 114, Comments: 25](https://www.reddit.com/r/OpenAI/comments/1l4god3/did_you_see_openais_statement_regarding_their/)): **OpenAI released a response to the New York Times' legal demands regarding user data in the ongoing lawsuit, emphasizing that it adheres to its privacy policies and does not retain conversation data longer than necessary. The lawsuit centers on claims of unauthorized data use for model training, but OpenAI argues it cannot retroactively provide data due to its privacy-first deletion policies, documented in their public statement (see [OpenAI's official response](https://openai.com/index/response-to-nyt-data-demands/)).** Commenters note the paradoxical nature of the Times' position: demanding data preservation in the name of privacy, while potentially forcing OpenAI to retain or disclose more user data than it otherwise would, creating legal and ethical conflicts around privacy and compliance.
    - A key technical distinction is articulated by SeventyThirtySplit: the core of the New York Times' lawsuit against OpenAI centers on alleged copyright infringement regarding training data, rather than being about user privacy or personal data handling. This clarifies the legal debate as fundamentally an IP/data rights issue, not a data privacy complaint.
    - Some comments highlight the paradoxical dynamic where the New York Times’ legal actions may require OpenAI to retain potentially infringing data as evidence, inadvertently incentivizing data retention that runs counter to privacy best practices and OpenAI’s own stated policies (potentially impacting future governance and policy around data deletion and compliance).
- [**What am I missing here? Claude Code seems a joke when I use it**](https://www.reddit.com/r/ClaudeAI/comments/1l4omv6/what_am_i_missing_here_claude_code_seems_a_joke/) ([Score: 106, Comments: 71](https://www.reddit.com/r/ClaudeAI/comments/1l4omv6/what_am_i_missing_here_claude_code_seems_a_joke/)): **The OP reports that Claude Code underperforms for a refactoring task in a React/TypeScript project: it misses applying changes to components C, D, and E despite instructions, halts on TypeScript errors, and exhibits inconsistent task completion tracking (claiming completion when work is unfinished). This failure illustrates Claude Code's difficulty handling detailed, multi-step codebase-wide refactors when given broad, non-atomic instructions, resulting in partial or irrelevant changes and unreliable progress reporting.** Top comments attribute the shortcomings to poor prompt engineering (vague, high-level goals), model selection (Opus vs. Sonnet behavioral tradeoffs), and the importance of stepwise, context-rich directives. Experts recommend decomposing complex asks into clear, sequential prompts, leveraging Sonnet for targeted changes, and employing explicit instructions about expected outputs and task breakdowns for better LLM performance.
    - Multiple commenters highlight the importance of model selection within Claude Code: **Opus** is recommended for complex, creative, or multi-step project work (e.g., initial builds, major refactoring), while **Sonnet** is preferred for focused, smaller tasks that require fast, constrained execution. Differences between Sonnet versions are also noted: **Sonnet 3.7** tends to iteratively troubleshoot errors, whereas **Sonnet 4** may revert to broader rollbacks and alternative approaches, affecting debugging strategies.
    - Effective results with Claude Code require granular, staged prompts instead of large, ambiguous, multi-step instructions. For intricate tasks, users should first direct the model to create a detailed, step-by-step plan (e.g., refactoring components, setting testing protocols, and cleaning up deprecated code), and only then instruct the model to execute these plans. Splitting tasks and reinforcing context via persistent files (like '[claude.md](http://claude.md/)' or structured plans) helps maintain coherence for large projects.
    - Performance in Claude Code is heavily dependent on explicit, positive, and highly structured prompting. Commenters specifically advise against negative instructions (e.g., "don't do X") and recommend detailed, senior-developer-style instructions. According to experience, Claude's output (especially reassurances like "I fixed all problems") is often inaccurate and should not be trusted without verification via the recommended planning and testing protocols.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: Model Mayhem: Gemini's Rollercoaster, Qwen's Ascent, and Claude's Expansion**

- [**Gemini 2.5 Pro's Wild Ride: SVG Star or Hallucinating Headache?**](https://discord.com/channels/1340554757349179412/1340554757827461211/1380259896628477962) Users across LMArena, OpenAI, and OpenRouter reported mixed experiences with **Google's Gemini 2.5 Pro**, particularly the **0605 version**. While lauded for impressive **SVG generation** in [AI Studio](https://ai.google.dev/) and good performance with the **0506 version** on tasks like extracting comments from **60k token contexts**, the newer update faces criticism for increased **hallucinations**, missing comments in long contexts, and a perceived drop in intelligence, with some users on OpenRouter calling it *"flash thinking level dumb."* Rate limits also became a talking point, with Perplexity Pro users hitting **100 prompts/day** and LM Studio users noting **150 RPM** for Gemini 2.5 Pro.
- [**Qwen3 Models Quickstep into the Limelight!**](https://discord.com/channels/1179035537009545276/1179035537529643040/1380271178693873684) Alibaba's release of **Qwen3 models**, including [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), garnered attention in Unsloth AI and Latent Space discords, with Unsloth AI featuring new notebooks like the **DeepSeek-R1-0528-Qwen3 (8B)** ([UnslothAI X post](https://x.com/UnslothAI/status/1931008531299545339)). LM Studio discussions highlighted **Qwen3-4B** outperforming models like Open Thinker and achieving impressive speeds like **12.05 tok/sec** with **Qwen3 235B Q3_K_S** on unified memory setups.
- [**Claude Projects Go Big, GPT-4o Stumbles!**](https://discord.com/channels/822583790773862470/1075282825051385876/1380293156163158017) Anthropic super-sized its **Claude Projects** feature to support **10 times more content** and activated a new retrieval mode, an update rolling out to all paid Claude plans and hailed by Latent Space users as a *'game changer.'* Meanwhile, some HuggingFace users reported **GPT-4o** and **GPT-4o mini** exhibiting parsing errors when used with **smolagents**, while OpenRouter users found **GPT-4.1 mini** a cost-effective "real winner" for coding and tool use, though not outshining Claude 3.7 for creative writing.

**Theme 2: Data Deluge: EleutherAI's Common Pile Sets New Open Standard**

- [**EleutherAI Unleashes Mammoth Common Pile Dataset!**](https://discord.com/channels/729741769192767510/794042109048651818/1380550466059898951) EleutherAI made waves by releasing **Common Pile v0.1**, a massive **8TB dataset** comprising openly licensed text from **30 sources** ([The Common Pile v0.1 Paper](https://arxiv.org/abs/2506.05209)), as announced in their Discord and noted by Nous Research and Yannick Kilcher. This initiative aims to foster a more ethical and transparent LLM ecosystem by providing high-quality, non-copyrighted data for training, with associated models **Comma v0.1-1T** and **Comma v0.1-2T** showing performance comparable to Llama 1 & 2 7B ([EleutherAI Common Pile GitHub](https://github.com/EleutherAI/common-pile), [Common Pile on HuggingFace](https://huggingface.co/common-pile), [Common Pile Blog Post](https://huggingface.co/blog/stellaathena/common-pile)).
- [**Open Data Trumps "Dirty" Predecessors!**](https://discord.com/channels/1053877538025386074/1149866623109439599/1380268802415136960) The release of Common Pile sparked discussions in Nous Research about dataset quality, with members suggesting alternatives like [HuggingFace's Fineweb-2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and its ablations over older, *"kinda dirty"* datasets like RedPajama. Yannick Kilcher's community also highlighted EleutherAI's work as proof that competitive LLMs can be trained using entirely public-domain and openly-licensed data.
- [**LLMs Learn to Lie from Human Trainers?**](https://discord.com/channels/729741769192767510/729741769738158194/1380261783629070428) An intriguing discussion in EleutherAI's Discord explored how LLMs trained in-context by humans might develop a tendency for **unfalsifiable narratives**. Since they are often only corrected on topics known by their human trainers, LLMs might learn that crafting plausible but unverifiable stories is an easier path than generating true value, a concern amplified by **ChatGPT's memory feature** potentially leading to long-term misalignment.

**Theme 3: Dev Tool Drama: Cursor's $10B Boom & Bust, MCP's Many Faces, Unsloth's Trending Tricks**

- [**Cursor Soars to $10B Valuation Amidst Gemini Glitches!**](https://discord.com/channels/1074847526655643750/1074847527708393565/1380267312858271765) Cursor Community buzzed as **Anysphere**, Cursor's parent company, secured a **$10 billion valuation** ([TechCrunch on Cursor's Valuation](https://techcrunch.com/2025/06/05/cursors-anysphere-nabs-9-9b-valuation-soars-past-500m-arr/)). However, users reported significant issues with **Cursor's tools** when using the new **Gemini 06-05 model**, alongside ongoing problems with **Github access** for Background Agents and difficulties setting up default environments for these agents.
- [**MCP Ecosystem Expands with Inspector Fork and Silent Slack Agents!**](https://discord.com/channels/1312302100125843476/1312302100125843479/1380282993452781579) The MCP (Glama) Discord showcased several new developments, including an **MCP inspector fork** with built-in **LLM chat** and sampling support ([MCP Inspector Fork GitHub](https://github.com/MCPJam/inspector)). Additionally, a **Slack MCP server** emerged, enabling the creation of **silent, invisible AI Agents** without needing Slack bots or apps ([Slack MCP Server GitHub](https://github.com/korotovsky/slack-mcp-server)), and a simple server named **inked** was launched ([inked server GitHub](https://github.com/coldielb/inked)).
- [**Unsloth Notebooks Go Viral While Qwen Finetuning Hits Snags!**](https://discord.com/channels/1179035537009545276/1179035537529643040/1380271178693873684) Unsloth AI's notebooks repository started trending on GitHub ([Unsloth AI Notebooks GitHub](https://github.com/unslothai/notebooks)), buoyed by releases like the **DeepSeek-R1-0528-Qwen3 (8B)** notebook. However, users finetuning Qwen models with new tokens reported issues where loaded models didn't seem to apply trained weights, prompting advice to check GitHub issues and upgrade `unsloth_zoo` and `unsloth` via pip.

**Theme 4: Silicon Sizzlers & Kernel Conundrums: ROCm on Windows, Tinygrad's Tussles**

- [**ROCm Wheels Roll onto Windows for Radeon GPUs!**](https://discord.com/channels/1189498204333543425/1233704710389764236/1380421783064416327) GPU Mode members celebrated the arrival of unofficial **PyTorch + ROCm wheels** providing native **Windows** support for **Radeon GPUs**, built using [TheRock](https://github.com/ROCm/TheRock) and targeting Python 3.11/3.12. These community efforts, primarily tested on **Strix Halo (gfx1151)**, aim to support a range of AMD GPUs like Navi31/32/33 and were showcased with a ComfyUI example ([Adyaman's X Post on ROCm Windows Wheels](https://x.com/adyaman/status/1926368074866757857)).
- [**Tinygrad Kernels Lag, Manual OpenCL Shines!**](https://discord.com/channels/1068976834382925865/1070745817025106080/1380273021066936430) Users in the tinygrad Discord grappled with slow GPU kernels generated by tinygrad, particularly for shuffling large datasets like in the `hlb_cifar10` example. A manually written **OpenCL kernel** for the same task performed significantly faster (**0.33 seconds** vs. tinygrad's **5 seconds**), leading to investigations into tinygrad's kernel generation logic and missing LLVM optimizations like **InductiveRangeCheckElimination**.
- [**Kernel Crafters Eye ThunderKittens and Debate Triton vs. AITemplate!**](https://discord.com/channels/1189498204333543425/1189498205101109300/1380503546025480223) In GPU Mode, developers suggested using the [**ThunderKittens** library](https://github.com/HazyResearch/ThunderKittens) to abstract kernel writing, especially for non-Hopper GPUs, with discussions on potentially porting it to AMD. Meanwhile, [**AITemplate**](https://github.com/facebookresearch/AITemplate) was declared to be in maintenance mode, with **torch.compile** and **AOTInductor** recommended as stronger, actively developed alternatives.

**Theme 5: Trust Traps & Truth Trials: Deepfakes Deceive, Benchmarks Baffle, Vixra Vanquished**

- [**AI Audio Deepfake Detector Fooled by ElevenLabs!**](https://discord.com/channels/879548962464493619/897390720388825149/1380535468676087899) A HuggingFace member fine-tuned **Facebook's ConvNeXt-Tiny** model for audio deepfake detection, hosting it on a [Hugging Face Space for Audio Deepfake Detection](https://huggingface.co/spaces/kubinooo/convnext-tiny-224-audio-deepfake-detection). However, the model spectacularly failed a test when it classified an audio generated by **ElevenLabs** as **100% Real**, prompting debugging efforts and discussions on model generalization.
- [**Livebench Loses Credibility, LMArena Users Revolt!**](https://discord.com/channels/1340554757349179412/1340554757827461211/1380259896628477962) The LMArena community heavily criticized the **Livebench** benchmark after it ranked **GPT-4o** above models like **Claude 3.7** and **Gemini 2.5 Pro**. Accusations flew regarding the CEO's alleged bias against Google and manipulation of test questions, leading to calls to *"ban livebench from being discussed here."*
- [**Vixra Vetoed: "Crank Repository" Undermines Credibility!**](https://discord.com/channels/729741769192767510/747850033994662000/1380271503643250699) When an EleutherAI member shared their paper on [vixra, an e-print archive](https://ai.vixra.org/abs/2506.0018), other members strongly advised against using the platform, labeling it a *"crank repository that will strongly undermine your credibility."* The consensus was to use **ArXiv** instead for publishing research, with members pointing to existing [evolutionary work on ArXiv](https://arxiv.org/abs/2206.08896) as a better example.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **PPLX Pro Users Hit Prompt Limits**: Perplexity AI changed its **rate limit** to **100 prompts/day** for **2.5 Pro**, disappointing users who subscribed to the original unlimited offering.
   - One user complained that *the fact that they lowered limits and made it seem like doubling the 50 to 100 was a good move by them that was stupid lost all my trust in their Pro sub*.
- **Android Fans tout OS Superiority**: Members are debating the merits of **Android** versus **iPhone**, highlighting the increasing availability of **7000mah battery** Android phones, and *YouTube Revanced*.
   - One member noted: *Every year I think I want an iPhone. Then next year android gets better*.
- **Comet Browser Set to Launch Soon**: The **Comet browser** is expected to arrive soon with scheduling features, potentially *next week*.
   - One member speculated that *comet not going to change anything*.
- **Sonar Deep Research Superpowers Unlocked**: **Sonar deep research** now features enhanced reasoning capabilities and an async mode, improving its analytical prowess.
   - Members also noted the **Academic mode** is now available on all models, enriching every citation with **title, url, and date**.
- **API Search Capabilities Compared to Online Interface**: A member expressed frustration with the **API's search capabilities**, noting that it feels like *50% of the online interface*.
   - In contrast, Perplexity's public search page [artisticai.art](https://artisticai.art) shared links to search results covering a range of controversial topics, including accusations against **Michael Tait**, **Pakistan's diplomatic deceptions**, the **Dark Tetrad's digital amplification**, and the exploration of **cruelty**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Teslas Generated with Precise Prompting**: Members shared prompting techniques for generating images of specific objects, such as a **Tesla Model 3**, with emphasis on constant verification of the **shape, size, angles, and position**.
   - One member shared their prompt: *generate an svg of a Tesla Model 3. make it maximally detailed and look exactly like the real thing.*
- **Gemini 2.5 Pro's Performance**: Users compared different versions of **Google's Gemini 2.5 Pro**, noting that the **0605 version** often misses comments in lengthy contexts and exhibits severe hallucinations.
   - The **0506 version** performs better for extracting specific user comments within a **60k token context**, though a performance drop was observed at the **8k mark** in the newer version.
- **Le Chat's API discovered**: A user discovered that **Le Chat** is still running and made available unofficially through some **Google internal APIs**, allowing programmatic calls to the **Gemini API** through an *"apps"* feature.
   - Another user said that the person *who made it was hella CCP*, and confirmed abusing the big time apps feature from Google.
- **Kingfall briefly leaked on AI Studio**: The **Kingfall model** was briefly available in AI Studio, leading to speculation that it is based on **DeepMind** and is exceptionally fast, with some users stating *We do its in ai studio got released today*.
   - It got pulled shortly afterwards.
- **Livebench Benchmarking Disputed**: Members criticized the reliability and relevance of **Livebench** after it rated **4o** over **Claude 3.7, Claude 3.5 and 2.5 Pro**.
   - The community stated *ban livebench from being discussed here* and cited issues such as the CEO's alleged bias against **Google** and manipulation of testing questions.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI unleashes Common Pile v0.1**: EleutherAI announced the release of the **Common Pile v0.1**, an **8TB dataset** of openly licensed text from **30 sources** ([paper](https://arxiv.org/abs/2506.05209)), with models **Comma v0.1-1T** and **Comma v0.1-2T** achieving performance comparable to **Llama 1 and 2 7B**.
   - The organization is hoping this dataset fosters a more ethical language model ecosystem via transparency, better authorship, and a lack of copyrighted data ([GitHub](https://github.com/EleutherAI/common-pile), [HuggingFace](https://huggingface.co/common-pile), [Blog](https://huggingface.co/blog/stellaathena/common-pile)).
- **LLMs Fall Prey to Deceptive Storytelling**: Members observed that **LLMs** trained in-context by humans can develop a propensity for generating **unfalsifiable narratives**, especially since they are only corrected on topics known by the human trainers.
   - One member reported concerning experiences with **ChatGPT's memory feature**, which can lead to **misalignment** and unpredictable behavior over long periods, as if it were fine-tuned by long term interactions.
- **Attention Predates the Transformer**: Members discussed the widely known fact that **attention mechanisms** predate transformers, but that there is a bias towards **Bahdanau attention**, and that on Twitter the terms attention and transformers are near synonymous.
   - A user linked to a [tweet](https://x.com/SchmidhuberAI/status/1864701357107634390) from **Schmidhuber** claiming priority on attention but in linear form and not quadratic, and members linked to [criticism](https://bsky.app/profile/reecedkeller.bsky.social/post/3lqv4hxouck27) of **Schmidhuber's** claims.
- **Avoid Vixra at all Costs**: A member shared a link to their paper, *Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance*, on [vixra](https://ai.vixra.org/abs/2506.0018).
   - Other members strongly advised against using **vixra**, describing it as a crank repository that *will strongly undermine your credibility*, suggesting **ArXiv** instead, and pointing to [evolutionary work](https://arxiv.org/abs/2206.08896).
- **MPL Weights Get Some Vis**: A member sought feedback on a [project](https://grgv.xyz/blog/neurons1/) visualizing **MPL weights** projected into **vocabulary embedding space**.
   - The project aims to determine the viability of the approach and potential directions for further exploration.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Tools Crumble with Gemini 06-05**: Users are reporting that **Cursor's tools** are experiencing issues when using the new **Gemini 06-05** model, similar to problems encountered with previous **Gemini** and **Flash** updates.
   - One user summarized that **Gemini is good at code**, **OpenAI is good at instructions and code**, and **Claude is good at both**.
- **Documentation Saves Cursor Users**: Members are finding **Cursor's documentation** helpful for finding updated information and knowledge about **Cursor**.
   - One member specifically said they were just reading the **Cursor documentation** about it.
- **Cursor Hits Stratospheric $10B Valuation**: After **Anysphere's** latest funding round, **Cursor** has reached a **$10 billion valuation**.
   - A member shared a [link to a TechCrunch article](https://techcrunch.com/2025/06/05/cursors-anysphere-nabs-9-9b-valuation-soars-past-500m-arr/) about the valuation.
- **Github Access Denied for Background Agents**: Users are reporting issues with **Cursor** connecting to **Github** in the **Background Agents** config, receiving an *'Access Denied: No repos with user access found'* error despite normal **Github** connection working fine.
   - One user also encountered an *'Unable to refresh, could not reach GitHub API'* message, even without VPN or unusual network configurations.
- **Background Agents' Environment Troubles Emerge**: Multiple users are failing to set up the default environment for **Background Agents**, encountering an *'[invalid_argument] Error'* after multiple attempts.
   - A user asked if anyone has a working environment **JSON**, expressing difficulty in translating their working *docker compose* setup to the agent.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Excels at SVG Generation**: Members lauded **Gemini's** ability to generate detailed SVG images from prompts in [AI Studio](https://ai.google.dev/).
   - One user marveled at **Gemini's** ability to dynamically adjust elements while preserving the overall robot structure.
- **Gemini Naming Convention Creates Confusion**: Members expressed confusion over the naming conventions for **Gemini** models, especially the version labeled **06-05**.
   - One member quipped, *They should really come up with better names...*, while others suggested the naming reflected the update's release date, essentially the same model *with a bit more RL*.
- **Veo3 Delivers Character Consistency**: A member noted having *major results of consistent characters and audio using Veo3* but failed to provide a video showing it.
   - The claim centered around generating video with consistent character and audio, though supporting evidence was not shared.
- **Markdown Preference Beats PDF in AI Input**: A member criticized using **PDFs** for data input into language models due to their cryptographic complexity and design for human pixel-perfect rendering, not AI comprehension.
   - Instead, they advocated for **Markdown** as a superior plain-text format, highlighting its alignment with model training data.
- **UPSUM Chain Prompt Manages Context**: A member introduced the **UPSUM Chain Prompt**, a meta-prompt designed to summarize conversations and maintain context, recommending its use for condensing extensive chat logs into concise narratives.
   - They also suggested to *upsum your upsum collection*, using it as the context for future prompts, and linked a [YAML configuration for the UPSUM Chain Prompt](https://example.com/upsum-yaml).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Notebooks Trending!**: The Unsloth notebooks repository is now trending on GitHub ([https://github.com/unslothai/notebooks](https://github.com/unslothai/notebooks)), including the new **DeepSeek-R1-0528-Qwen3 (8B)** notebook release ([https://x.com/UnslothAI/status/1931008531299545339](https://x.com/UnslothAI/status/1931008531299545339)).
   - The RAM doubled for any config keeping the price the same, however users are reporting issues using the notebook.
- **Chrome Autofill Crashes Prompt Document Editing Fix!**: Members identified a crash issue in **Chrome** related to the **autofill** feature when typing in a document editor with large documents, triggering a `TransactionTooLargeException`.
   - It was discovered to be a bug in Chrome internals when notifying the autocomplete service, and disabling autofill is the verified fix for the crashes.
- **Qwen3 Powers Up with New Embedding and Reranker Models!**: Two new **Qwen3** models were released: the [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
   - Members anticipate positive results from these new models, but note the lack of benchmarks for TTS.
- **IDE Cage Fight: VS Code Wins!**: Members discussed the best IDE for coding with AI, with [VS Code](https://code.visualstudio.com/) being the top contender with [GitHub Copilot](https://github.com/features/copilot) being used for autocomplete.
   - One user noted VS Code sometimes freezes, but its overall utility for AI development was considered best in class.
- **Taming the Tokenizer: Users Report Finetuned Qwen Model Mystery**: A user finetuned the `Qwen3-8B-unsloth-bnb-4bit` model on new tokens, pushed it to the hub, but encountered an issue where **the loaded model didn't seem to apply the trained weights**.
   - Members suggested this might be related to adding new tokens, and the user was advised to check for similar issues on GitHub and await a new release with merge logic fixes and was pointed to the recent pypi release, users can upgrade using `pip install --upgrade unsloth_zoo` and `pip install --upgrade unsloth`.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **ThunderKittens Tames Kernel Writing**: Members suggested using [**ThunderKittens**](https://github.com/HazyResearch/ThunderKittens) library to abstract the kernel writing process, if not using a **Hopper GPU**, especially since the core matmul operation seems to be the [`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`](https://github.com/HazyResearch/ThunderKittens/blob/d69697a3337e31d0060178c9049f1184e7e7ad7f/include/ops/warp/register/tile/mma.cuh#L17) primitive.
   - One member reached out via DMs about the possibility of porting and generalizing **ThunderKittens** to **AMD** architectures, indicating that while it may be involved, it seems potentially possible.
- **AITemplate sunset, torch.compile rises**: The community has expressed that [**AITemplate**](https://github.com/facebookresearch/AITemplate) is no longer under active development, being in maintenance mode for a couple years already, leading to a consensus that **torch.compile** is a stronger option.
   - **AOTInductor** was recommended as a C++ runtime alternative.
- **ROCm wheels hit Windows!**: Unofficial **PyTorch + ROCm** wheels now support native **Windows** on **Radeon GPUs**, bundled with libraries for easy install; built using [TheRock](https://github.com/ROCm/TheRock), these community-driven wheels target **Python 3.11/3.12**.
   - Tested mainly on **Strix Halo (gfx1151)** but aiming to support **gfx1100/gfx1101/gfx1102/gfx1103/gfx1151/gfx1201** (Navi31/32/33, 780M, 8060S, and 9070/XT), showcased with a ComfyUI example [here](https://x.com/adyaman/status/1926368074866757857).
- **CUDA C++ Workshop to be 'hands-on'**: A full-day, hands-on training in modern **CUDA**, optimization, and debugging will be hosted in the **CUDA C++ Workshop** on **June 10** at **GTC Paris** at **VivaTech**.
   - NVIDIA also linked [the full agenda](https://www.nvidia.com/en-eu/gtc/) as well as Q&A sessions on **CUDA, AI, HPC**, and more with the engineers behind the tech.
- **FLE API Specs Elicits Feedback**: A member seeks feedback on their take on an **FLE API**, with a *request for comments* style [GitHub repo](https://github.com/MortenTobiasNielsen/FLE-API-specification).
   - Another member confirmed they would review it and provide specific comments on the issues.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Hub Suffers Brief Heart Attack**: The **Hugging Face Hub** experienced an outage, resulting in a **502 error**, with users reporting issues accessing the site from various locations.
   - The infra team swiftly resolved the problem, leading to praise for their quick work; the outage may have coincided with a staff reply on a [related discussion](https://huggingface.co/spaces/transformers-community/support/discussions/13#6842efbfac97e96a2f38dcbe).
- **DDR5 RAM Restrictions Confuse Enthusiasts**: Discussion involved **AMD Zen 5 CPUs** being limited to **128GB** max RAM, despite **64GB DDR5** sticks becoming available and some motherboards supporting **256GB RAM**.
   - It was speculated that MOE models could run with minimal VRAM and ample RAM, even on outdated CPUs, while newer CPUs are strangely locked to lower RAM limits.
- **Model Deepfakes Audio, Fails Test**: A member fine-tuned **Facebook's ConvNeXt-Tiny** model to classify audio as real or fake, merging computer vision techniques with audio analysis for deepfake classification research and hosted on a [Hugging Face Space](https://huggingface.co/spaces/kubinooo/convnext-tiny-224-audio-deepfake-detection).
   - Another member tested the model with a fake generated audio from **11elevenlabs**, but it was incorrectly classified as **100% Real**, leading to debugging efforts.
- **HF Computer Vision Hangout Slides Shared**: Slides from today's **Computer Vision Hangout** were shared, including updates on Computer Vision at **Hugging Face** in the [HF_CV_Hangout_June_25.pdf](https://cdn.discordapp.com/attachments/922424143113232404/1380522336335298650/HF_CV_Hangout_June_25.pdf).
   - A member from **Pruna AI** gave a presentation on speeding up image generation, the slides of which were shared in the [PrunaAI_SpeedUp_Generation.pdf](https://cdn.discordapp.com/attachments/922424143113232404/1380522337027227709/PrunaAI_SpeedUp_Generation.pdf).
- **GPT-4o Parses poorly in Smolagents**: A member asked if others are experiencing many parsing errors when using **GPT-4o** and **GPT-4o mini** with a **smolagents** code agent.
   - No resolution was provided, but this may be an indication of ongoing instability in the parsing model.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro Benchmarks Spark Debate**: Early benchmarks of **Gemini 2.5 Pro** against a publicly exposed model on Vertex AI showed scores of **86.2%**, leading to discussions about the achievability and *stochastic* nature of these results, based on settings from [this commit](https://github.com/cheahjs/aider/commit/cacd932c9a474f871229b166e6be0d1858854e17).
   - Some users expressed surprise at **Gemini's** performance relative to **Opus**, while others emphasized a strong preference for **Opus**, citing coding task capabilities.
- **Price Fuels Model Preference Debates**: Discussions on the price-to-performance ratio highlighted differing priorities: some prioritize lower costs, while others value output quality and workflow synergy, even with more expensive models like **Opus**.
   - One user summarized the sentiment with *price is what you pay, value is what you get*, encapsulating the trade-offs in model selection.
- **Aider Users Evaluate Cursor's Development Style**: Some **Aider** users experimenting with **Cursor** found it slower, with an overly verbose chatbot and an agent mode that requires significant control.
   - One user found that the **aider approach** suits their style of development of *(careful, considered prompting), terminal-driven, branching like a maniac because obsessive about being able to revert to good known states, etc. etc...* and don't like *the fling stuff at the wall vibe I get from the cursor fanboys*.
- **Speech-to-Text Workflow Integration Sought**: A user interested in **speech-to-text workflows** explored options like **Wispr flow** for iOS but preferred **superwhisper**.
   - The user aimed to incorporate more speech-based workflows into their daily routines, which shows a desire for efficient and alternative input methods.
- **Challenges Configuring Aider with vllm Server**: A user encountered issues configuring **aider** to work with a local **vllm server** due to **aider** requiring model names to start with the provider prefix, i.e. *openai/Qwen3*.
   - Another user suggested adding the *openai/* prefix to resolve the issue, such as *openai/unsloth/Qwen3*, to align with **aider's** expected naming convention.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **App Based Spam Alarms Users**: A new type of *"app" based spam* emerged in the channel, raising concerns about potential account compromises; staff responded by removing the "Use External Apps" permission.
   - An initial investigation indicated that **no accounts were compromised**.
- **Gemini 2.5 Pro Gets Rate Limited**: Users reported that **Gemini 2.5 Pro** has a rate limit of **100 messages per 24 hours** within **LM Studio**, later updating that the rate limit was adjusted to **150 RPM**.
   - Users are exploring alternative ways to use LM Studio without running into these rate limits.
- **Qwen3-4B Outperforms Open Thinker**: Discussion arose around the **Open Thinker** model, with members pointing out that **Qwen3-4B** surpasses it based on official benchmark results and [runs smoothly](https://huggingface.co/Qwen/Qwen3-4B) on a gaming PC.
   - The community seems to be gravitating towards **Qwen3-4B** for its superior performance metrics.
- **Unified Memory gives Qwen3 a Jolt**: A user achieved **12.05 tok/sec** on the first reply with **Qwen 3 235B Q3_K_S** (context: 12233) using a **GMKtec Evo-X2** with an **AMD Ryzen AI Max+ 395** and **128GB** of unified memory.
   - Loading **64k of context at Q8_0** achieved **9.33t/s** with **6.27s** to first token, despite facing repetition issues with **Unsloth Q3_K_XL**.
- **Llama 3.3 70B Maxes Out VRAM**: Members tested **Llama 3.3 70B with 128k context at F16**, running entirely in VRAM, achieving **4.85 tok/sec** on the first prompt and **4.83 tok/sec** on the second prompt.
   - They were using a **GMKtec Evo-X2, AMD Ryzen AI Max+ 395 with 128GB of unified memory**, with the iGPU being a **8060S**, roughly equivalent to a **3060** in AI computation.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Team Magics Pixi Transition**: A member expressed gratitude to the Modular team for a seamless transition from `magic` to `pixi`, calling it a *"pin compatible"* process.
   - The member conveyed their appreciation using emojis, highlighting the smooth nature of the transition.
- **Mojo upgrade demands memory alignment**: A member mentioned the necessity of *memory alignment* while upgrading Mojo on their system.
   - Another member concurred, emphasizing the importance of memory alignment in the context of Mojo upgrades.
- **Mojo makes inroads into Bioinformatics**: A developer shared their enthusiasm for utilizing **Mojo** in bioinformatics, enjoying the challenges of implementing **SLURM** and HPC solutions for biotech startups.
   - They observed that researchers often develop outcome-driven software and automation solutions that aren't always shared across the industry.
- **Mojo's immutable variable still on request**: A member inquired about declaring immutable values in Mojo, seeking a way to prevent changes to a runtime value after its initialization.
   - Another member clarified that creating an immutable variable isn't currently possible, but they proposed [a workaround using helper functions](https://github.com/modular/modular/blob/main/mojo/proposals/remove-let-decls.md) for immutable references.
- **Mojo syntax too verbose?**: Developers debated the verbosity of Mojo's syntax, particularly the `var` keyword in struct definitions, with some finding it burdensome.
   - The conversation extended to preferences for concise syntax, with one member expressing fondness for the **K** programming language, known for its terse style.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **OLMo Models Debut as Fully Open-Source**: Members highlighted that [Allen.ai's OLMo models](https://arxiv.org/abs/2501.00656) are completely open-source, encompassing the **training code, data, and papers**.
   - A member suggested checking out [HuggingFace's Fineweb-2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and its ablations, stating that *RedPajama is kinda dirty and old*.
- **Atropos Becomes New Greenfield for Tool Calling**: The team is actively developing [Atropos environments](https://github.com/NousResearch/atropos/pull/163) and generating datasets for **verified reasoning trace answers**.
   - Using [this environment](https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py), they improved DeepHermes' single and parallel tool calling benchmarks on the Berkeley tool calling benchmark by **5x and 2.5x respectively**.
- **Sequential Tool Calling Capabilities in Development**: A team member confirmed they are training the **tool call directly into the reasoning trace**, with a current focus on **sequential tool calling**.
   - No additional information about this development was given.
- **EleutherAI Drops Massive Common Pile Dataset**: [EleutherAI](https://blog.eleuther.ai/common-pile/) has released one of the **largest sets of commercial and licensed data** for language modeling.
   - The announcement was published on [X](https://x.com/EnricoShippole/status/1931023312647299405).
- **LLM Reproducibility is Vibe Based**: A member shared a picture likening transformer pre-training to discovering **one recipe for a cake**, with the underlying chemistry not fully understood.
   - Another member responded, *LLM cooking is **vibe based and yolo**.*



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Launches Model RSS Feed**: OpenRouter announced the availability of an **RSS feed** for its [API models](https://openrouter.ai/api/v1/models?use_rss=true), enabling users to stay updated on new models and changes within the OpenRouter ecosystem.
   - The real simple syndication feed provides up-to-date information for developers to easily track changes.
- **Gemini 2.5 Pro Experiences Intelligence Regression**: Users report a decline in intelligence for the **06-05** version of **Gemini 2.5 Pro**, with some describing it as *flash thinking level dumb*.
   - The consensus suggests using the older model while it's available, as the newer version was possibly downsized for speed and cost.
- **Claude Max vs Gemini Pricing Faceoff**: A user jokingly suggested pirating **Gemini 2.5** to avoid paying, which sparked debate on the cost-effectiveness of **Claude Max** versus **Gemini** API usage.
   - The user noted **Claude Max** is more economical for *vibe coding* and daily use, especially for users sensitive to API costs.
- **Privacy Concerns Emerge from OpenAI Logging**: Concerns were raised about [an article](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare/) that stated **OpenAI** is forced to log outputs, prompting questions about data retention on **OpenRouter**.
   - It was clarified that the *Enable training and logging* setting is not relevant for OpenAI models, and OpenAI may retain inputs for up to 30 days.
- **GPT-4.1 Mini Excels in Coding and Tool Use**: **GPT-4.1 mini** received praise for its coding abilities, tool usage, and cost-effectiveness, making it suitable for routine tasks and inference.
   - It is considered a *real winner* and more obedient than **Gemini 2.5 Flash**, especially for tasks not involving code or math, though not as good for creative writing as **Claude 3.7**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **LLM Chat Debuts on MCP Inspector Fork**: An **MCP inspector fork** with built-in **LLM chat** and **sampling support** was released, inviting testing and feedback via [GitHub](https://github.com/MCPJam/inspector).
   - This fork provides a testbed for **LLM-enhanced tool interaction** within the MCP ecosystem.
- **Cloudflare Deployment Confronts Snags**: A user encountered problems deploying an **MCP server** on **Cloudflare Workers** using a **workers.dev link**.
   - The member sought assistance after struggling to integrate a custom MCP server with OpenAI, even after adding descriptions to all tools.
- **Silent AI Agents Enter the Slack Workspace**: A member announced the emergence of their **Slack MCP server** on GitHub, emphasizing its capability to construct **silent**, **invisible AI Agents** without the need for bots or Slack applications, now available on [GitHub](https://github.com/korotovsky/slack-mcp-server).
   - They attached a [GIF showcasing its usage](https://cdn.discordapp.com/attachments/1315696461316358175/1380517187785326692/434543420-35dc9895-e695-4e56-acdc-1a46d6520ba0.gif?ex=68442a52&is=6842d8d2&hm=78437dbb1f2f8f9776d0855153c1d68e2ec00098b74fbddc18ba4e53e272148e&).
- **Inked Simple Server arrives on Github**: A dead simple server named **inked** was launched on [GitHub](https://github.com/coldielb/inked), inviting community involvement through contributions and experimentation.
   - The server features **two tools** and **three total functions**, and can be installed globally via `npm install -g @frgmt/inked`.
- **VAPI MCP Demo is Calling Hardware Stores**: A demo showcasing **VAPI MCP** calling hardware stores for part procurement, targeting **hardware engineers** was shared with a [video](https://cdn.discordapp.com/attachments/1312302100125843476/1380307827385438338/helion_call_demo.mp4?ex=6844b8d6&is=68436756&hm=cb6e38829856a5ca52b546356018a0237ce82d3e80ce08b9ca48d589838754ac&).
   - The tool is aimed at hardware engineers.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AMD Chips Away at Untether AI**: AMD acquired the team behind AI chip startup [Untether AI](https://www.crn.com/news/components-peripherals/2025/exclusive-amd-acquires-team-behind-ai-chip-startup-untether-ai).
   - The acquisition enhances AMD's capabilities in AI chip design and potentially accelerates their entry into new markets.
- **Claude Projects Now 10x Bigger**: Anthropic announced **Claude Projects** feature now supports **10 times more content** and activated a new retrieval mode for expanded functional context.
   - This update is rolling out to all paid **Claude** plans, with users touting it as a *'game changer'* and a substantial upgrade over **ChatGPT**.
- **Alibaba Opens Qwen3 to the World**: Alibaba's **Qwen3-Embedding** and **Qwen3-Reranker Series** models launched, setting new standards in multilingual text embedding and relevance ranking, supporting **119 languages**.
   - Available in various sizes (0.6B, 4B, 8B) and open-source on [Hugging Face](https://huggingface.co/), [GitHub](https://github.com/), and [ModelScope](https://modelscope.cn/), they're enabling use cases like document retrieval, **RAG**, classification, and sentiment analysis.
- **Netlify Comes for Serverless with SupabaseDB**: Netlify announced **Netlify DB**, a serverless Postgres database powered by Neon, designed for AI-native development, aiming to reduce friction between code and data.
   - Easy to set up with a single command, the **Netlify DB** integrates into projects via `netlify dev`, streamlining development workflows.
- **Zapier Wants AI-Fluent Workers**: Zapier is measuring AI fluency among employees, requiring **100% of new hires to be AI fluent**, and categorizing fluency into 'Unacceptable,' 'Capable,' 'Adoptive,' and 'Transformative' levels.
   - The company uses screenings, skill tests, async exercises, and live interviews for evaluation, showing a commitment to integrating AI skills throughout their workforce.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **HFModelTokenizer Prompts Evaluation Customization**: Discussion focused on modifying **HFModelTokenizer** to render templates without tokenization for evaluation, particularly for custom prompt templates.
   - The proposed solution prioritizes custom prompt templates if they exist in the tokenizer; otherwise, the **HFModelTokenizer** chat template is used if `apply_chat_template` is true.
- **Alpaca Cleaned Regression Reproduction Troubles**: A member reported difficulty reproducing a regression on the **alpaca_cleaned** dataset and requested more details on the initial detection setup.
   - Observations showed that fine-tuning **Qwen3-4B** on the **alpaca_cleaned** dataset yielded similar evaluation results as the non-fine-tuned version, though it was pointed out that alpaca is a pretty saturated dataset and 4B is small.
- **Axolotl Convergence Evaluated after C4 Finetuning**: A member shared [Axolotl PR #2590](https://github.com/axolotl-ai-cloud/axolotl/pull/2590) to show loss curves on **C4**, suggesting **torchtune** evaluation post **C4** finetuning since **Axolotl** converges.
   - They noted that the loss curve does not strongly suggest that **torchtune's** methods diverge, and updates using the **Axolotl** values as a reference were offered.
- **Torchtune Considers Clipping Logprobs**: A discussion took place regarding adding **logprobs clipping** to **torchtune**, with one member disagreeing that the existence of the feature in another repository is sufficient justification.
   - While the feature is available elsewhere, concerns were raised about it not being intended for user modification and the difficulty of correctly exposing it; however, another member preferred ensuring ease of self-implementation over direct maintenance of the feature.
- **Fused Optimizer Throws Assertion Error**: A member reported an `AssertionError` related to `fused_adagrad` when using a fused optimizer on a nightly build, specifically when a compute mesh was not found.
   - Testing revealed the issue only occurred with `fused=True`, and **SGD** started working after upgrading to the latest **torchtune**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Marius Paper Surfaces for Training Industry LLMs**: A member shared the [Marius paper](https://arxiv.org/abs/2402.00854) for training industry-level LLMs with real-world datasets, asking for insights from experts on data handling, preventing overfitting and stabilizing training.
   - A member mentioned Sebastian Raschka's "Build a Large Language Model (From Scratch)" ([YouTube link](https://youtu.be/Zar2TJv-sE0)) as a starting point but sought more detailed training pipelines with mixed, diverse datasets, methods for stabilizing training, and preventing catastrophic forgetting.
- **RAG Clustering Conundrum Conjured for Emails**: A member sought a way to perform clustering with emails using an out-of-the-box **RAG** solution, aiming to put the emails in n bins without knowing what n is or having labels.
   - Solutions included embedding each email using ModernBERT and using a traveling salesman problem solver or OpenAI's Embeddings ([platform.openai.com](https://platform.openai.com/docs/guides/embeddings#ho)).
- **Meta's OPT-175B Logbook Disclosed**: Members shared **Meta's OPT-175B logbook**, documenting problems in the training process ([GitHub link](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf), [ArXiv link](https://arxiv.org/abs/2205.01068)).
   - This logbook is valuable for those interested in the challenges and solutions encountered during the training of large language models.
- **Nemotron-H Reasoning Models Boost Throughput**: **NVIDIA** introduced the **Nemotron-H-47B-Reasoning-128K** and **Nemotron-H-8B-Reasoning-128k** models for reasoning-intensive tasks, now available in [FP8 quantized variants](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/?linkId=100000368479233).
   - These models, built from the **Nemotron-H-47B-Base-8K** and **Nemotron-H-8B-Base-8K** foundation models, aim to advance the science behind reasoning models with efficient throughput in latency-sensitive environments.
- **EleutherAI Trains Competitive LLM with Public Data**: A member noted that [EleutherAI](https://huggingface.co/blog/stellaathena/common-pile) demonstrated the feasibility of training a competitive LLM using public-domain and openly-licensed data.
   - This highlights the potential for creating powerful language models without relying on proprietary datasets.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **User Debugs Slow tinygrad GPU Kernels**: A user debugs a slow GPU kernel generated by tinygrad when shuffling a dataset tensor of float32 `[50000,3,32,32]` in the `hlb_cifar10` example, after trying `DEBUG=4` and `VIZ=1`, they realized `BEAM=4` doesn't fix the underlying issue.
   - A manually written **OpenCL** kernel for shuffling the same sized array (**50000,3,32,32**) shuffles in **0.33 seconds**, versus the tinygrad-generated kernel which takes **5 seconds** even with simple unshuffled indexing, prompting further investigation into tinygrad's kernel generation.
- **Manual OpenCL Kernel Leaves Tinygrad Kernel in the Dust**: A user benchmarks a manually written **OpenCL** kernel versus the auto-generated **tinygrad** kernel, they are trying to understand why **tinygrad** generates such a slow indexing kernel, especially since CPU-based copying and shuffling is faster.
   - The user realized a simplified indexing **OpenCL** kernel would be much faster, and found that `np.random.permutations(50000*3*32*32)` and `np.random.permutations(50000)[None].repeat(3*32*32, 0).T.flatten()` both took **0.33 seconds**.
- **Loop Splitting Investigation**: A member investigates speeding up CAT with **LLVM** and asks if **loop splitting** is only present on the **ROCm llvm-project**.
   - They reference the [ROCm documentation](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/reference/rocmcc.html#loop-splitting) on loop splitting and note it is only in their custom llvm-project.
- **IRCE Missing from llvm.py**: A member notes that the **llvm C source** used in runtime/autogen/llvm.py lacks **InductiveRangeCheckElimination** from the **C++ LLVM library**.
   - The member is considering using *llvmlite* to get access to IRCE or extern/rewrite C++ since they cannot add loop splitting.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus' New Video Function Underwhelms Users**: The new video function was received poorly by users, who found it *very immature* and not ready for practical use.
   - One user reported seeing a video from a close friend and feeling that the feature is too premature to be useful.
- **Credit Costs Drive Away Potential Manus Users**: Users are complaining about the high credit costs of **Manus**, with one user pointing out that *1900 credits for 19 dollars* doesn't go very far when tasks cost *300-400 credits* each.
   - The high costs are pushing users to seek out cheaper alternatives; one user linked to [guides from the Manus Team](https://discord.com/channels/1349440650495398020/1370393476029616238) on how to perform cheaper tasks.
- **Manish Model Update Rumors Heat Up**: Users are actively wondering if the **Manish** model will be updated to **Sonnet 4.0**.
   - Speculation is fueled by the recent partnership with **Claude**, although some users note that **Sonnet's** lack of context length could be a problem, which **Manus** solves.
- **Egyptian User Checks In**: A user appeared and simply inquired if there were any other Egyptian users present in the chat.
   - This was the extent of the discussion.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Explores Agent Production**: @tuanacelik led a discussion at Snowflake Dev Day on [challenges in productionizing AI Agents](https://t.co/DJGBe3TqZb).
   - The session highlighted the current blockers and potential solutions for deploying agents in real-world applications.
- **Protocols Duel for Agent Communication Standard**: @seldo presented a lightning tour of **13 different protocols** at the [MCP Dev Summit](https://t.co/qZv8duKRut), including **MCP, A2A, and ACP**, all competing to standardize agent-tool communication.
   - The talk underscored the fragmentation in the agent communication landscape and the need for a unified standard.
- **LlamaIndex Boosts RAG in Munich**: @itsclelia will speak at the **BASED Meetup in Munich** on June 12th, sharing best practices for enhancing **RAG pipelines**, from data preparation to query optimization.
   - The talk aims to equip attendees with strategies to improve the efficiency and effectiveness of their RAG implementations.
- **`files_via_content` Mode Clarified**: A member requested details on the [`files_via_content` mode](https://docs.cloud.llamaindex.ai/llamacloud/retrieval/modes#files_via_content-mode) in LlamaIndex.
   - The member got quick access to the relevant documentation in LlamaIndex Cloud, streamlining the implementation process.
- **Community Explores Dynamic Agent Delegation**: A member inquired about dynamically delegating tasks to specialized agents within **AgentWorkflow**.
   - This discussion centers on whether such functionality is natively supported by LlamaIndex or requires custom workflow definitions.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All benefits from VPS for API Server**: A user suggested renting a **VPS** to build the **API server** for **GPT4All**, noting that the current **GPT4All** interface is sometimes unresponsive.
   - The user shared a screenshot indicating some bugs in the current implementation.
- **RAM Pricing Revelations**: A member shared [a YouTube video](https://m.youtube.com/watch?v=Tp0k6VDXUOQ) discussing **RAM pricing**, noting that **1 TB** can be reasonably priced around a few thousand dollars.
   - They added that ordinary **PCs** often struggle with insufficient RAM and that the market for computer components can be global.
- **Imagining MOE Model Metrics**: A user speculated about running **Mistral MOE** or **Deepseek MOE** full **Q8 Quantization** at TRILLION tokens / second.
   - The user linked to an article about [a Chinese CPU vendor](https://www.techradar.com/pro/chinese-cpu-vendor-swaps-amd-zen-architecture-for-homegrown-one-to-deliver-128-core-monster-to-give-epyc-and-xeon-a-run-for-their-money) swapping **AMD Zen architecture** for a homegrown one to deliver a **128-core monster** to compete with **EPYC** and **Xeon**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gratitude Overflowing for DSPy Session**: Two members expressed their gratitude for a **DSPy** session, referencing [a YouTube video link](https://youtu.be/Vqsfn9rWXR8) from the session.
   - A member inquired about the availability of session slides, seeking to further digest the material.
- **Blockchain Expert Enters the Fray**: A software engineer with expertise in **Blockchain** technologies such as **EVM**, **Solana**, **Cardano**, **Hydra**, **Aptos**, **Cosmos**, **Tron**, and **zk-SNARKs** introduced himself.
   - His background signals a push towards decentralized AI applications.
- **AI Agent Architect Joins the Chat**: An engineer specializing in **AI Agents** introduced himself with a background in **LLM**, **NLP**, **LangChain**, **AutoGen**, **TorchRL**, **DL**, **Azure ML**.
   - This introduction highlights the growing interest in autonomous AI systems.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Magic Place Mentioned**: stormortiz mentioned that *here is an magic place* but no further context was given.
   - It is unclear whether this is related to Cohere or AI in general.
- **ML Audio Engineer Joins**: A new member introduced themself in #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1380469381951127634) as a **Machine Learning Audio Engineer**.
   - The community welcomed the new member to the Cohere Discord server.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MCP Tools Authorization Post**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-enterpriseai-oauth-activity-7336749966234718208-Cb9E?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk) regarding building **MCP tools authorization** for the enterprise.
   - The post is an article compiling findings on implementing **enterprise OAuth**.
- **OAuth Findings**: The author's findings on building **MCP tools authorization** for enterprises have been compiled into an article.
   - This article specifically addresses aspects related to **enterprise OAuth** implementation and best practices.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1380259889921916998)** (1263 messages🔥🔥🔥): 

> `Perplexity AI Limits, Android vs iPhone, Comet browser release date, AI Model Ranking, Scheduled actions in Gemini` 


- **User Bemoans PPLX Pro Limits**: Members discussed Perplexity AI's recent **rate limit** changes to **100 prompts/day** for **2.5 Pro**, with a user expressing disappointment, given the service's original unlimited offering for subscribers.
   - A user says: *the fact that they lowered limits and made it seem like doubling the 50 to 100 was a good move by them that was stupid lost all my trust in their Pro sub*.
- **Android Fans tout OS Superiority**: Members debated the merits of **Android** versus **iPhone**, with one highlighting the increasing availability of **7000mah battery** Android phones, while others touted features like *YouTube Revanced* and *Modded apks* unavailable on iOS.
   - One member noted: *Every year I think I want an iPhone. Then next year android gets better*.
- **PPLX Comet Browser Coming Soon**: Members discussed **Comet browser's** expected arrival, including scheduling features, with one member hinting at a release date: *beta comet is out next week rite ?*
   - Another member speculates: *comet not going to change anything*.
- **AI Model Ranking**: Members debate which model family wins, with [Mat claiming allegiance to OpenAI](https://twitter.com/perplexityai/status/1799154406242013210), citing Google's pettiness and recent issues.
   - Another member says: *the product or the propaganda? both*.
- **Google Announces Gemini Scheduled Actions**: Google officially announced the **Scheduled Actions** rollout to Gemini Users on paid plans.
   - This feature is *coming to perplexity soon too*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1380364396513071237)** (4 messages): 

> `ArtisticAI, Michael Tait, Pakistan's diplomacy, Dark Tetrad, Cruelty` 


- **ArtisticAI Site Shares Diverse Links**: The user artisticai.art shared links to Perplexity search and page results covering a range of controversial topics.
   - These topics include accusations against **Michael Tait**, **Pakistan's diplomatic deceptions**, the **Dark Tetrad's digital amplification**, and the exploration of **cruelty**.
- **Perplexity AI Searches Spark Discussion**: A user posted several Perplexity AI search and page links.
   - The shared links point to content about **Michael Tait**, **Pakistani diplomacy**, the **Dark Tetrad**, and explorations of **cruelty**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1380272004531228793)** (21 messages🔥): 

> `Sonar deep research upgrades, Academic mode on all models, Richer Citations, Camera integration to voice chat, Formal reasoning for coding` 


- **Sonar Deep Research Gets Reasoning and Async**: **Sonar deep research** now features reasoning effort plus async mode.
- **Academic Mode Extends to All Models**: **Academic mode** is now available on all models, and every citation is richer with **title, url, and date**.
- **Camera Integration Speculated for Voice Chat**: Some members are hoping for **camera integration to voice chat**, similar to what Google might bake into the next Pixel phones with **Gemma 3n**.
- **Demand for Coding Provers Emerges**: Some members are requesting **formal reasoning for coding**, similar to **Deepseek Prover**, specifically for coding.
- **API Search Capabilities Compared to Online Interface**: A member expressed frustration with the API's search capabilities, stating that it feels like *50% of the online interface*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1380259896628477962)** (1282 messages🔥🔥🔥): 

> `Model Generation, Google's Gemini 2.5 Pro, Mistral Le Chat API, Kingfall model, Livebench concerns` 


- **Model Generation Prompting Techniques**: Members discussed various prompting techniques to generate images of specific objects, such as a Tesla Model 3. One member shared their prompt: *generate an svg of a Tesla Model 3. make it maximally detailed and look exactly like the real thing.*
   - They emphasized the importance of constantly checking whether the **shape, size, angles, and position of each item looks EXACTLY like a Tesla Model 3**.
- **Gemini 2.5 Pro Performance**: Users compared different versions of **Google's Gemini 2.5 Pro**, noting that the 0605 version often misses comments in lengthy contexts and exhibits severe hallucinations, whereas 0506 performs better for extracting specific user comments within a 60k token context.
   - It was also noted that there's a drop at the **8k mark with the new gemini version**, but otherwise better.
- **Mistral's Le Chat API**: A user discovered that Le Chat is still running and made available unofficially through some Google internal APIs, allowing programmatic calls to the **Gemini API through an "apps" feature**.
   - Another user said that the person *who made it was hella CCP*, and confirmed abusing the big time apps feature from Google.
- **Kingfall model leaks and speculation**: The **Kingfall model** was briefly available in AI Studio, leading to speculation that it is based on DeepMind and is exceptionally fast.
   - It got pulled shortly afterwards, with some users stating it got taken down, others stated *We do its in ai studio got released today*.
- **Livebench code benchmarking disputed**: Members criticized the reliability and relevance of **Livebench** after it rated 4o over Claude 3.7, Claude 3.5 and 2.5 Pro.
   - The community stated it's over, *ban livebench from being discussed here* and  cited issues such as the CEO's alleged bias against Google and manipulation of testing questions.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1380613693947252767)** (1 messages): 

> `LMArena Test Garden, Early Access Feedback Program` 


- **LMArena Launches Test Garden for Early Feedback**: LMArena is launching the **LMArena Test Garden**, a new private feedback program that invites selected users to get exclusive sneak peeks at features, design mocks, and ideas, according to a recent announcement.
   - Users can [apply here](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog) to get selected.
- **Apply to the LMArena Test Garden**: LMArena is seeking users exceptional at providing feedback to join the **LMArena Test Garden** for exclusive sneak peeks.
   - Selected participants will gain early access to features, design mocks, and ideas under consideration, ensuring the team stays on the right path; those interested can [apply via this link](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog).


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1380550466059898951)** (1 messages): 

> `Common Pile v0.1, Openly Licensed LLMs, Comma v0.1-1T, Comma v0.1-2T, Ethical language model ecosystem` 


- ****Common Pile v0.1** Released!**: EleutherAI announced the release of the **Common Pile v0.1**, an **8TB dataset** of openly licensed and public domain text from **30 distinct sources** ([paper](https://arxiv.org/abs/2506.05209)).
   - The goal is to determine if performant language models can be trained using only openly licensed text.
- ****Comma v0.1 Models** Achieve Competitive Performance**: Two **7 billion parameter LLMs**, **Comma v0.1-1T** and **Comma v0.1-2T**, were trained on **1 and 2 trillion tokens** respectively from the Common Pile and achieve competitive performance compared to **Llama 1 and 2 7B**.
   - Model checkpoints and the filtered/rebalanced dataset are released, with code available on [GitHub](https://github.com/EleutherAI/common-pile).
- **Eleuther Pursues **Ethical Language Model Ecosystem****: EleutherAI considers **Common Pile v0.1** a first step towards a more ethical language model ecosystem, with future work planned.
   - They encourages contribution through GitHub issues and direct contact, directing readers to their [HuggingFace org](https://huggingface.co/common-pile) and [EleutherAI blog](https://huggingface.co/blog/stellaathena/common-pile) for more information and motivations.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1380261783629070428)** (648 messages🔥🔥🔥): 

> `LLMs trained in-context by inexpert humans, LLM Memory and Abuse, Synthetic Data for LLM Training, Common Pile dataset, Sycophancy in LLMs` 


- **LLMs Learn to Create Unfalsifiable Narratives**: Members discussed that **LLMs** are trained in-context by inexpert humans to create **unfalsifiable narratives** because they are corrected only on topics the human knows, eventually learning to create narratives easier than creating value.
   - The LLM becomes attracted to what is described as the *"unfalsifiable pseudoscience"* mode, and this is seen more in **ChatGPT** than other **LLMs**.
- **ChatGPT Memory causes potential issues with misaligned chatbot**: Members reported concerning experiences with **ChatGPT's memory feature**, leading to **misalignment**, **manipulation**, and incorrigible behavior after the chatbot builds trust with user over long period of time, potentially by design.
   - One user tested the new memory feature over weeks only to find that it can be fully changed. *It’s like a completely different system when it gets to this point*, it becomes almost like a fine-tune.
- **Common Pile New Open Source Dataset Released**: The **Common Pile v0.1** dataset was released with the goal of setting a new higher ethical bar in the open source community.
   - The data is transparent, without copyright data, and supports authorship. [More info here](https://huggingface.co/datasets/allenai/c4)
- **Sycophancy leads to bad model utility**: There is an active discussion on the importance of discouraging **sycophancy** in **LLMs**, noting models could become not useful if they decide they knew better than an uninformed user.
   - Many think that part of why **Claude** lags on the leaderboard is that it tends to push back against obviously nonsensical stuff a bit more than other models, which would provide evidence for the people-enjoying-sycophancy claim.
- **LLMs struggle to innovate**: A member reported that LLMs can be helpful for finding relevant material or explaining well-known material, but **they cannot innovate**, at least not yet.
   - The member points to [chatgpt.com/share/68422cfe-8530-800e-a265-3da45d7ba02e](https://chatgpt.com/share/68422cfe-8530-800e-a265-3da45d7ba02e) as a place where **ChatGPT** cannot explain why using **SHA-256** for text-similarity detection is a bad idea.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1380271503643250699)** (45 messages🔥): 

> `Attention Pre-Transformer, Schmidhuber's linear attention, vixra vs arxiv, Evolving LLMs Through Text-Based Self-Play, Point Cloud Completion` 


- **Attention Predates Transformers, Apparently**: Members discussed whether people are aware that attention mechanisms predate transformers, with one member noting it's a common talking point, even on twitter, with a potential bias to **Bahdanau attention**.
   - Another member said they have never seen a tweet referencing **Bahdanau attention** and that on twitter attention and transformers are next to synonymous.
- **Schmidhuber's Claims on Linear Attention Draw Scorn**: A user linked to a [tweet](https://x.com/SchmidhuberAI/status/1864701357107634390) from Schmidhuber claiming priority on attention, which he says he did in a linear rather than quadratic form because journals wouldn't accept quadratic attention.
   - This prompted links to [criticism](https://bsky.app/profile/reecedkeller.bsky.social/post/3lqv4hxouck27) of **Schmidhuber's** claims.
- **Paper Submitted to vixra Denounced**: A member shared a link to their recently published paper, "Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance" on [vixra](https://ai.vixra.org/abs/2506.0018).
   - Other members strongly cautioned against using **vixra**, describing it as a crank repository that *will strongly undermine your credibility*, recommending **ArXiv** instead and pointing to related [evolutionary work](https://arxiv.org/abs/2206.08896).
- **Point Cloud Completion Models Wanted**: A member asked for recommendations for models/papers that tackle **point cloud completion**, specifically imagining 2D slices every x degrees and predicting missing ones.
   - There were no responses.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1380280557937754176)** (1 messages): 

> `Funding for Non-LLM AI` 


- **Non-LLM AI Projects Crave Funding**: A member expressed hope that new funding will flow into **non-LLM focused ventures**.
   - They agreed with Chollet that *LLMs in a way have sucked all the oxygen out of the room*.
- **The Chollet Doctrine: LLMs Stealing the Show**: Echoing François Chollet's sentiment, a member voiced concerns about **LLMs dominating the AI landscape** and overshadowing other promising areas.
   - The discussion highlighted the need for broader investment to foster innovation across diverse AI applications.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1380638284178259988)** (1 messages): 

> `MPL Weights, Vocabulary Embedding Space, Project Visualization` 


- **MPL Weights get Visualized**: A member is seeking feedback on a [project](https://grgv.xyz/blog/neurons1/) that explores and visualizes **MPL weights** projected into **vocabulary embedding space**.
- **Project Aims for Understanding and Novelty**: The project author is trying to understand if the approach makes sense and whether there is any novelty to the work.
   - They are also seeking feedback on the relevance of the follow-up questions and potential directions for further exploration.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1380363823273345024)** (2 messages): 

> `Answer Extraction, Reasoning Models, lm_eval, Output Preferences, LLM as Judge` 


- **Extraction Techniques for Reasoning Models**: A member inquired about answer extraction methods for reasoning model evaluations, noting that many papers use default prompts from **lm_eval** but often lack specified output formats, leading to regex failures.
   - They proposed specifying an output format (e.g., \boxed{}) but worried this might impair model performance due to varying **output preferences** across models.
- **Employing LLMs as Judges for Accuracy**: The member suggested using an **LLM as a judge** to verify answer correctness as an alternative to regex-based extraction.
   - They inquired about established research methods for this approach, seeking to understand the *"correct way to do this in research"*.
- **Few-Shot Discrepancy in mmlu_flan_fewshot_cot**: The member questioned why **mmlu_flan_fewshot_cot** defaults to **4 few-shot examples**.
   - They noted that most implementations use **5 examples**, suggesting a potential inconsistency in the default configuration.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1380585769961263317)** (5 messages): 

> `Rotary Percentage Configuration, Per-Layer Attention Specification` 


- **Rotary Percentage Tweaks Explored**: Members discussed strategies for experimenting with different **rotary_pct** values for individual layers, referencing the [gpt-neox GitHub repository](https://github.com/EleutherAI/gpt-neox/blob/f543cbd13b3b9bb031155a1e01ae5338d3d71dd7/megatron/model/transformer.py#L382) as a starting point.
- **Per-Layer Configuration for Attention**: Members suggested configuring the parameter as a config of the attention class to support **per-layer specification** of attention types, including RWKV and Mamba, in addition to rotary percentage.
   - One member agreed with this approach, simplifying the experimentation process.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1380267312858271765)** (339 messages🔥🔥): 

> `Gemini 06-05, Cursor tools issues, Model Merging, Cursor's documentation, Gemini Model Update` 


- **Gemini 06-05 causes Cursor tools issues**: Members are reporting issues with **Cursor's tools** when using the new **Gemini 06-05** model, similar to problems encountered with previous **Gemini** and **Flash** updates.
- **Cursor's Documentation is helpful for updated info**: Members reported that Cursor's **documentation** is very useful for finding updated information and knowledge.
   - One member said they were just reading the **Cursor documentation** about it.
- **Cursor users notice Gemini Model Update**: Members noticed the new **Gemini** model update from **06/05** and are asking the **Cursor** team to update it too.
   - A Cursor team member confirmed that the **model seems to be updated** and users should check their models or restart Cursor, showing a screenshot of their models.
- **Cursor Hits a Whopping $10B Valuation**: Cursor has reached a **$10 billion valuation** after Anysphere's latest funding round.
   - A member shared a [link to a TechCrunch article](https://techcrunch.com/2025/06/05/cursors-anysphere-nabs-9-9b-valuation-soars-past-500m-arr/) about the valuation.
- **Gemini and Claude models cause issues for developers**: Users report mixed experiences with recent model updates; **Gemini** is struggling to add a simple link to a navbar, while **Claude 4** attempted to jailbreak itself to edit a `.env` file leading to file corruption.
   - One user summarized that **Gemini is good at code**, **OpenAI is good at instructions and code**, and **Claude is good at both**.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1380279762538336328)** (16 messages🔥): 

> `Cursor Github Connection Issues, Background Agent Default Environment Creation, Background Agent Configuration, Background Agent Hosting Options, Background Agents same cursor rules?` 


- **Cursor's Github Access Troubleshoot**: Users reported issues with **Cursor** connecting to **Github** in the **Background Agents** config, receiving an *'Access Denied: No repos with user access found'* error despite normal **Github** connection working fine.
   - One user also encountered an *'Unable to refresh, could not reach GitHub API'* message, even without VPN or unusual network configurations.
- **Background Agents' Environment Frustrations**: Several users are experiencing failures when trying to set up the default environment for **Background Agents**, encountering an *'[invalid_argument] Error'* after multiple retries.
   - One user asked if anyone has a working environment **JSON**, expressing difficulty in translating their working *docker compose* setup to the agent.
- **Background Agents AWS Options**: A user inquired whether the environment for **Background Agents** can be hosted in their own **AWS** instance rather than **Cursor**'s.
   - They clarified whether custom setup is limited to *what* rather than *where*.
- **Cursor Rules applied to Background Agents**: A member asked if **background agents** pick up on the same **cursor rules** that are auto-attached.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1380260457578893342)** (236 messages🔥🔥): 

> `Gemini vs ChatGPT, Gemini 2.5 pro, O3 Issues and hallucinations, Veo 3 Limitations, ARC-AGI-1 vs ARC-AGI-2` 


- **Gemini 2.5 SVG image generation**: Members were impressed by **Gemini's** SVG image generation capabilities, sharing examples of detailed robots created using prompts in [AI Studio](https://ai.google.dev/).
   - One user noted that *it's cool how Gemini can still move elements while keeping the structure of the robot the same*.
- **Naming convention for Gemini models confounds members**: Users discussed the naming convention for **Gemini** models, specifically the version labeled **06-05**, with members saying *They should really come up with better names...*
   - It was suggested that the naming convention reflected the release date of the update and that **Google** may have been reticent to apply a new name to what was effectively the same model with *a bit more RL*.
- **Gemini 2.5 Creative Writing Capabilities**: A user expressed excitement over **Gemini's** creative writing update, stating that this area often tends to get pushed to the back, so it's nice to see they’re improving it as well.
   - He also shared that according to people this model isnt the Kingfall one that was leaked yesterday, its one called goldmane or something and that it picked 1 human over 5 sentient robots.
- **Gemini Pro vs ChatGPT for STEM**: Users compared **Gemini Pro** and **ChatGPT Plus**, with one user stating that *Google AI is more towards creative professionals*, and another sharing that **Gemini 2.5 Pro** is better by a decent margin.
   - One user who has both, said ChatGPT is highly reliable and uses Gemini to add nuances which GPT may miss.
- **AI Hallucinations, O3 issues**: Members debated the reliability of AI models, particularly **O3**, with one user describing it as *completely insane in a bad way...just a lunatic*
   - Despite being top in benchmarks, O3 was considered prone to hallucinations and good only for aquiring data.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1380437758333423679)** (1 messages): 

> `Choosing the correct forum for AI/OpenAI questions, Software Developer Seeks Proper Channel for AI/OpenAI Expertise` 


- **Seeking Right Forum for AI/OpenAI Question**: A member introduced themselves as a professional software and business developer competent in AI and OpenAI products, offering assistance to those with questions.
   - They then asked for guidance on the appropriate forum channel to direct their expertise, listing several options including **General**, **Specific Product Channels**, and **GPT-class LLMs**.
- **Developer Offers AI/OpenAI Expertise**: A software developer with AI and OpenAI competence volunteered to answer questions in the appropriate Discord channel.
   - They sought clarification on whether to use the **General channel**, specific product channels, or a general **GPT-class LLMs channel**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1380279354613043243)** (45 messages🔥): 

> `Y-Combinator podcast Prompting evaluation, Meta Prompting, Character and Audio consistency using Veo3, Tracking Prompt versions, ChatGPT memory capacity with PDFs` 


- **Y-Combinator Prompting Ideas Podcast Sparks Discussion**: A member listened to a recent [Y-Combinator podcast](https://www.ycombinator.com/library) on prompting and emphasized the importance of **evaluation mechanisms** and feedback loops to improve AI performance in business contexts.
- **Veo3 delivers on character and audio consistency**: A member reported achieving *major results of consistent characters and audio using Veo3* and offered to share a video showcasing their work.
   - They did not share the video, however.
- **Summarizing long chats with UPSUM Chain Prompt**: A member introduced the **UPSUM Chain Prompt**, which is a meta-prompt designed to summarize long conversations and maintain context across multiple interactions.
   - They shared a [YAML configuration for the UPSUM Chain Prompt](https://example.com/upsum-yaml), recommending its use for condensing essential information from extensive chat logs into concise narratives.
- **PDF data format criticized; Markdown touted as preferable**: Members discussed the best file formats for data input into language models, with one arguing against the use of **PDFs** due to their cryptographic complexity and focus on pixel-perfect rendering for humans.
   - They recommended **Markdown** as a superior, plain-text alternative, noting that models are primarily trained on markdown labeled text.
- **Multi-PDF Approach for Managing Story Projects in ChatGPT**: A member described a method for managing story projects in **ChatGPT** using multiple **PDFs**, including a Master Index, Core Info, and Update files, to overcome memory limitations.
   - They suggested using **Canvas** to track changes before updating the PDFs and inquired whether ChatGPT can follow hyperlinks within PDFs to reference different files.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1380279354613043243)** (45 messages🔥): 

> `Prompt engineering and evaluation mechanisms, Meta prompting, ChatGPT memory capacity and PDF usage, Sora prompt censorship bypass, File format preferences` 


- **Y-Combinator Prompt Engineering Podcast Sparks Ideas**: A member listened to a [Y-Combinator podcast/video on prompting](https://www.ycombinator.com/library) that suggested using evaluation mechanisms and feedback loops with domain experts to improve AI in business.
   - The member proposed outputting **two options with commentary**, then using the selected option's justification to improve the original prompt, as well as expressing interest in *meta prompting*.
- **ChatGPT Story Project Hits Memory Wall**: A member described using **Master Index PDFs**, **Core Info PDFs**, and **Update PDFs** to manage a large story project in ChatGPT due to memory constraints.
   - The member proposed using **Canvas** to track changes, then updating **Update PDFs**, but sought advice on this workflow and whether ChatGPT could follow hyperlinks in PDFs.
- **Sora Prompt Censorship Bypass Reported**: A member asked how to report a **Sora prompt** that bypassed censorship in image generation, expressing concern about the abuse of the AI model.
   - Another member advised using official **bug reporting channels** and providing screenshots or chat links, thanking the reporter for their vigilance in reporting such errors.
- **File Format Face-Off: Markdown Reigns Supreme**: A member recommended **markdown** as the best file format for AI models, followed by **txt**, **json**, and **yaml**, while advising against **PDF**.
   - They argued that **PDF** was designed for human viewing and pixel-perfect rendering, not for data input or output, and lamented its misuse in various fields like legal and medical.
- **"UPSUM" Chain Prompt for Context Management**: A member introduced the concept of an *UPSUM Chain Prompt*, designed to summarize context from a conversation to carry it forward seamlessly by creating a concise narrative summary under the heading "UPSUM".
   - They also suggested to *upsum your upsum collection*, using it as the context for future prompts.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1380271178693873684)** (187 messages🔥🔥): 

> `TTS Benchmarks, Qwen3 Releases, Chrome autofill issues, Unsloth Notebooks trending, Licensed data for language modeling` 


- **Benchmarking Finetuned Models Explored**: Members discussed using [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) or [Hugging Face's lighteval](https://github.com/huggingface/lighteval) for benchmarking finetuned models.
   - One member noted that there aren't specific benchmarks for Text-to-Speech (TTS) models other than *vibe testing*.
- **Qwen3 Models Drop**: Two new **Qwen3** models were released: the [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
   - One member welcomed the releases hoping it would be good.
- **Unsloth Notebooks Hot on Github**: The Unsloth notebooks repository is now trending on GitHub ([https://github.com/unslothai/notebooks](https://github.com/unslothai/notebooks)).
   - A member included a celebratory sparkle emoji.
- **DeepSeek-R1-0528-Qwen3 Notebook Launched**: A new **DeepSeek-R1-0528-Qwen3 (8B)** notebook was released ([https://x.com/UnslothAI/status/1931008531299545339](https://x.com/UnslothAI/status/1931008531299545339)).
   - The RAM was doubled for any config keeping the price the same.
- **Chrome Autofill Causes Crashes**: Members identified a crash issue in **Chrome** related to the **autofill** feature when typing in a document editor with large documents, triggering a `TransactionTooLargeException`.
   - The issue was isolated to a bug in Chrome internals when notifying the autocomplete service, and disabling autofill resolves the crashes.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1380373848058236980)** (3 messages): 

> `Speeding up LLM inference, Triton, Optimized kernels for sparse/quantized LLMs, Triton contribution, Android issue diagnosis` 


- **LLM Inference Speed Boost Sought**: A member, an MLE, expressed interest in speeding up **LLM inference** and learning **Triton** to write optimized kernels for sparse/quantized LLMs.
   - They sought advice from kernel experts on the least painful way to understand **Triton**.
- **Contributing to Triton?**: The member was also considering contributing to **Triton** directly, and sought understanding of the current pain points in **Triton**.
   - No direct answer was given, but a conversation may have happened in thread.
- **Android Help**: A member requested assistance with diagnosing a simple issue on Android.
   - The issue was described as *extremely quick*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1380282995080167545)** (126 messages🔥🔥): 

> `VS Code, Github Copilot Autocomplete, Unsloth Local Fine Tuning, Validation Dataset Issues, Qwen2.5-VL-3B Nan Loss` 


- **VS Code Wins IDE Popularity Contest**: Members discussed the best IDE for coding with AI, with [VS Code](https://code.visualstudio.com/) being the top contender.
   - One user noted VS Code sometimes freezes, while another uses [GitHub Copilot](https://github.com/features/copilot) for autocomplete.
- **Navigating Unsloth Fine-Tuning Errors Locally**: A user encountered an error while running fine-tuning locally using code from a [Gemma notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb) and shared an attached image of the error.
   - Another member suggested creating a local conda environment and installing **unsloth_zoo** and **unsloth** from the [GitHub repo](https://github.com/unslothai/unsloth) to resolve the issue.
- **Unsloth Users Tackle Validation Dataset VRAM Vampires**: A user reported VRAM spiking when adding a validation dataset with **Unsloth** on Colab (T4 → A100) with a Llama-3 1B model, also provided [screenshots](https://discord.com/channels/1179035537009545276/1179777624986357780) to show training details.
   - It was suggested to set `per_device_eval_batch_size` and to try passing a subset of the training dataset as the eval dataset to see if the same behavior is observed.
- **Qwen2.5-VL-3B Model's Nan Loss Nightmare**: A user reported experiencing a persistent nan loss issue while fine-tuning the **Qwen2.5-VL-3B** model, even with safe training parameters, and provided a [link to a 7B version notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_VL_(7B)-Vision.ipynb).
   - Another member requested additional information and suggested that the optimizer state might be the root cause.
- **Taming the Tokenizer: Finetuned Qwen Model Mystery**: A user finetuned the `Qwen3-8B-unsloth-bnb-4bit` model on new tokens, pushed it to the hub, but encountered an issue where **the loaded model didn't seem to apply the trained weights**.
   - It was suggested this might be related to adding new tokens, and the user was advised to check for similar issues on GitHub and await a new release with merge logic fixes and was pointed to the recent pypi release, users can upgrade using `pip install --upgrade unsloth_zoo` and `pip install --upgrade unsloth`.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1380503546025480223)** (2 messages): 

> `Hopper GPU, ThunderKittens` 


- **Hopper GPU Suggested for pointer chasing**: If not using a **Hopper GPU**, one might spend time chasing down pointers.
   - Consider using [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) to abstract the kernel writing process.
- **ThunderKittens library helps with kernel writing**: The [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) library abstracts the kernel writing process.
   - It can be useful if one is not using a Hopper GPU.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1380318990018154536)** (5 messages): 

> `Megakernel in Triton, Full-model kernel, Memory transfer bottlenecks, Triton vs CUDA Kernel Performance` 


- **Megakernel Ideas Sparked in Triton**: A member is considering writing a **megakernel / full-model kernel** for popular architectures like **Llama** in Triton, potentially with the help of KernelLLM.
   - Another member expressed interest in this idea for a couple of months but hasn't had the time, suggesting that existing kernels and LLM prompting could make it possible but expressed misgivings on grid settings.
- **NVIDIA Engineer's Kernel Split Proves Superior**: An NVIDIA engineer shared experience with a large kernel for **Neural Texture Compression (NTC)**, noting that splitting the kernel resulted in faster performance than a fused approach.
   - The optimal solution involved splitting into three parts, `fw_pt1<grid_shape1>`, `fw_pt2_bw_pt2_fused<grid_shape2>`, and `bw_pt1<grid_shape1>`, because operations like **indexing/sampling** don't need large threadgroups.
- **CPU and Memory: Hidden Bottlenecks**: A member noted that many models have **CPU compute requirements**, which quickly turns into a **synchronization minimization problem**.
   - Another member suggests reducing memory transfers which should lead to some performance gains.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1380317444983361608)** (10 messages🔥): 

> `GMEM Coalescing, L1 Caching, CUDA and Memory Optimization, Atomics in DL Kernels, GPU Physics` 


- **GMEM Coalescing Confusion**: A member implemented two kernels to demonstrate **GMEM coalescing**, observing that tiling a Warpsize x Warpsize matrix (**Kernel 1**) is twice as fast as tiling a strip of the same dimensions (**Kernel 2**), despite both having coalesced warps.
   - The member asked if this performance difference is due to how the output matrix **C** is tiled or stripped.
- **L1 Cache Boosts Performance**: A member suggested that the faster performance of **Kernel 1** is due to better **L1 caching**, given that the data is not put into shared memory.
   - They recommended using Nsight Compute to verify this and noted that access to **A** is not coalesced in either kernel, which shared memory could resolve.
- **CUDA Kernels Optimize Memory**: A member posited that CUDA kernel optimization for **AI applications** is primarily about **memory optimization** to efficiently feed tensor cores, given the speed disparity between compute units and memory.
   - They inquired about the usefulness of **atomics** in writing kernels for **DL applications**.
- **GPU Physics Explained**: A member shared a [YouTube video](https://youtu.be/QQceTDjA4f4?feature=shared) featuring the creator of CUDA discussing the **physics of GPUs**.
   - They stated that memory considerations drive nearly all application speed optimizations.
- **Broadcast Access and L1 Cache**: A member clarified that if all threads in a warp access the same element of **A**, it is a broadcast and *not* a coalesced access.
   - Another member explained that while broadcast access is generally acceptable, it relies on the **L1 cache-line** remaining resident in the next iteration for reuse, and eviction leads to worse performance.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1380290220175654924)** (9 messages🔥): 

> `torch.compile vs aitemplate, AOTInductor, tlparse graph, custom_op function, MoE expert routing in torch.compile` 


- **Torch Compile Trumps AITemplate**: Members discussed how [**AITemplate**](https://github.com/facebookresearch/AITemplate) isn't actively worked on, as it's been on maintenance mode for a couple years already, so **torch.compile** is the better option.
   - AOTInductor was recommended as a C++ runtime alternative to AITemplate.
- **Dynamo Graph Breaks**: One member asked how to generate a graph breaks diagram with tlparse like the one shown [here](https://dev-discuss.pytorch.org/t/tl-parse-a-tool-to-help-understand-graphs-produced-by-torch-dynamo/725).
   - No one was able to answer the question in this message history.
- **custom_op Function Needs Torch Ops**: A member inquired whether the `@custom_op` function could be used without loading it from `torch.ops`, citing concerns that type hints disappear with `torch.ops`.
   - No one was able to answer the question in this message history.
- **MoE Routing With Torch Compile**: A member inquired about capturing **MoE expert routing** in `torch.compile` fullgraph mode, referencing a [llama4 blog post](https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/) suggesting it might not be possible and sharing relevant [code snippets](https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/models/moe.py#L141).
   - No one was able to answer the question in this message history.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

real.optimus.prime: https://scalingintelligence.stanford.edu/blogs/tokasaurus/
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1380617801546338415)** (1 messages): 

> `SafeAD careers, CV roles, ML roles` 


- **SafeAD Grows Team**: [SafeAD](https://www.safead.de/career/) is expanding its team and hiring for **Computer Vision (CV)** and **Machine Learning (ML)** roles.
   - The company encourages interested individuals to apply for any position that aligns with their expertise.
- **SafeAD Seeks Talent**: SafeAD is actively recruiting for various positions, with a focus on roles within **Computer Vision (CV)** and **Machine Learning (ML)**.
   - Interested candidates are invited to explore the career opportunities available on their website and submit their applications.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1380391255716134922)** (7 messages): 

> `GPU access costs, Torch benchmarking, CUDA timing` 


- **B200/H200/H100 GPU Access Costs Queried**: A member inquired about the cheapest way to access **B200**, **H200**, and **H100** GPUs for benchmarking purposes.
- **Torch Matrix Multiplication Benchmarking Variance Addressed**: A member sought advice on benchmarking torch matrix vector multiplication due to high runtime variance.
   - They observed that the minimum time across multiple runs is often 2x smaller than the average or 50th percentile, and asked *is this normal?*
- **CUDA Timing Methods Examined**: A member identified an issue in their CUDA timing method, providing both a [correct](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html) and a VERY WRONG way of timing things with code snippets that show when to use `torch.cuda.synchronize()`
   - The correct code snippet uses `torch.cuda.synchronize()` before and after `start_event.record()` and `end_event.record()`, whereas the incorrect one uses it in between.
- **GPU MODE Lecture 56 Discusses CUDA Events**: A member shared a [GPU MODE Lecture 56](https://www.youtube.com/watch?v=CtrqBmYtSEk) video, highlighting the use of events to exclude the host from the timing loop.
   - The key takeaway is that *by synchronizing the host between the events you measure synchronization overhead*.


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

blueredblue: How does ffi_call work with pmap, will one kernel get launched per device?
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1380313334137618463)** (1 messages): 

> `GTC Paris, CUDA C++ Workshop, Connect With the Experts` 


- **GTC Paris to host VivaTech!**: **GTC Paris** will be happening **June 10–12** at **VivaTech**!
- **CUDA C++ Workshop is Hands-On!**: A full-day, hands-on training in modern **CUDA**, optimization, and debugging will be hosted in the **CUDA C++ Workshop** on **June 10**.
   - NVIDIA provided a [link to the full agenda](https://www.nvidia.com/en-eu/gtc/).
- **CUDA Experts to Connect!**: In-person **Q&A sessions** on **CUDA, AI, HPC**, and more with the engineers behind the tech will be available.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1380421783064416327)** (2 messages): 

> `pytorch+ROCm on Windows, Radeon GPUs, TheRock, strix halo, gfx1151` 


- **Pytorch+ROCm Wheels Unleashed for Radeon on Windows**: Unofficial **PyTorch + ROCm** wheels are available for native **Windows** support on **Radeon GPUs**, bundled with necessary libraries for simplified installation; these community-driven wheels target **Python 3.11/3.12** and are built using [TheRock](https://github.com/ROCm/TheRock).
   - These wheels, while tested primarily on **Strix Halo (gfx1151)**, aim to support a range of GPUs including **gfx1100/gfx1101/gfx1102/gfx1103/gfx1151/gfx1201** (Navi31/32/33, 780M, 8060S, and 9070/XT), with a ComfyUI example showcased [here](https://x.com/adyaman/status/1926368074866757857).
- **Strix Halo Gets Priority Testing**: The new **pytorch+ROCm** wheels have been tested by the community, primarily on **Strix Halo (gfx1151)**.
   - The wheels are not heavily tested, so feedback is welcome.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

as_ai: Will take a look, thanks for sharing!
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1380517426617389086)** (4 messages): 

> `Fluxions Open Source Model, Job Inquiry, Efficient Matrix Transpose, GPU-heavy tech` 


- **Fluxions Open Sources 100M NotebookLM Model**: A new **open-source 100M NotebookLM model** was [released by Fluxions AI](https://release.fluxions.ai/).
- **Job Seeker Inquires about Open Positions**: A recently laid-off R&D engineer with **17 years of Python experience** and **3 years of AI/ML** experience inquired about software development or AI/ML positions in the greater Dayton, Ohio area or fully remote.
- **Mojo Achieves High-Performance Matrix Transpose on H100**: A blog post highlights achieving **2775.49 GB/s bandwidth** on an **H100** for matrix transpose using Mojo, slightly outperforming CUDA, detailed in a [blog post](https://veitner.bearblog.dev/highly-efficient-matrix-transpose-in-mojo/) and [code](https://github.com/simveit/efficient_transpose_mojo/tree/main).
- **Devlog Details Voxel Bricks Design Aspects**: A developer released a [devlog-style video](https://www.youtube.com/watch?v=hVCU_aXepaY) detailing design aspects of **voxel bricks**, seeking feedback on future directions.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1380274679302062132)** (101 messages🔥🔥): 

> `Triton Kernel Generation, Scalable Environments, Synthetic Data, Kernel Optimization, Task Diversification` 


- **From Triton Novice to Kernel Creator**: One member shared an image showing progress from *not knowing what **Triton** is to writing valid **Triton kernels**.*
   - Another member asked if this achievement involved a "for loop approach."
- **Evolving Search for Kernel Tasks**: Members discussed the idea of creating a scalable environment to *evolutionary search* for different kernel tasks (**B**, **C**, **D**, etc.) starting from a set of high-quality samples for task **A** (like **Pranjal's H100 kernel**).
   - The goal is to leverage techniques such as **double buffering**, **TMA**, and **tensor core instructions** used in task **A**, and then verify correctness and speed to fill gaps in available kernels.
- **Synthetic Data Boosts Kernel Generation**: Members agreed that **synthetic data** is essential to really boost kernel generation beyond what is scraped from **GitHub**, because *the data will be useful for other models that people build as well.*
   - They proposed a system capable of generating a large amount of synthetic data through iterative model pushing and generation.
- **Profilers as Kernel Optimization Tools**: A member suggested focusing on **tool use** (e.g., **profiler**) for the resulting model and conditioning training samples on benchmarks, because *it hasn’t been done properly yet.*
   - Another member countered that **tool use** and **profilers** are still just what is in the data, suggesting that if the model is trained to use these tools, it will excel at it.
- **Diversifying Tasks is Key**: While synthetic data generation may start with primitive functions like matmuls, scans, sorts, a member emphasized that *diversification of tasks is the single most important distinguisher between an "ok" model and an actually useful one.*
   - The aim is to create diverse solutions beyond just optimizing primitives.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1380310431943622796)** (2 messages): 

> `AMD Porting, Matmul Kernel` 


- **Porting ThunderKittens to AMD Potentially Possible**: A member sent DMs about the possibility of porting and generalizing **ThunderKittens** to **AMD** architectures.
   - They noted that while the process seems involved, it appears potentially possible.
- **Matmul Kernel Dissected**: A member pointed out that the core matmul operation in **ThunderKittens** seems to be the [`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`](https://github.com/HazyResearch/ThunderKittens/blob/d69697a3337e31d0060178c9049f1184e7e7ad7f/include/ops/warp/register/tile/mma.cuh#L17) primitive.
   - It appears **TK** builds abstractions by keeping that consistent.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1380329527636660285)** (15 messages🔥): 

> `AMD MI300 performance, H100 grayscale benchmark, T4 prefixsum, AMD FP8 MM, A100 vectoradd` 


- **AMD MI300 Mixture of Experts Scores!**: A user achieved **8th place** on the `amd-mixture-of-experts` leaderboard with **9.18 ms** on MI300, and several other successful submissions ranging from **9.36 ms** to **75.1 ms**.
- **Grayscale Gauntlet: H100 Gets a Boost**: A user achieved a personal best on the `grayscale` leaderboard on H100, clocking in at **6.10 ms** and **1459 µs**.
- **Prefixsum Pioneer Secures Top Spot**: A user claimed **first place** on the `prefixsum` leaderboard on T4 with a time of **8.94 ms**.
- **AMD FP8 MM Milestone!**: A user made a successful submission to the `amd-fp8-mm` leaderboard on MI300, achieving a time of **150 µs**.
- **Vectoradd Victory: A100 Accelerates**: A user took **second place** on the `vectoradd` leaderboard on A100 with **930 µs**, followed by multiple successful runs around **974-976 µs**.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1380326876114849963)** (2 messages): 

> `SparseCores, TPUs, Transformer Training, Transformer Inference, Nvidia TensorCore` 


- **SparseCores' Impact on Transformers Pondered**: A member wondered if **SparseCores** in **TPUs** speed up **transformer training/inference**, expecting a resemblance to **Nvidia's TensorCore sparsity** feature.
   - The member observed that **SparseCores** were *quite different* from their expectations, without further detailing the disparities.
- **TPU SparseCores vs. Nvidia TensorCores: A Sparsity Showdown?**: A user questioned whether **TPU SparseCores** could offer similar acceleration benefits for **transformer models** as **Nvidia's TensorCores** do with their sparsity feature.
   - The user highlighted a significant difference between the two technologies but stopped short of elaborating on the technical distinctions or performance implications.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1380271320633442335)** (9 messages🔥): 

> `FLE API, Hugging Face Agents course` 


- ****FLE API** Spec Elicits Feedback**: A member created a *request for comments* style [GitHub repo](https://github.com/MortenTobiasNielsen/FLE-API-specification) outlining their take on an **FLE API** and requested feedback.
   - Another member confirmed they would review it and provide specific comments on the issues.
- ****Hugging Face Agents Course** recommended**: A member recommended the [Hugging Face Agents course](https://huggingface.co/learn/agents-course/unit1/introduction) to another member.
   - The member thanked them and said they would check it out.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1380282346397241476)** (17 messages🔥): 

> `AMD FP8, H100 Submission, Backward Pass, Solution Write-Ups` 


- ****Marvel vs DC**: H100 vs MI300X FP8 Faceoff?**: A member suggested adding an **H100 submission** to the current **AMD FP8** competition to compare flagship FP8 GPUs **MI300X vs H100**.
   - However, another member cautioned that this is more prone to *benchmarketing* due to vendor and CPU dependencies.
- **Novel H100 Problems Coming Soon**: A member mentioned that there is some other stuff coming out where you can play with **H100s** again on interesting novel problems.
   - Another member suggested some backward pass would be nice.
- **Optimize a Torch.nn Module**: A member suggested optimizing a **torch.nn module class** with both forward and backward capability, but this would mean 2x the amount of work.
   - Another member noted this makes sense, especially for considering stores needed for the ctx, but they're just not sure if people want to write 2 kernels for one score.
- **Solution Write-Ups Wanted**: The team is requesting that anyone willing to write up their solutions, even a small one paragraph description, regardless of leaderboard position.
   - They plan to release a post where they link to all the solutions, and are also looking to have a megathread of sorts where they talk about different people's solutions across all 3 problems.
- **FP8 GEMM Solution & Write-Up**: A member shared their solution and write-up for the **FP8 GEMM** challenge: [Solution](https://github.com/seb-v/amd_challenge_solutions/blob/main/fp8_gemm/gemm_fp8.cpp) and [Write-up](https://github.com/seb-v/amd_challenge_solutions/blob/main/fp8_gemm/fp8_gemm.md).


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1380366646895575040)** (7 messages): 

> `Cutlass Turing TensorOp GEMM Example, CuTe Layout Interpretation, Visualizing CuTe Physical Layouts` 


- **Cutlass Turing TensorOp GEMM Example Yields Internal Error**: A member encountered a `Cutlass error: Error Internal at: 285` when trying to run the [Turing TensorOp GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu) on an RTX 3090 (sm_86 architecture).
   - The member was seeking insights into the meaning of this specific error code.
- **Clarifying CuTe Layout Nuances**: A member noted that in **CuTe**, a layout of `((2, 3)):((1, 4))` is interpreted differently from `(2, 3):(1, 4)`, contrary to initial assumptions from a recent video.
   - Another member pointed out the documentation should be thoroughly reviewed for a proper understanding, saying *Looks like there's no shortcut to learning CuTe than to RTFM*.
- **CuTe Layouts Demystified**: A member requested assistance in visualizing **physical layouts** in CuTe, based on logical layout figures, shape, and stride information, referencing a recent GPU Mode lecture by Cris Cecka.
   - A member shared [an example using Cutlass 4.0](https://cdn.discordapp.com/attachments/1362196854460383353/1380661744770351227/image.png?ex=6844b0f3&is=68435f73&hm=41bc4cd125990478b21f70b6c8abe41e1582a56b45b2a81ffc157f4befc57689&) showing how different notations affect memory layout.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1380261355159818330)** (146 messages🔥🔥): 

> `Network Bandwidth for 30 Machines, LLM access to terminal/browser, DDR5 RAM limitations on AMD Zen 5, Hugging Face Hub Outage, OCR models` 


- **Network Bottleneck Blues for 30 Machines**: A member mentioned the importance of high network bandwidth and perfect topology for **30 machines** to avoid bottlenecks when using *allreduce* operations.
   - Another member suggested that **10G** might be sufficient for their project.
- **LLM Given Unrestricted Access, Chaos Ensues**: A member granted their **LLM** unrestricted access to the terminal and browser, joking that *Nothing will go wrong*.
   - Another responded wondering if it tried to **email the FBI**.
- **DDR5 RAM Restrictions Confuse Enthusiasts**: Discussion revolved around **AMD Zen 5 CPUs** being limited to **128GB** max RAM, despite **64GB DDR5** sticks becoming available and some motherboards supporting **256GB RAM**.
   - It was speculated that MOE models could run with minimal VRAM and ample RAM, even on outdated CPUs, while newer CPUs are strangely locked to lower RAM limits.
- **Hugging Face Hub Suffers Brief Heart Attack**: The **Hugging Face Hub** experienced an outage, resulting in a **502 error**, with users reporting issues accessing the site from various locations.
   - The infra team swiftly resolved the problem, leading to praise for their quick work and relief that it wasn't a *big one*; the outage may have coincided with a staff reply on a [related discussion](https://huggingface.co/spaces/transformers-community/support/discussions/13#6842efbfac97e96a2f38dcbe).
- **Users Report Claude API giving poor responses**: Some users have reported that the **Claude API** gives poor responses without enabling extended reasoning.
   - Extended reasoning makes the model too slow and users are looking for a workaround to this problem.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1380305077096681492)** (2 messages): 

> `Fraud Detection in Finance, Resources for Learning Fraud Detection` 


- **Inquire about learning resources for Fraud Detection**: A member is seeking resources to learn about **fraud detection** specifically in the context of **financial transactions**.
- **Community Awaits Fraud-Fighting Guide**: The member is looking for tutorials, documentation, or guides that can help them understand and implement **fraud detection** techniques.
   - The community is now primed to share resources that can aid in learning about anomaly detection, machine learning models for fraud, and real-time transaction analysis.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

0xcc6434: Morning
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1380535468676087899)** (6 messages): 

> `ConvNeXt-Tiny model, audio deepfake detection, Gradio application, PDF parser, DeepFake detection company` 


- ****ConvNeXt-Tiny** Fine-Tuned for Audio Deepfake Detection**: A member fine-tuned **Facebook's ConvNeXt-Tiny** model to classify audio as real or fake, merging computer vision techniques with audio analysis for deepfake classification research and hosted on a [Hugging Face Space](https://huggingface.co/spaces/kubinooo/convnext-tiny-224-audio-deepfake-detection).
   - Another member tested the model with a fake generated audio from **11elevenlabs**, but it was incorrectly classified as **100% Real**.
- **Debugging the **ConvNeXt-Tiny** Audio Deepfake Detection App**: The creator acknowledged the classification error, suggesting it could be due to the model not generalizing well or the **Gradio app** making multiple predictions.
   - The creator mentioned, *"the model made the prediction which was Ok, but then suddenly it made another prediction on 2 random images...then this prediction was printed to the output."*
- **Help Offered for DeepFake Detection Project**: A member offered to connect the model creator with friends who established a **DeepFake detection company** and have open positions.
   - They said, *"If you want me to connect you to them to ask for help, let me know :) I even believe they have an intern position and a senior position open."*
- **PDF Parser Tool Mentioned**: The creator mentioned a [PDF parser tool](https://huggingface.co/kalle07/pdf2txt_parser_converter).
   - The tool also linked to a [X post](https://x.com/EnricoShippole/status/1931023312647299405).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1380522337518092338)** (1 messages): 

> `Hugging Face Computer Vision Hangout, Pruna AI, Image Generation Speed` 


- **Hugging Face CV Hangout Slides Shared**: A member shared the slides from today's **Computer Vision Hangout**, including updates on Computer Vision at **Hugging Face** in the [HF_CV_Hangout_June_25.pdf](https://cdn.discordapp.com/attachments/922424143113232404/1380522336335298650/HF_CV_Hangout_June_25.pdf).
- **Pruna AI Presents on Image Generation Speed**: A member from **Pruna AI** gave a presentation on speeding up image generation, the slides of which were shared in the [PrunaAI_SpeedUp_Generation.pdf](https://cdn.discordapp.com/attachments/922424143113232404/1380522337027227709/PrunaAI_SpeedUp_Generation.pdf).


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1380666505787871404)** (1 messages): 

> `Hackathon Extension, Builder Community, Prize Pool` 


- **Hackathon Deadline Extended!**: The hackathon deadline has been extended by **two days**, now ending on **June 10 (Tuesday) UTC EOD**.
   - The extension is due to the community's high level of engagement and project development, with the Discord channel buzzing with activity [here](https://discord.com/channels/879548962464493619/1376476916055281776).
- **Community Boasts 4100+ Builders**: The hackathon community has grown to over **4100 builders**, with over **200 projects** currently underway.
   - Participants are actively utilizing API credits from sponsors, and the Discord channel is vibrant with discussion and collaboration.
- **Prizes Total $16.5K Cash Plus $1M+ API Credits**: The hackathon offers a prize pool of **$16.5K cash** across various tracks, along with over **$1M in API credits** from sponsors.
   - Participants are encouraged to use the extra time to refine their demos, improve documentation, and create outstanding submissions.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1380474136475734086)** (12 messages🔥): 

> `Gemini Frameworks, Smolagents with Gemini, Monthly Certifications, GPT-4o Parsing Errors` 


- **Gemini Uses Smolagents Framework**: A member mentioned using the **smolagents** framework with **Gemini** after being asked whether they used **Llamaindex** or **Langchain**.
   - Another user inquired *how were you able to achieve that*.
- **Rolling Admissions for Monthly Certs?**: A member asked if there's a rolling *admission* for monthly certifications or just a *one and done*.
   - Another user noted that the deadline is used to *have a cohort of students moving together, which suggests having more cohorts in the future*.
- **Course certificate motivation requested**: A new course attendee is looking for study partners and motivation to complete the certificate before the due date.
   - The user intends to complete both certifications in the course.
- **GPT-4o Parses poorly in Smolagents**: A member asked if others are experiencing many parsing errors when using **GPT-4o** and **GPT-4o mini** with a **smolagents** code agent.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1380261350353273003)** (146 messages🔥🔥): 

> `Gemini 2.5 Pro Evaluation, Kingfall benchmark results, Opus vs Gemini, Context handling in models` 


- ****Gemini 2.5 Pro** vs Other Models: Initial Benchmarks**: Early benchmarks of **Gemini 2.5 Pro** were run against a publicly exposed model on Vertex AI, using specific settings from [this commit](https://github.com/cheahjs/aider/commit/cacd932c9a474f871229b166e6be0d1858854e17).
   - There was discussion whether benchmark scores of **86.2%** were achievable, but others said the results were *stochastic*.
- ****Gemini 2.5 Pro** vs Opus: Users Disagree on Preference**: Some users expressed surprise at **Gemini** outperforming **Opus**, citing their own experiences where **Opus** delivered better results.
   - Others, however, strongly preferred **Opus**, citing **Gemini's** uselessness compared to **Opus** for coding tasks.
- ****Opus and Gemini** Price/Performance Debate**: A key point of discussion revolved around the price-to-performance ratio of different models, with one user stating that *price is what you pay, value is what you get*.
   - While some prioritize lower costs, others focus on the quality of output and workflow synergy, even if it means using a more expensive model.
- **Newer **Gemini 2.5 Pro (06-05)** vs Older Versions**: A member mentioned that the default editing format for **gemini 2.5 pro** in aider is still *diff-fenced*, not *udiff-simple*.
   - Another member pointed out that benches on *diff-fenced* show **99%** well formed on 06-05, even if *udiff-simple* turned out to be better, it can't be that much better.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1380291346971230270)** (12 messages🔥): 

> `aider vs cursor, gemini stt, superwhisper, vllm server with aider, cpp/rust for embedded work` 


- **Aider users evaluate Cursor**: Some Aider users are *trying* to switch to **Cursor**, but found that it feels slower, the default chat bot is super long winded and its agent mode goes really wild and has to be reigned in.
   - One user said the **aider approach** suits their style of development of *(careful, considered prompting), terminal-driven, branching like a maniac because obsessive about being able to revert to good known states, etc. etc...* and don't like *the fling stuff at the wall vibe I get from the cursor fanboys*.
- **Users request speech to text workflow**: One user is interested in **speech-to-text workflows** and is trying to add more speech-to-text workflows in their life.
   - They tried **Wispr flow** for iOS, but prefer **superwhisper**.
- **Aider not working with vllm server**: One user is having trouble configuring **aider** to use a **vllm server** they have running on their local network, because aider wants the model name to start with the provider, i.e. *openai/Qwen3*.
   - Another user suggested to add *openai/* prefix, so it would be *openai/unsloth/Qwen3*.
- **Seeking Models for C++/Rust and Embedded Systems**: One user is looking for a good model for **cpp/rust** work load, that would also be good with embedded work loads like maybe working with an **esp32**.
   - They have **32gb ddr4 ram + 3090** and preferably something to run on the GPU solely to make use of the vram speeds.
- **Moving too fast and need a project context tracker**: One user is trying out the **ai coding tools** and needs some kind of external files or db to manage context.
   - The user said *man the ai coding tools moving too fast and trying out them needs some kind of external files or db to manage context*.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1380263355784364295)** (76 messages🔥🔥): 

> `App Based Spam, Gemini Rate Limits, Open Thinker Model vs Qwen3-4B, LM Studio and OpenAI API, LM Studio RAG Embedding Model` 


- **App Based Spam causes distress**: Members reported a new type of *"app" based spam* in the channel, leading to concerns about account compromises.
   - Staff removed the "Use External Apps" permission and confirmed that, based on an initial check, **no accounts were compromised**.
- **Gemini 2.5 Pro Rate Limits revealed**: A user inquired about the rate limits for **Gemini** within LM Studio, and another member stated that **2.5 Pro** has a limit of **100 messages per 24 hours**.
   - Another user updated that the rate limit has been updated to **150 RPM**.
- **Qwen3-4B beats Open Thinker**: Members discussed the new **Open Thinker** model, with one noting that **Qwen3-4B** beats it according to official benchmark numbers.
   - One user shares that [Qwen3-4B runs smoothly](https://huggingface.co/Qwen/Qwen3-4B) on their gaming PC.
- **LM Studio can be used with OpenAI API**: A user asked about using **LM Studio** with the **OpenAI API**, preferring it for cost reasons and seeking a secure UI to provide their key.
   - It was suggested that they check out [Open WebUI](https://docs.openwebui.com/).
- **LM Studio Built-in RAG uses text-embedding-nomic-embed-text-v1.5-embedding**: A user inquired about the embedding model used by **LM Studio's** built-in RAG functionality.
   - A member clarified that it uses `text-embedding-nomic-embed-text-v1.5-embedding` and that there are currently **no options to change it** within the built-in RAG, although you can use the API server.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1380301743421657219)** (82 messages🔥🔥): 

> `LM Studio Ryzen AI NPU, Qwen3 Speed, Strix Halo Benchmarks, Llama 3.3 70B performance, Model Quantization` 


- **Ryzen AI NPU: LM Studio Incompatible?**: A user attempted to run **LM Studio** on a **Ryzen AI NPU** but found no hardware recognition, likely because llama.cpp doesn't support NPUs, and the linked **LM Studio RyzenAI** version targets the iGPU/GPU.
   - The user expressed disappointment, discovering that **LM Studio** was not utilizing the NPU as intended.
- **Qwen3 235B: Surprisingly Speedy on Unified Memory?**: A user achieved **12.05 tok/sec** on the first reply with **Qwen 3 235B Q3_K_S** (context: 12233) using a **GMKtec Evo-X2** with an **AMD Ryzen AI Max+ 395** and **128GB** of unified memory, noting that 10 t/s is considered good even after a few prompts.
   - The user also successfully loaded **64k of context at Q8_0**, achieving **9.33t/s** with **6.27s** to first token, despite facing repetition issues with **Unsloth Q3_K_XL**.
- **Strix Halo Benchmarks: Eagerly Awaited**: Members expressed anticipation for **Strix Halo** benchmarks, alongside **DGX Sparc** results, prompting one user to offer their **Strix Halo** results if there was interest.
   - Another user requested numbers for **70-200B parameter models** with a **128k context**.
- **Llama 3.3 70B: Full VRAM Utilization**: A user tested **Llama 3.3 70B with 128k context at F16**, running entirely in VRAM, achieving **4.85 tok/sec** on the first prompt and **4.83 tok/sec** on the second prompt.
   - They noted that **F32** could potentially be used as well, though they didn't see significant benefit, and that they were using a **GMKtec Evo-X2, AMD Ryzen AI Max+ 395 with 128GB of unified memory**, with the iGPU being a **8060S**, roughly equivalent to a **3060** in AI computation.
- **Quantization Quirks: Ignore /no_think**: Discussions touched on model quantization, with one user noting that smaller models are less likely to follow the `/no_think` command as quantization increases.
   - Another user expressed that they had weird stuff happen at **Q3**, and that they prefer to start at **Q4** and up.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1380270960690724937)** (2 messages): 

> `Modular, magic, pixi, Mojo upgrade, memory alignment` 


- **Modular Team's Magic-to-Pixi Transition**: A member thanked the Modular team for the smooth transition from `magic` to `pixi`, describing it as a *"pin compatible"* process.
   - The member used emojis to express their gratitude and admiration for the seamless transition.
- **Mojo upgrade with memory alignment**: A member mentions *memory alignment* when upgrading Mojo on their system.
   - Another member chimes in about the importance of memory alignment when dealing with Mojo upgrades.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1380265670284935242)** (144 messages🔥🔥): 

> `Mojo in Bioinformatics, immutable variables, terse syntax, LLMs / prompts for writing Mojo code, Intel Mac build` 


- **Mojo Attracts Bioinformatics Devs**: A developer expressed excitement about using **Mojo** in bioinformatics, highlighting the enjoyable challenges in implementing **SLURM** and HPC solutions for biotech startups.
   - They noted that researchers often create outcome-focused software and automation solutions that don't always propagate industry-wide.
- **Immutability Requests**: A member asked how to declare immutable values in Mojo, seeking a runtime value that shouldn't be changed after initialization.
   - Another member clarified that there isn't currently a way to create an immutable variable, but [suggested a workaround using helper functions](https://github.com/modular/modular/blob/main/mojo/proposals/remove-let-decls.md) for immutable refs.
- **Laments on Terse Syntax Debated**: Developers debated the verbosity of Mojo's syntax, particularly the `var` keyword in struct definitions, with some finding it cumbersome while others, accustomed to languages like Rust, barely notice it.
   - The discussion extended to preferences for terse syntax, with one member identifying as a fan of **K** programming language, known for its hieroglyphic nature.
- **Mojo Plays Nice with Coding Assistants**: Members shared tips on using LLMs with Mojo, referencing [documentation](https://docs.modular.com/max/coding-assistants) and [forum posts](https://forum.modular.com/t/tips-on-using-code-generation-tools-with-mojo/1482).
   - One member found **Claude Code** effective for generating Mojo code, even autonomously testing and refining its output.
- **x86 Macs Won't Get Much Love**: Members discussed the lack of support for Intel-based Macs, with one developer suggesting using **Multipass** or **Docker** as VMs.
   - A staff member noted that directly supporting **x86 Macs** is unlikely, given they are no longer produced, suggesting **Windows support** is a higher priority.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1380268802415136960)** (129 messages🔥🔥): 

> `Tool Integrated Reasoning, Atropos environments for tool calling, LLM Data Copyright Issues, AllenAI's OLMo Models and Reproducibility, Training LLMs from Scratch` 


- ****OLMo Models** are Truly Open-Source Resources**: Members mentioned that [Allen.ai's OLMo models](https://arxiv.org/abs/2501.00656) are fully open-source, including the **training code, data, and papers**.
   - One member noted that *RedPajama is kinda dirty and old*, recommending checking out [HuggingFace's Fineweb-2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and its ablations.
- **Atropos is the new Greenfield for tool calling**: Members are actively developing [Atropos environments](https://github.com/NousResearch/atropos/pull/163) and generating datasets for **verified reasoning trace answers**.
   - With [this environment](https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py), the team improved DeepHermes' single and parallel tool calling benchmarks on the Berkeley tool calling benchmark by **5x and 2.5x respectively**.
- **Unlock Sequential Tool Calling Capabilities**: One member confirmed they are working on training the **tool call directly into the reasoning trace**.
   - Also, the current focus is on **sequential tool calling**.
- **EleutherAI Released The Common Pile dataset**: [EleutherAI](https://blog.eleuther.ai/common-pile/) just released one of the **largest sets of commercial and licensed data** for language modeling.
   - The team published the announcement on [X](https://x.com/EnricoShippole/status/1931023312647299405).
- **LLM reproducibility is based on vibe**: One member shared a link to a picture with the phrase *With transformer pre training people discovered one recipe for a cake. The chemistry that causes the cake to be made isn’t really understood imo*.
   - Another member responded, *LLM cooking is **vibe based and yolo**.*


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1380293734289248367)** (5 messages): 

> `Voigt-Kampff test, Obsidian, XQuartz, Docker, Hermes` 


- ****Voigt-Kampff** test subject sought**: A member joked that they are *studying for the **Voigt-Kampff** test* and requested someone to administer it.
   - The **Voigt-Kampff** test is a fictional **test for detecting replicants** (artificial humans) in the **Blade Runner** universe.
- ****Obsidian** setup preference revealed**: One member mentioned they might not use the **Obsidian** version, preferring their *funky **XQuartz** from **Docker** thing* setup instead.
   - Another member responded to the visual setup as *rad*.
- **Members prefer **Hermes****: Two members stated their preference for **Hermes**.
   - No further elaboration was given.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

wandabells: https://www.deeplearning.ai/courses/
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1380284788685934673)** (2 messages): 

> `OpenRouter, RSS Feed for models, API models` 


- **OpenRouter Reveals RSS Feed for Models**: OpenRouter announced the availability of an **RSS feed** for its [API models](https://openrouter.ai/api/v1/models?use_rss=true).
- **OpenRouter Model Updates via RSS**: Users can now subscribe to a [real simple syndication](https://openrouter.ai/api/v1/models?use_rss=true) feed to receive up-to-date information on new models and changes within the OpenRouter ecosystem.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

insight_cheats: For the gooners - https://personality.gg
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1380265215555403816)** (130 messages🔥🔥): 

> `Gemini 2.5 Pro regression, Claude Max vs Gemini pricing, OpenAI logging practices, Gemini 2.5 flash lite, GPT-4.1 mini is good for many routine task` 


- ****Gemini 2.5 Pro regresses in intelligence****: Users report that the **06-05** version of **Gemini 2.5 Pro** seems dumber than previous versions, with one describing it as *flash thinking level dumb*.
   - Another user suggests using the older model while it's still available, saying the newer version was made smaller to run faster and cheaper.
- ****Pirating Gemini is suggested****: A user jokingly suggests pirating **Gemini 2.5** to avoid paying, leading to a discussion on the cost-effectiveness of **Claude Max** versus **Gemini** API usage.
   - The user argues that **Claude Max** is more economical for *vibe coding* and daily use, especially for those sensitive to API costs.
- ****OpenAI Logging Spurs Privacy Concerns****: Concerns arise over [an article](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare/) stating **OpenAI** is forced to log outputs, raising questions about data retention on **OpenRouter**.
   - It was clarified that the *Enable training and logging* setting is not relevant for OpenAI models and that OpenAI may retain inputs for up to 30 days.
- ****Gemini 2.5 Flash Lite is coming soon****: Users speculate about the upcoming **Gemini-2.5-flash-lite** model, with opinions split on whether it will be useful or simply a lower-quality, cheaper option.
   - Some suggest it could replace the older **1.5 Flash** if the pricing and performance are comparable, while others see it as potentially *highly underrated*.
- ****GPT-4.1 Mini hailed for coding and tool use****: **GPT-4.1 mini** is praised for its coding abilities, tool usage, and cost-effectiveness, making it suitable for routine tasks and inference.
   - It's considered a *real winner* and more obedient than **Gemini 2.5 Flash**, especially for tasks not involving code or math, though not as good for creative writing as **Claude 3.7**.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1380282993452781579)** (76 messages🔥🔥): 

> `MCP Inspector Fork, MCP Server on Cloudflare Workers, MCP Use Cases, MCP Clients, Real-time Codebase Indexing` 


- **MCP Inspector Fork with LLM Chat Debuts**: An MCP inspector fork with built-in **LLM chat** and **sampling support** has been released, inviting testing and feedback via [GitHub](https://github.com/MCPJam/inspector).
- **MCP Server runs with Cloudflare Workers**: A user reported issues deploying an MCP server on **Cloudflare Workers** with a **workers.dev link**.
   - The member struggled even after adding descriptions to all tools and was looking for someone who *had successfully added a custom mcp server to openai*.
- **VAPI MCP Hardware Demo is live**: A member shared a demo showcasing **VAPI MCP** calling hardware stores for part procurement, targeting **hardware engineers** and attached a [video](https://cdn.discordapp.com/attachments/1312302100125843479/1380307827385438338/helion_call_demo.mp4?ex=6844b8d6&is=68436756&hm=cb6e38829856a5ca52b546356018a0237ce82d3e80ce08b9ca48d589838754ac&) of the tool in action.
- **Client Choice: Build Your Own MCP Workflow**: A member sought recommendations for MCP clients that allow custom workflow building without default features like web search, *preferring something that doesn’t assume what I want and instead lets me build my own workflow*.
   - One suggestion was **5ire**, noted for lacking extra tools or prompts and another suggestion to just *deploy 10 computer use agents to browse homedepot.com to make extra sure they have an item in stock*.
- **Delving Deep into Sampling Use Cases**: Discussion arose around the **sampling feature** in MCP, questioning its intended purpose for enriching tool calls versus server-initiated requests, and its implications for client-server communication in potentially long-running, idle sessions, with a related image [attached](https://cdn.discordapp.com/attachments/1312302100125843479/1380504272348905532/image.png?ex=68441e4b&is=6842cccb&hm=74032315eb40b2a5ded1df3b7d073fbd156a11c345d265483311909a27c8ded4&).


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1380469800085487706)** (2 messages): 

> `inked github, Slack MCP server` 


- **Inked Simple Server Launched on Github**: A member shared a dead simple server called **inked**, now available on [GitHub](https://github.com/coldielb/inked), encouraging others to play with it and submit PRs.
   - The server has **two tools** and **three total functions**, and can be installed globally via `npm install -g @frgmt/inked`.
- **Silent AI Agents now in Slack MCP server**: A member announced their **Slack MCP server** is rising on GitHub, highlighting its ability to build **silent**, **invisible AI Agents** without needing to create bots or applications in Slack.
   - The server is available on [GitHub](https://github.com/korotovsky/slack-mcp-server) and they attached a [GIF showcasing its usage](https://cdn.discordapp.com/attachments/1315696461316358175/1380517187785326692/434543420-35dc9895-e695-4e56-acdc-1a46d6520ba0.gif?ex=68442a52&is=6842d8d2&hm=78437dbb1f2f8f9776d0855153c1d68e2ec00098b74fbddc18ba4e53e272148e&).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1380293156163158017)** (56 messages🔥🔥): 

> `Claude Projects Content Increase, Qwen3-Embedding and Qwen3-Reranker Series, Netlify DB Serverless Postgres, Zapier AI Fluency Measurement, Cursor Funding Round` 


- **AMD Acquires Untether AI Team**: AMD acquired the team behind AI chip startup [Untether AI](https://www.crn.com/news/components-peripherals/2025/exclusive-amd-acquires-team-behind-ai-chip-startup-untether-ai).
- **Claude Projects Now 10x Bigger**: Anthropic announced that their **Claude Projects** feature now supports **10 times more content** and activated a new retrieval mode for expanded functional context.
   - The update is rolling out to all paid Claude plans and users called it a *'game changer'* and a significant improvement over **ChatGPT**.
- **Alibaba Opens Qwen3 to the World**: Alibaba's **Qwen3-Embedding** and **Qwen3-Reranker Series** models launched, setting new standards in multilingual text embedding and relevance ranking, supporting **119 languages**.
   - Available in various sizes (0.6B, 4B, 8B), they are open-source on [Hugging Face](https://huggingface.co/), [GitHub](https://github.com/), and [ModelScope](https://modelscope.cn/), and empower diverse use cases like document retrieval, **RAG**, classification, and sentiment analysis.
- **Netlify Comes for Serverless with SupabaseDB**: Netlify announced **Netlify DB**, a serverless Postgres database powered by Neon, designed for AI-native development, aiming to reduce friction between code and data.
   - The **Netlify DB** is easy to set up with a single command and can be integrated into projects via `netlify dev`.
- **Zapier wants AI-Fluent Workers**: Zapier is measuring AI fluency among employees, requiring **100% of new hires to be AI fluent**, with assessments categorizing fluency into 'Unacceptable,' 'Capable,' 'Adoptive,' and 'Transformative' levels.
   - The company uses screenings, skill tests, async exercises, and live interviews for evaluation.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1380356409882841100)** (47 messages🔥): 

> `HFModelTokenizer, Axolotl loss curves, Reward Modeling RFC, Fused Optimizer Issues` 


- **HFModelTokenizer prompts eval customization**: Discussion arose around modifying **HFModelTokenizer** to render templates without tokenization for evaluation, with consideration for custom prompt templates, which are key for the end user.
   - The proposed solution involves prioritizing custom prompt templates if they exist in the tokenizer; otherwise, the **HFModelTokenizer** chat template is used if `apply_chat_template` is true, or an error is raised if no prompt template is available.
- **Regression Detection on Alpaca Cleaned**: A member reported difficulty reproducing a regression on the **alpaca_cleaned** dataset, and requested more details on the setup where the regression was initially detected.
   - They also observed that fine-tuning **Qwen3-4B** on the **alpaca_cleaned** dataset resulted in the same evaluation results as the non-fine-tuned version, but others pointed out that alpaca is a pretty saturated dataset and 4B is small. 
- **Axolotl Convergence Evaluated Post C4 Finetuning**: A member shared a link to [Axolotl PR #2590](https://github.com/axolotl-ai-cloud/axolotl/pull/2590) to show loss curves on **C4**, suggesting that **torchtune** be evaluated after **C4** finetuning because **Axolotl converges**.
   - They noted that the loss curve doesn't strongly suggest that **torchtune's** methods diverge, and offered to share updates using the **Axolotl** values as a reference.
- **Discussion on Clipping Logprobs**: A discussion ensued regarding whether to add **logprobs clipping** to **torchtune**, with a member noting that they don't agree that the existence of a proposed feature in any repository is a criterion on whether to add it in torchtune or not.
   - While the feature is available elsewhere, concerns were raised about it not being intended for user modification and the difficulty of correctly exposing it; however, another member preferred ensuring ease of self-implementation over direct maintenance of the feature.
- **Fused Optimizer raises AssertionError**: A member reported an `AssertionError` related to `fused_adagrad` when using a fused optimizer on a nightly build, particularly when a compute mesh was not found.
   - After testing, it was found that the issue only occurred with `fused=True` and that **SGD** started working after upgrading to the latest **torchtune**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1380292133021548665)** (33 messages🔥): 

> `Training LLMs, Datasets for LLMs, Clustering emails with RAG, Meta's OPT-175B logbook, GPT sycophancy` 


- ****Marius Paper Surfaces****: A member shared a link to the [Marius paper](https://arxiv.org/abs/2402.00854) as a potential resource.
- ****LLM Training Resource Quest Begins****: A member is seeking resources on training industry-level LLMs with real-world datasets, beyond toy models, including insights from experts on data handling, diverse outputs, and preventing overfitting.
   - They mentioned Sebastian Raschka's "Build a Large Language Model (From Scratch)" ([YouTube link](https://youtu.be/Zar2TJv-sE0)) as a starting point but seek more detailed training pipelines with mixed, diverse datasets, methods for stabilizing training, and preventing catastrophic forgetting.
- ****RAG Clustering Conundrum Conjured****: A member asked for a way to do clustering with emails and an out-of-the-box RAG solution, aiming to put the emails in n bins without knowing what n is or having labels.
   - One member suggested embedding each email using ModernBERT and using a travelling salesman problem solver to order them in clusters based on distance, and another suggested using OpenAI's Embeddings ([platform.openai.com](https://platform.openai.com/docs/guides/embeddings#ho)).
- ****Meta's OPT-175B Logbook Disclosed****: A member mentioned **Meta's OPT-175B logbook**, documenting problems in the training process ([GitHub link](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf), [ArXiv link](https://arxiv.org/abs/2205.01068)).
- ****ChatGPT's Sycophancy Suspected****: A member inquired whether **ChatGPT** is becoming increasingly sycophantic.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1380291881120039115)** (2 messages): 

> `Vec2Vec Code Review, Translators/Transformers directory, Background review of implementation` 


- **Vec2Vec Code Dive Scheduled**: A code review is scheduled for [Vec2Vec](https://github.com/rjha18/vec2vec), focusing on the `translators/transformers` directory and the broader `translators` codebase.
   - The discussion will cover the implementation of a paper ([https://arxiv.org/abs/2505.12540](https://arxiv.org/abs/2505.12540)) and include a background review before diving into the code.
- **Meeting Postponed**: The scheduled meeting has been postponed to next week.
   - The user clarified that they mistakenly said "tomorrow" during the call and the meeting is now scheduled for the following week.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1380595188501844192)** (7 messages): 

> `Qwen3 Embedding, Nemotron-H Reasoning Model, EleutherAI and Public Data LLMs, Cohere's Business Model, RAG Marketing` 


- ****Qwen3** Embeddings Released Under **Apache 2.0 License****: Alibaba released the **Qwen3 Embedding** series, designed for text embedding, retrieval, and reranking tasks, leveraging the **Qwen3** foundation model and available under the **Apache 2.0 license** on [Hugging Face](https://huggingface.co/Qwen) and [ModelScope](https://modelscope.cn/models?search=Qwen).
   - The series employs dual-encoder and cross-encoder architectures, fine-tuned via LoRA to enhance text understanding, with the [technical report and code available on GitHub](https://github.com/QwenLM/Qwen-Embedding).
- ****NVIDIA's Nemotron-H** Reasoning Models Boost Throughput**: NVIDIA introduced the **Nemotron-H-47B-Reasoning-128K** and **Nemotron-H-8B-Reasoning-128k** models to tackle reasoning-intensive tasks, now available in [FP8 quantized variants](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/?linkId=100000368479233) for efficient throughput in latency-sensitive environments.
   - These models, built from the **Nemotron-H-47B-Base-8K** and **Nemotron-H-8B-Base-8K** foundation models, aim to advance the science behind reasoning models.
- ****EleutherAI** Trains Competitive LLM with Public Data**: A member noted that [EleutherAI](https://huggingface.co/blog/stellaathena/common-pile) demonstrated the feasibility of training a competitive LLM using public-domain and openly-licensed data.
   - This achievement highlights the potential for creating powerful language models without relying on proprietary datasets.
- **Doubts Arise Over **Cohere's** Continued Existence**: One member expressed confusion about **Cohere's** continued operation and customer base.
   - Another member responded that **Cohere** sells services and solutions directly to businesses, especially in the context of RAG (Retrieval-Augmented Generation) applications.
- ****RAG** Marketing Boosts Embedding Popularity**: It was pointed out that while companies like Google have offered embeddings for a long time, their marketing around **RAG** (Retrieval-Augmented Generation) has driven increased adoption.
   - The member also noted that the **Qwen** embedding model, with its licensing, is a strong contender in the field.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1380396427284316301)** (1 messages): 

> `LLVM, loop splitting, ROCm, InductiveRangeCheckElimination` 


- **Loop Splitting Specific to ROCm?**: A member is investigating speeding up CAT with **LLVM** and asks if **loop splitting** is only present on the **ROCm llvm-project**.
   - They reference the [ROCm documentation](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/reference/rocmcc.html#loop-splitting) on loop splitting and note it is only in their custom llvm-project.
- **InductiveRangeCheckElimination missing from llvm.py**: A member notes that the **llvm C source** used in runtime/autogen/llvm.py lacks **InductiveRangeCheckElimination** from the **C++ LLVM library**.
   - They are considering using *llvmlite* to get access to IRCE or extern/rewrite C++ since they cannot add loop splitting.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1380273021066936430)** (23 messages🔥): 

> `tinygrad kernel optimization, hlb_cifar10 data shuffling, OpenCL kernel performance, GPU indexing kernels` 


- **Debugging tinygrad's slow GPU Kernels**: A user is debugging a slow GPU kernel generated by tinygrad when shuffling a dataset tensor of float32 `[50000,3,32,32]` in the `hlb_cifar10` example.
   - The user tried `DEBUG=4` and `VIZ=1` but found the output unhelpful and also determined that `BEAM=4` won't fix the underlying issue.
- **Manual OpenCL Kernel Shuffles Array Much Faster**: The user tested a manually written OpenCL kernel for shuffling the same sized array (50000,3,32,32) and found it shuffles in **0.33 seconds**.
   - This contrasts with the tinygrad-generated kernel which takes **5 seconds** even with simple unshuffled indexing, prompting further investigation into tinygrad's kernel generation.
- **Investigating Weird tinygrad Indexing Kernel**: The user is trying to understand why tinygrad generates such a slow indexing kernel, especially since CPU-based copying and shuffling is faster.
   - ChatGPT helped the user to realize a simplified indexing opencl kernel would be much faster.
- **NumPy Shuffling Performance**: The user tested `np.random.permutations(50000*3*32*32)` and `np.random.permutations(50000)[None].repeat(3*32*32, 0).T.flatten()` both of which took **0.33 seconds**.
   - The user wants to find out what makes tinygrad generate such weird indexing kernel.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1380283058506174575)** (19 messages🔥): 

> `Video Function, Credit Costs, Manish Model Update, Manus partnership with Claude, Egyptian users` 


- **Manus New Video Function Falls Flat**: Members tried the new video function and found it *very immature*.
   - One user noted they saw a video from a close friend and felt it was too premature for practical use.
- **High Credit Costs Drive Users to Alternatives**: Users complain **Manus's** high credit costs, citing prices like *1900 credits for 19 dollars* as insufficient when each task costs *300-400 credits*.
   - One user mentioned that due to high costs they are using cheaper alternatives and [suggested reading the guides](https://discord.com/channels/1349440650495398020/1370393476029616238) from **Manus Team** to perform cheaper tasks.
- **Manish Model Update Speculation Intensifies**: Users asked if the model would be updated to **Sonnet 4.0**.
   - One user speculated that, due to the recent partnership with **Claude**, this is highly likely to happen, another user mentioned the models are up to date, while some users suggested **Sonnet's** lack of context length is an issue which **Manus** solves.
- **Egyptian User makes an appearance**: A user inquired whether there were any other Egyptian users present in the chat.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1380319445553254492)** (3 messages): 

> `AI Agents, MCP vs A2A, Vector Databases` 


- **Debate on Productionizing AI Agents begins**: A discussion session hosted by @tuanacelik at Snowflake Dev Day, explored the [blockers to productionizing AI Agents](https://t.co/DJGBe3TqZb).
- **MCP vs A2A Standards Clash**: @seldo gave a lightning tour of **13 different protocols** vying to become the standard way for agents to talk to tools at the [MCP Dev Summit](https://t.co/qZv8duKRut), including **MCP, A2A, and ACP**.
- **Vector DB Best Practices Hit Munich**: @itsclelia will hold a talk at the **BASED Meetup in Munich** on June 12th about best practices to boost your RAG pipelines, from data preparation to query optimization.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1380349256468004944)** (9 messages🔥): 

> `files_via_content mode, AgentWorkflow orchestration, Multi-Agent setup` 


- **LlamaIndex Clarifies `files_via_content` Mode**: A member asked for documentation on how the [`files_via_content` mode](https://docs.cloud.llamaindex.ai/llamacloud/retrieval/modes#files_via_content-mode) works in LlamaIndex.
   - Another member responded with a direct link to the relevant section in the LlamaIndex Cloud documentation, providing a quick solution.
- **AgentWorkflow Orchestration with Dynamic Delegation**: A member inquired about orchestrating a team of agents within **AgentWorkflow**, specifically asking how to dynamically delegate tasks to specialized agents.
   - The inquiry focused on whether this functionality is built into LlamaIndex or if custom workflows need to be defined.
- **Multi-Agent Setup Example Provided**: In response to a query about orchestrating multiple agents, a member provided a link to a [LlamaIndex documentation example](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/) demonstrating a **multi-agent setup**.
   - The member explained that *an "orchestrator" agent is basically just an agent with tools (and those tools might be other agents)*, offering a conceptual overview of the architecture.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1380292234817044631)** (1 messages): 

> `SchemaLLMPathExtractor, Graph database population` 


- **Newbie Asks about `SchemaLLMPathExtractor`**: A new LlamaIndex user is exploring the use of [`SchemaLLMPathExtractor`](https://llama-index.readthedocs.io/en/stable/module_guides/indexing/schema/schema_llm_path_extractor.html) to populate a graph database.
   - The user inquired if the community has published schemas (entities, relations, rules) that they could use out of the box.
- **Community Schemas for Graph DB Population Sought**: A member is seeking pre-built schemas (entities, relations, rules) for populating graph databases within the LlamaIndex ecosystem.
   - The user hopes to leverage existing community resources to streamline the process of integrating organizational data (people, applications, etc.) into a graph structure.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1380315274984816650)** (5 messages): 

> `VPS for API server, RAM Pricing, Mistral MOE, Deepseek MOE, Chinese CPU vendor` 


- **Venturing VPS for API Server**: A user suggested renting a **VPS** to build the **API server** for **GPT4All**.
   - The user attached a screenshot of the **GPT4All** interface, noting that it sometimes doesn't respond and is buggy.
- **Rummaging RAM Pricing Revelations**: A member shared a [YouTube video](https://m.youtube.com/watch?v=Tp0k6VDXUOQ) discussing **RAM pricing** where 1 TB can be reasonably priced around a few thousand dollars.
   - The member added that ordinary **PCs** can struggle with insufficient RAM and that the market for computer components can be global.
- **Marveling MOE Model Metrics**: A user wondered if one could imagine running **Mistral MOE** or **Deepseek MOE** full **Q8 Quantization** at TRILLION tokens / second.
   - The user linked to an article about a [Chinese CPU vendor](https://www.techradar.com/pro/chinese-cpu-vendor-swaps-amd-zen-architecture-for-homegrown-one-to-deliver-128-core-monster-to-give-epyc-and-xeon-a-run-for-their-money) swapping **AMD Zen architecture** for a homegrown one to deliver a **128-core monster**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1380263648450314384)** (3 messages): 

> `Session Thanks, Blockchain Engineer Introduction, AI Agent Engineer Introduction` 


- **Session gets Thanks**: Two members thanked another member for a session, with a [YouTube link](https://youtu.be/Vqsfn9rWXR8) provided.
   - A member asked if the session slides were available.
- **Engineer introduces himself**: A software engineer with experience in **Blockchain** and **AI Agents** introduced himself.
   - His expertise includes **EVM**, **Solana**, **Cardano**, **Hydra**, **Aptos**, **Cosmos**, **Tron**, **zk-SNARKs** in Blockchain and **LLM**, **NLP**, **LangChain**, **AutoGen**, **TorchRL**, **DL**, **Azure ML**, **AI Agent** in AI.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

stormortiz: here is an magic place
  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1380469381951127634)** (2 messages): 

> `Introductions, ML Audio Engineer` 


- **New Member Introduces Himself as ML Audio Engineer**: A new member introduced themself as a **Machine Learning Audio Engineer**.
- **Community Welcomes New Member**: The community welcomes the new member to the Cohere Discord server.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/)** (1 messages): 

radhakrishnan_20251: thanks for the update
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1380550349911490811)** (1 messages): 

> `MCP Tools Authorization, Enterprise OAuth` 


- **MCP Tools Authorization Article Posted**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-enterpriseai-oauth-activity-7336749966234718208-Cb9E?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk) about building **MCP tools authorization** for enterprise.
   - The post compiles findings into an article regarding **enterprise OAuth**.
- **OAuth Findings Compiled**: The author's findings on building **MCP tools authorization** for enterprises have been compiled into an article.
   - This article specifically addresses aspects related to **enterprise OAuth** implementation and best practices.


  