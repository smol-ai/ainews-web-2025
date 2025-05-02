---
id: MjAyNS0w
title: not much happened today
date: '2025-05-01T05:44:39.731046Z'
description: >-
  **Microsoft** released **Phi-reasoning 4**, a finetuned 14B reasoning model
  slightly behind QwQ but limited by data transparency and token efficiency
  issues. **Anthropic** introduced remote MCP server support and a 45-minute
  Research mode in **Claude**. **Cursor** published a model popularity list.
  **Alibaba** launched **Qwen3-235B** and other Qwen3 variants, highlighting
  budget-friendly coding and reasoning capabilities, with availability on
  **Together AI** API. **Microsoft** also released **Phi-4-Mini-Reasoning** with
  benchmark performance on AIME 2025 and OmniMath. **DeepSeek** announced
  **DeepSeek-Prover V2** with state-of-the-art math problem solving, scaling to
  671B parameters. **Meta AI**'s **Llama** models hit 1.2 billion downloads,
  with new **Llama Guard 4** and **Prompt Guard 2** for input/output filtering
  and jailbreak prevention. **Xiaomi** released the open-source reasoning model
  **MiMo-7B** trained on 25 trillion tokens. Discussions on AI model evaluation
  highlighted issues with the **LMArena leaderboard**, data access biases
  favoring proprietary models, and challenges in maintaining fair benchmarking,
  with suggestions for alternatives like **OpenRouterAI** rankings. *"LMArena
  slop and biased"* and *"61.3% of all data going to proprietary model
  providers"* were noted concerns.
companies:
  - microsoft
  - anthropic
  - cursor
  - alibaba
  - togethercompute
  - deepseek
  - meta-ai-fair
  - xiaomi
  - openrouterai
  - cohere
models:
  - phi-4
  - phi-4-mini-reasoning
  - qwen3-235b
  - qwen3-moe-235b
  - qwen3-moe-30b
  - qwen3-dense-32b
  - qwen3-dense-14b
  - qwen3-dense-8b
  - qwen3-dense-4b
  - qwen3-dense-0.6b
  - qwen2.5-omni-3b
  - deepseek-prover-v2
  - llama
  - llama-guard-4
  - prompt-guard-2
  - mimo-7b
topics:
  - reasoning
  - model-fine-tuning
  - model-evaluation
  - benchmarking
  - model-popularity
  - open-source
  - math
  - model-scaling
  - model-filtering
  - jailbreak-prevention
people:
  - cline
  - reach_vb
  - vipulved
  - akhaliq
  - omarsar0
  - zhs05232838
  - huajian_xin
  - mervenoyann
  - karpathy
  - random_walker
  - sarahookr
  - blancheminerva
  - clefourrier
---



**a quiet day.**

> AI News for 4/30/2025-5/1/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (214 channels, and 4767 messages) for you. Estimated reading time saved (at 200wpm): 453 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Microsoft released [Phi-reasoning 4,](https://www.reddit.com/r/LocalLLaMA/comments/1kbvwsc/microsoft_just_released_phi_4_reasoning_14b/) a reasoning finetune of the 14B Phi-4 that is slightly behind QwQ in performance, but lack of transparency around their data and complains of inference-token hungriness limit the excitement around it.

Anthropic launched [remote MCP server support in Claude](https://news.ycombinator.com/item?id=43859536) and an up to 45-min long Research mode.

Cursor released their model popularity list, with not much surprises.

[](https://resend-attachments.s3.amazonaws.com/hM6qEzvvHIVmVdX)

---

# AI Twitter Recap

**Language Models and Releases**

- **Qwen Model Updates**: [@cline](https://twitter.com/cline/status/1917708041857949983) reports on early user feedback for **Qwen3-235B**, noting its potential as a budget-friendly coding model with positive initial results. [@reach_vb](https://twitter.com/reach_vb/status/1917938596465750476) highlights the release of various **Qwen3 models, including MoE (235B, 30B) and Dense (32, 14, 8, 4, 0.6B) versions**. [@togethercompute](https://twitter.com/togethercompute/status/1917616701249565120) and [@vipulved](https://twitter.com/vipulved/status/1917777842466889873) announce **Qwen 3 235B's availability on the Together AI API**, emphasizing its reasoning capabilities and efficiency. Additionally, [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1917585963775320086) introduces **Qwen2.5-Omni-3B**, which reduces VRAM consumption while maintaining multimodal comprehension.
- **Phi-4 Reasoning Models from Microsoft**: [@_akhaliq](https://twitter.com/_akhaliq/status/1917761687723147707) and [@omarsar0](https://twitter.com/omarsar0/status/1917954418173247909) mention the release of **Microsoft's Phi-4-Mini-Reasoning**, a small language model for math, with a technical report available. [@reach_vb](https://twitter.com/reach_vb/status/1917852036369916081) notes that **Phi 4 Reasoning and Reasoning plus are now on Hugging Face**, highlighting its performance on benchmarks like AIME 2025 and OmniMath.
- **DeepSeek's Prover V2**: [@zhs05232838](https://twitter.com/zhs05232838/status/1917600755936018715) announces the release of **DeepSeek-Prover V2**, which achieves high scores on miniF2F problems and improves SoTA performance on the PutnamBench. [@reach_vb](https://twitter.com/reach_vb/status/1917549921470972172) notes that **DeepSeek Prover V2 is available directly on the model page powered by Novita Labs**. [@huajian_xin](https://twitter.com/huajian_xin/status/1917603640124363090) celebrates the scaling up to **671B for the Prover project**, expressing gratitude to colleagues at DeepSeek.
- **Meta's Llama Updates**: [@AIatMeta](https://twitter.com/AIatMeta/status/1917353526088589409) reports that **Llama has reached 1.2 billion downloads**, with most being Llama derivatives. [@mervenoyann](https://twitter.com/mervenoyann/status/1917503204826255730) announces the release of **Llama Guard 4 and Prompt Guard 2 models for filtering model inputs/outputs and preventing jailbreaks**.
- **Xiaomi's MiMo-7B**: [@_akhaliq](https://twitter.com/_akhaliq/status/1917410882939715608) and [@omarsar0](https://twitter.com/omarsar0/status/1917582720341008814) discuss the release of **Xiaomi's MiMo-7B**, an open-source reasoning model with Multi-Token-Prediction trained on 25 trillion tokens.

**AI Model Evaluation and Leaderboards**

- **Issues with Chatbot Arena Leaderboard**: [@karpathy](https://twitter.com/karpathy/status/1917546757929722115) discusses issues with the **LMArena leaderboard**, noting discrepancies between arena scores and real-world performance, and suggests the @OpenRouterAI LLM rankings as a potential alternative eval. [@random_walker](https://twitter.com/random_walker/status/1917516403977994378) calls the LMArena slop and biased. [@sarahookr](https://twitter.com/sarahookr/status/1917547727715721632) shares a cross-institutional collaboration highlighting the difficulty in maintaining fair evaluations on @lmarena_ai.
- **Concerns About Data Access and Testing**: [@sarahookr](https://twitter.com/sarahookr/status/1917547738553803018) notes large differences in **Arena data access, with 61.3% of all data going to proprietary model providers**. [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1917445722380681651) links to the same @Cohere_Labs work, noting the difficulties in maintaining fair evaluations. [@clefourrier](https://twitter.com/clefourrier/status/1917488919450374383) notes this has lead to overfitting the slop, and states that closed source companies have unfair access to data. [@sarahookr](https://twitter.com/sarahookr/status/1917547733994594420) highlights the hidden testing.
- **LMArena's Response**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1917668731481907527) responds to the criticisms, stating that the article contains factual errors and misleading statements, including those regarding unequal treatment of model providers and the simulation of LMArena. They also link to their policy and actual stats.

**Applications of AI Agents and Tools**

- **AI in Coding and Development**: [@LangChainAI](https://twitter.com/LangChainAI/status/1917646746798416121) announces a partnership with @UiPath to facilitate building, deploying, and observing AI agents for enterprise automation. [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1917602387381924173) highlights an updated course on LLMs as Operating Systems, focusing on agent memory and building adaptive AI systems.
- **AI in Search and Information Retrieval**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1917977286713758073) announces that **Perplexity can now fact check WhatsApp messages**, providing instant verification of forwarded content. [@alexalbert__](https://twitter.com/alexalbert__/status/1917973599044116576) discusses the integrations. [@karpathy](https://twitter.com/karpathy/status/1917961248031080455) at a vibe coding hackathon found building and deploying a full web app today is a painful slog.
- **AI for Robotics and Automation**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1917593514566901800) introduces the Summarize, Analyze, Synthesize (SAS) prompt, enabling robots to self-improve through interaction and learning.

**AI Safety, Ethics, and Responsible Development**

- **OpenAI's GPT-4o Rollback**: [@OpenAI](https://twitter.com/OpenAI/status/1917411480548565332) announces the rollback of last week's GPT-4o update due to overly flattering and agreeable behavior, providing access to an earlier version with more balanced behavior. The announcement is criticized by [@nearcyan](https://twitter.com/nearcyan/status/1917449708647375159) as a lie.
- **Need for Strong Controls**: [@jackclarkSF](https://twitter.com/jackclarkSF/status/1917629784940597514) emphasizes the need for strong controls on AI infrastructure to prevent offshoring of critical production capacity. [@johnschulman2](https://twitter.com/johnschulman2/status/1917487672983183433) highlights the need to remove some bad causal contributors to the preferences.

**Education and Learning in AI**

- **Coding and AI Education**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1917985792607363189) discusses the importance of teaching AI-enabled coding, highlighting the role of teachers and AI in K-12 education. [@alexalbert__](https://twitter.com/alexalbert__/status/1917603519227650533) highlights the importance of learning to code for human-AI collaboration.
- **Recommendations for ML Teams**: [@ProfTomYeh](https://twitter.com/ProfTomYeh/status/1917634404903539022) provides advice on what makes someone invaluable on an AI/ML team, stressing math intuition, whiteboard diagrams, and understanding people.

**Humor and Miscellaneous**

- **Dentist asks p(doom)**: [@AmandaAskell](https://twitter.com/AmandaAskell/status/1917770005988663412) said her dentist asked her what her p(doom) is.
- **AI as code assistant**: [@sarahcat21](https://twitter.com/sarahcat21/status/1917649137543377235) said that Cursor once tried to suggest a line of code to @ekzhang1, and it’s still apologizing.
- **Korean Tteokbokki spot**: [@dylan522p](https://twitter.com/dylan522p/status/1917719768066363629) shares an anecdote from Korea, noting he bought an elder lady some cigarettes so she would take his card, and then serve him a meal.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Phi 4 Reasoning Model Release and Discussion

- [**Microsoft just released Phi 4 Reasoning (14b)**](https://huggingface.co/microsoft/Phi-4-reasoning) ([Score: 641, Comments: 126](https://www.reddit.com/r/LocalLLaMA/comments/1kbvwsc/microsoft_just_released_phi_4_reasoning_14b/)): **Microsoft has released Phi-4 Reasoning (14B), a static model trained on an offline dataset with reported cutoff dates reaching March 2025, indicating the dataset likely contains future-dated content or has forward-dated cutoff as a quirk. Early comparison interest is with the Qwen 3 30B MoE, suggesting high expectations of reasoning and general performance benchmarks. Links to community-uploaded GGUF (Quantized General Unstructured File Format) conversions for both 'phi-4-mini-reasoning' and 'phi-4-reasoning-plus' are provided: [Phi-4-mini-reasoning GGUF](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF), [Phi-4-reasoning-plus-GGUF](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF), plus mention of dynamic 4-bit safetensors for optimized inference.** Technically-minded users express anticipation over phi-4's performance, especially compared to the new MoE Qwen models, and show eagerness for continued rapid model releases and quantized format availability.
    - There is technical curiosity about how Phi-4 Reasoning (14B) compares to Qwen 3 30B MoE, particularly in terms of inference quality and reasoning benchmarks, given the latter's strong reputation in the MOE (Mixture of Experts) space.
    - Phi-4 Reasoning is available in multiple formats on HuggingFace, including GGUF and 4bit safetensors, with both 'mini-reasoning' and 'reasoning-plus' versions uploaded by the community for easier deployment and quantization-friendly use cases ([Hugging Face links](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF), [Phi-4-reasoning-plus-GGUF](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF)).
    - A user observed that the model handled a 32k token inference (likely context length), noting repeated correct responses during thinking. This points to potential issues with attention patterns, verbosity, or energy usage during long context generation—a relevant concern for optimization.
- [**Phi 4 Reasoning**](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf) ([Score: 114, Comments: 12](https://www.reddit.com/r/LocalLLaMA/comments/1kbvrgs/phi_4_reasoning/)): **Microsoft's [Phi-4-reasoning](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf) is a 14B parameter language model that leverages supervised fine-tuning (SFT) on a carefully filtered, diverse set of reasoning-heavy prompts, with additional data from o3-mini outputs, achieving state-of-the-art performance (on par with or surpassing models like DeepSeek-R1 and Gemini 2 Flash) across math, science, and planning benchmarks at a much smaller parameter budget. The enhanced Phi-4-reasoning-plus variant uses outcome-based reinforcement learning to further boost reasoning performance, with testing on new NP-hard, algorithmic, and planning tasks, highlighting that data curation and targeted SFT/RL can yield small models with excellent reasoning generalization and efficiency. The release also proposes more rigorous, variance-aware evaluation protocols for reasoning LLMs due to issues with small benchmark set sensitivity.** Commenters note the direct benefit of Microsoft's relationship with OpenAI (using o3-mini outputs for SFT), the rapid evolution and impact for edge AI by compact models, and Microsoft's capability to capitalize on these advances at scale, as detailed in the associated [Azure blog post](https://azure.microsoft.com/en-us/blog/one-year-of-phi-small-language-models-making-big-leaps-in-ai/).
    - A key technical point is that Phi-4's reasoning capabilities may be unique among open models due to its direct training from OpenAI's O-series models, suggesting transfer of reasoning skill and knowledge distillation otherwise reserved for closed-source models.
    - The linked Microsoft Azure blog post provides deeper technical context, highlighting Phi's progression and claims of *small language models* (SLMs) making substantial AI advances, potentially relevant for on-device (Edge AI) applications thanks to their computational efficiency and performance balance.
    - Discussion suggests that Microsoft's resources position it to effectively deploy SLMs like Phi for real-world Edge AI use cases, leveraging efficient reasoning performance where larger models would be impractical due to latency and hardware limits.

### 2. Qwen 3 Models: Impressions and Capabilities

- [**We crossed the line**](https://www.reddit.com/r/LocalLLaMA/comments/1kc10hz/we_crossed_the_line/) ([Score: 651, Comments: 132](https://www.reddit.com/r/LocalLLaMA/comments/1kc10hz/we_crossed_the_line/)): **The OP reports that the QWEN3 32B large language model is now capable of solving all their programming needs—tasks previously requiring access to leading commercial LLMs like ChatGPT or Grok (version 3). They highlight the model's local deployability and capability, suggesting a significant leap in open-source or locally-hosted model performance for coding assistant use-cases.** Top commenters are seeking further technical specificity: one asks about the OP's coding expertise to gauge model utility, another requests a comparison to the 30b-a3b model, and a third calls for concrete task examples to benchmark performance more objectively.
    - Several commenters request clarification on the specific coding tasks and examples that the 32B model was evaluated on, emphasizing the need for detailed benchmarks or task descriptions to meaningfully assess performance. This highlights the technical community's preference for reproducible, concrete benchmarks over anecdotal claims.
    - A technical inquiry is made about comparing the 32B model to a 30B-A3B model, specifically suggesting running the same set of tasks on both and reporting on relative performance. This stresses the importance of direct, model-to-model benchmarking within the same task domains for fair evaluation.
    - A commenter asks about implementation details: specifically which quantization, Hugging Face repository, and inference server were used for running the model. There is also mention of interest in testing Unsloth's 128k context-length versions, suggesting comparative evaluation scenarios and a focus on deployment/inference optimizations.
- [**Impressive Qwen 3 30 MoE**](https://www.reddit.com/r/LocalLLaMA/comments/1kc6hgn/impressive_qwen_3_30_moe/) ([Score: 107, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1kc6hgn/impressive_qwen_3_30_moe/)): **The post discusses the multilingual translation abilities of the Qwen 3 30 MoE model, particularly noting high accuracy in Spanish, Dutch, German, and English, even handling regional dialects convincingly. Commenters highlight that the MoE (Mixture of Experts) architecture enables efficient CPU-only inference, but caution that specific languages (like German) can still exhibit mistakes, such as grammatical gender errors or unnatural English-influenced phrasing.** Technical commenters raise concerns about potential degradation of complex reasoning and overall 'intelligence' in non-English use, recommending additional testing for logical tasks, and warn against overgeneralizing benchmarks as real-world performance ('benchmaxxing').
    - Qwen 3 30 MoE is noted as one of the fastest and most capable models for CPU-only inference, especially on AVX2 CPUs, where even quantized versions (like q2_K_L) achieve around '7 t/s' on low-end hardware with 16GB RAM. Comparisons against other models, such as a 4B Q_4_K_M on llama.cpp, show Qwen is both faster and provides more accurate answers for complex and precise queries.
    - Despite excellent multilingual abilities (noted for Polish), users report that the model makes typical errors in other languages like German: issues with gender, Anglicized phrasing, and degraded performance on complex/logical tasks outside English. These indicate limitations in non-English training data coverage and generalization.
    - The model's strong practical value is emphasized for users with limited hardware: it requires only 'plenty of ram with AVX2 cpu and 10-ish gb of space' for full LLM functionality, distinguishing itself from comparable models in both speed and memory efficiency.
- [**Qwen 3 4B is the future, ladies and gentlemen**](https://i.redd.it/2aw947hyi3ye1.png) ([Score: 318, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1kc016i/qwen_3_4b_is_the_future_ladies_and_gentlemen/)): **The image presents a QA scenario where Qwen 3 4B (a 4-billion parameter large language model by Alibaba/Qwen) accurately reasons that 9.9 is greater than 9.11, providing a clear step-by-step explanation that demonstrates number parsing ability lacking in many prior smaller open LLMs. The post and comments emphasize that, while such tests are now "in the training data" and may not be novel, the performance signals maturity in basic numeracy and reasoning, especially for a model of this small size. Commenters draw comparisons to models like Llama 3.1, hinting at Qwen 3 4B's competitive edge as an open-source alternative in the 4-8B parameter range.** The top technical comment expresses concern that basic reasoning tests (like number parsing or spelling) have become trivial and are overly emphasized, with calls for more challenging and novel benchmarks. Another commenter is excited to see how the 8B version of Qwen 3 compares and notes that Qwen 3 might rival or surpass Llama 3.x in this space.
    - Discussion centers on the need for *more robust and meaningful benchmarks* for evaluating models like Qwen 3 4B, given that many popular test prompts (e.g., counting letters, comparing decimals) are frequently present in training data and may not accurately measure real-world reasoning capabilities.
    - A comparison is made between Qwen 3 4B (and its 8B counterpart) and Llama 3.1, with suggestions that Qwen represents a strong open-source alternative as model sizes in the 4-8B range become more powerful.
    - One user shares their experience running Qwen 3 4B on an iPhone 16 Pro Max, noting that inference speed is *"fast enough"*, indicating notable efficiency and feasibility of using advanced small models on consumer mobile hardware.
- [**The models developers prefer.**](https://i.redd.it/mg9ey4l4b7ye1.jpeg) ([Score: 103, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1kcdpce/the_models_developers_prefer/)): **The image, sourced from Cursor AI, presents two lists: "Most Popular on Cursor" and "Fastest Growing on Cursor" for April 2025. Popular models include Claude 3.7 Sonnet and Gemini 2.5 Pro, while the fastest-growing models highlight o3, o4-mini, and DeepSeek V3.1. This snapshot provides a cross-section of current LLM usage trends among Cursor's developer user base, highlighting both established models and newer entrants gaining traction.** Comments point out that Cursor's infrastructure complicates local model use, potentially biasing results toward API/hosted models. There is also skepticism about generalizability, as the data reflects only Cursor's user preferences, and alternative leaderboards (like Aider) may show different trends.
    - A commenter points out that running local models via Cursor is challenging due to network constraints, specifically requiring a public IP or proxy, which could skew any reported preferences or statistics regarding which models are popular among developers using Cursor.
    - Another technical point raised concerns the high usage cost of models like o3 (presumably OpenAI's GPT-3.5 or GPT-4 tier), with the comment noting that their expense makes them less accessible for widespread use, impacting their adoption in developer workflows.
    - One participant differentiates between model leaderboards, noting their personal preference for the "Aider leaderboard" over Cursor's, implying that model popularity or preference metrics can vary significantly depending on the context and platform, which affects how results should be interpreted in technical discussions about model selection.

### 3. Novel Model and Training Method Announcements (TTS/ASR, KL Optimization)

- [**New TTS/ASR Model that is better that Whisper3-large with fewer paramters**](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) ([Score: 164, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1kcdxam/new_ttsasr_model_that_is_better_that/)): **NVIDIA has released the Parakeet-TDT-0.6B-v2 model, a speech-to-text (STT/ASR) system that achieves performance surpassing Whisper-v3 Large with fewer parameters (**`~0.6B`**). The model provides char/word/segment-level timestamps, but is limited to English only; no text-to-speech (TTS) support is described. Users note low compute requirements for training compared to large language models (LLMs), and there is anticipation for speaker recognition support.** Discussion questions the TTS classification, clarifying it's an ASR/STT model. There is technical interest in the availability of fine-grained timestamps and surprise at the efficiency relative to LLMs.
    - The model supports char, word, and segment level timestamps, which could have significant implications for downstream tasks like speaker recognition—something users note would make the model particularly useful if further diarization support is added.
    - A key highlight is the model's data mix: 10,000 hours of human-transcribed data spanning datasets such as LibriSpeech, Fisher Corpus, VoxPopuli, Europarl-ASR, and more, plus 110,000 hours of pseudo-labeled data from YouTube-Commons, YODAS, and Librilight. Commenters point out that this data composition is significantly more comprehensive and superior compared to Whisper's training corpus, potentially contributing to better generalization and performance.
    - Several users observe that the model achieves its results with significantly less computational resources relative to large language models (LLMs), underscoring the efficiency and thoughtful design of its training pipeline compared to more compute-heavy models like Whisper-3.
- [**New training method shows 80% efficiency gain: Recursive KL Divergence Optimization**](https://arxiv.org/abs/2504.21707) ([Score: 139, Comments: 13](https://www.reddit.com/r/LocalLLaMA/comments/1kbytzk/new_training_method_shows_80_efficiency_gain/)): **A new training method, Recursive KL Divergence Optimization, is reported to achieve an ~80% efficiency gain according to the original post. The method references benchmarks and use cases based on image datasets (CIFAR-10, CIFAR-100, STL-10), as indicated by the use of PIL in the corresponding notebook and by citations in the associated research paper, but was shared in a LocalLLaMA subreddit raising questions about its applicability to LLMs (Language Models) versus image models. No direct implementation for popular open-source model trainers (e.g., kohya, SimpleTuner) is provided, limiting immediate reproducibility and cross-domain validation.** Commenters question whether the technique can be effectively used for ongoing fine-tuning scenarios, and seek clarification on its applicability to LLMs versus image datasets, indicating a need for generalized implementation and benchmarking. There is also interest in audio summaries of the paper for broader dissemination (example: notebooklm audio link).
    - StableLlama notes that although the method is discussed in the context of LLMs due to the post's subreddit, the original paper and accompanying code primarily use image datasets like CIFAR-10, CIFAR-100, and STL-10, with implementations referencing `PIL`, an image library. This raises confusion about applicability to language models. They also inquire about integration with open-source training frameworks such as kohya or SimpleTuner, suggesting a need for benchmarking RLKD on real-world tasks beyond images.
    - Revolaition highlights the claimed practical benefits of Recursive KL Divergence Optimization (RKLD): improved efficiency in training, especially for fine-tuning, with potential for faster training times, reduced computational costs, and lower hardware requirements. These points suggest promising implications for resource-constrained settings if validated.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. CivitAI Adult Content Purge and Community Alternatives

- [**CIVITAI IS GOING TO PURGE ALL ADULT CONTENT! (BACKUP NOW!)**](https://www.reddit.com/r/StableDiffusion/comments/1kbxq93/civitai_is_going_to_purge_all_adult_content/) ([Score: 656, Comments: 346](https://www.reddit.com/r/StableDiffusion/comments/1kbxq93/civitai_is_going_to_purge_all_adult_content/)): **CivitAI has introduced a new AI-powered content tagging and moderation system, reportedly Clavata, which users claim is producing flawed content classifications—specifically, mislabeling bodily fluids (e.g., cum as vomit) and innocuous gestures (e.g., fingers or fellatio gestures) as violating content. The post warns of widespread automatic blocking and potential deletion of adult content under the updated Terms of Service, citing test uploads of both explicit and non-explicit material that were disproportionately blocked. The concern centers on the technical limitations and false positives of the AI system, potentially leading to overzealous or unintended censorship of adult or borderline content.** Comments challenge the sensational nature of the claims, pointing out that some blocked NSFW content was unblocked after metadata review and highlighting a lack of evidence for a planned site-wide purge. Skeptics note the OP's visible explicit uploads that remain accessible, suggesting that while false positives may exist, claims of an imminent purge are unsubstantiated.
    - A user clarifies that the new CivitAI policy revolves around automated blocking (via metadata-based review) rather than a mass purge of NSFW content. They report that some of their NSFW videos were automatically blocked after a policy change but were subsequently unblocked upon submitting correct metadata, indicating the system's reliance on content classification rather than outright bans. They also mention the presence of potential false positives in the block system, but see no current evidence for wholesale removal of explicit content.
    - Another commenter discusses backup strategies for models, mentioning backing up over `200+ GB` of LoRAs for Stable Diffusion 1.5 XL and pony models, as well as the use of automated scripts (referenced as a likely existing GitHub project) for category-based downloading. The user notes their local storage size (`~10 TB`) and identifies download speed from CivitAI, rather than their fiber internet or disk space, as the primary bottleneck.
    - There is parallel drawn to past internet censorship events (such as the Tumblr NSFW ban), highlighting importance of model/data preservation for those interested in adult content generation, as users are proactively downloading and archiving their datasets in anticipation of potential policy changes.
- [**What is the preferred substitute for the adult stuff soon to be purged from CivitAI? Where do we move the stuff? We need a Plan B!**](https://www.reddit.com/r/StableDiffusion/comments/1kc4jqc/what_is_the_preferred_substitute_for_the_adult/) ([Score: 195, Comments: 192](https://www.reddit.com/r/StableDiffusion/comments/1kc4jqc/what_is_the_preferred_substitute_for_the_adult/)): **Reddit users are discussing alternatives and contingency plans for hosting adult/NSFW Stable Diffusion models as CivitAI prepares to remove such content. Technical solutions highlighted include the imminent mass upload of archived models and ancillary data to torrent platforms (e.g., 1337x), manual downloading/preservation of metadata, and use of specialized mirrors like [civitaiarchive.com](http://civitaiarchive.com/) and [diffusionarc.com](http://diffusionarc.com/) which feature support for torrenting and metadata ingestion (even for deleted models). Further centralized lists of alternatives and resources are referenced in [this post](https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/), also [archived here](https://archive.ph/https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/).** Technically substantive opinions note the historical precedent of platforms that remove NSFW/content, often leading to user migration and platform decline; users emphasize the need for decentralized hosting to safeguard against single-point failures.
    - Users are creating torrents of the soon-to-be-removed adult models from CivitAI. One mentioned preparing a torrent for release on '1337' and archiving associated metadata, suggesting a decentralized, peer-to-peer backup strategy for model preservation.
    - Alternative hosting sites are emerging, with [civitaiarchive.com](http://civitaiarchive.com/) noted as a dedicated alternative for hosting models, especially LoRAs, displaced by CivitAI's policy changes. There are ongoing efforts to develop further solutions, indicating a fragmented migration landscape.
    - There are shared resources and compiled posts on Reddit (e.g., [this post](https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/), with an [archived version](https://archive.ph/https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)), listing repositories, mirror sites, and current workarounds for accessing or backing up models affected by CivitAI's new restrictions.
- [**Civitai torrents only**](https://www.reddit.com/r/StableDiffusion/comments/1kcb7ge/civitai_torrents_only/) ([Score: 120, Comments: 32](https://www.reddit.com/r/StableDiffusion/comments/1kcb7ge/civitai_torrents_only/)): **A new free tool [datadrones.com](http://datadrones.com/) enables users to generate torrent files and index LoRA models (Limited Rank Adaptation models for diffusion networks) for Civitai-style content sharing, aiming to ensure decentralized, persistent distribution without monetary exchange or central hosting (no UI and minimal file scans for simplicity and compliance). The tool uses a single public tracker by default, with a max file size limit of 2GB per LoRA, and leverages hash-based duplication checks for uploads to enforce uniqueness. The creator explicitly avoids services like HuggingFace due to policy risks and encourages community seeding for longevity, stating private trackers are discouraged. Search functionality and improved scanning are pending; usenet support is considered low priority due to access barriers.** Commenters referenced alternative Civitai archival sites ([diffusionarc.com](http://diffusionarc.com/), [civitaiarchive.com](http://civitaiarchive.com/)) as parallel preservation projects. Debate touched on the simplicity versus feature set balance, with nostalgia for older, decentralized file sharing methods (eMule/edonkey) highlighted.
    - Two alternative sites, https://www.diffusionarc.com/ and https://civitaiarchive.com/, are mentioned as projects specifically focused on the preservation and archival of Civitai content, potentially offering their own access methods and archival strategies distinct from torrent distribution.
    - A maintainer notes that search functionality is already implemented on the site, with plans to add a directory feature next, indicating ongoing technical development toward more robust file navigation and user access features.

### 2. Recent AI Model and Tech Launches & Benchmarks

- [**Google launches the Ironwood chip, 24x faster than the world’s most powerful supercomputer. Is this the start of a new rivalry with NVIDIA?**](https://v.redd.it/wnldw6ib0ywe1) ([Score: 160, Comments: 24](https://www.reddit.com/r/singularity/comments/1kcdlhg/google_launches_the_ironwood_chip_24x_faster_than/)): **Google's newly announced Ironwood chip purportedly delivers 42.5 exaflops of FP8 compute in a cluster of 9216 TPUs, which Google claims is over '24 times faster than El Capitan,' the current leader in supercomputing. However, Ironwood is not available for external purchase and appears intended solely for Google's own data centers, raising questions about the directness of any rivalry with NVIDIA. Further architectural details, independent benchmarks, or implementation specifics are not provided and performance claims remain uncorroborated.** Commenters point out that without external availability, Ironwood does not currently compete directly with NVIDIA or other hardware vendors, and should not be considered a rival at this stage.
    - Discussion highlights that Google's *24x faster* claim is based on a fully scaled cluster of 9,216 Ironwood TPUs, delivering a reported 42.5 exaflops of FP8 performance, which is compared to the FP8 performance metrics of El Capitan, the current top supercomputer. This comparison hinges on a specialized data type (FP8), rather than more traditional FP32 or FP64, emphasizing the chip’s deep learning and AI task orientation rather than general-purpose HPC.
    - Multiple commenters note that Ironwood TPUs do not represent a direct market threat to NVIDIA hardware as they are not commercially available. The lack of open market sales for Google's TPU technology means that, despite its technical prowess, it does not create a real rivalry with NVIDIA’s GPU offerings, which are widely available for purchase and integration.
- [**F-Lite - 10B parameter image generation model trained from scratch on 80M copyright-safe images.**](https://huggingface.co/Freepik/F-Lite) ([Score: 129, Comments: 47](https://www.reddit.com/r/StableDiffusion/comments/1kc2j5g/flite_10b_parameter_image_generation_model/)): **The post announces 'F-Lite', a 10B parameter image generation model reportedly trained from scratch on 80M copyright-safe images. The referenced [Hugging Face model page](https://huggingface.co/Freepik/F-Lite) is currently inaccessible due to repeated** `HTTP 429 Too Many Requests` **errors and does not provide technical documentation or model specifics at this time.** Top comments criticize the model's reported image quality (specifically poor anatomical accuracy) and question the value of a "neutered" or restricted dataset model, suggesting that the resource expenditure is wasteful compared to training more versatile, higher-capability models.
    - Several comments note that F-Lite's anatomy generation is poor, with specific mention that its *"understanding of anatomy looked very bad"*, suggesting significant technical limitations in its training or dataset curation. This observation implies the model struggles with acutely human-sensitive features, often crucial in image generation benchmarking.
    - The value of strictly copyright-safe training data is questioned technically, since *"styles cannot be copyrighted anyway"*. This raises a dataset composition debate: is a copyright-restricted corpus materially limiting model expressiveness or only marginally relevant to output fidelity for most use cases?
    - A question is raised about hardware efficiency: can a 10B parameter model like F-Lite *"run on 8GB"* of VRAM, which is a technical consideration for local and consumer-level deployment compared to large models requiring substantial memory resources.
- [**Livebench has become a total joke. GPT4o ranks higher than o3-High and Gemini 2.5 Pro on Coding? ...**](https://i.redd.it/o28kdmdxq1ye1.jpeg) ([Score: 205, Comments: 62](https://www.reddit.com/r/singularity/comments/1kbt07z/livebench_has_become_a_total_joke_gpt4o_ranks/)): **The image presents a leaderboard from Livebench ([https://livebench.ai](https://livebench.ai/)), showcasing coding performance scores of major AI models such as "o4-Mini High", various OpenAI GPT-4o/3 models, and Google Gemini 2.5 Pro. Notably, GPT-4o and even lower-capacity models like o3-Medium are ranked above more advanced tiers like o3-High and significantly ahead of Google Gemini 2.5 Pro (score: 71.08), questioning the reliability of Livebench's coding assessment. This aligns with ongoing debate about the methodology and validity of Livebench's coding benchmarks.** Technical commenters widely distrust Livebench's coding scores, citing illogical model rankings (e.g., Sonnet scoring lower than less capable models) and a perceived need for more rigorous coding benchmarks. Some also allude to skepticism about the platform's management and data credibility.
    - Multiple users criticize Livebench's coding benchmark; specifically, they question the validity of its results when models such as **GPT-4o** rank higher than **Ollama o3-High** or **Gemini 2.5 Pro** and when 'thinking' models like Sonnet score lower than non-thinking models, highlighting serious concerns about the evaluation methodology.
    - A user notes that Livebench appears to bias toward competitive programming-style tasks, which may not reflect a model's broader coding capabilities and could explain inconsistent rankings against generally stronger 'thinking' models.
    - There is skepticism toward synthetic benchmarks as a whole, with several comments asserting that none of the evaluated models (including the much-hyped Gemini 2.5) perform as poorly or as inconsistently in real-world coding, writing, or context retention as the benchmark suggests.

### 3. Instructional Image Editing and UI Integration Releases

- [**Chroma is now officially implemented in ComfyUI. Here's how to run it.**](https://www.reddit.com/r/StableDiffusion/comments/1kc7jwq/chroma_is_now_officially_implemented_in_comfyui/) ([Score: 221, Comments: 90](https://www.reddit.com/r/StableDiffusion/comments/1kc7jwq/chroma_is_now_officially_implemented_in_comfyui/)): **Chroma, a new model, is now officially integrated into ComfyUI per the [pull request](https://github.com/comfyanonymous/ComfyUI/pull/7355). Guidance on installation includes placing the [ae.sft VAE](https://huggingface.co/Madespace/vae/blob/main/ae.sft), [T5XXL FP16 text encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) (requires >9GB VRAM), and [Chroma UNet](https://huggingface.co/lodestones/Chroma/tree/main) (BF16 mode needs >19GB VRAM) into respective folders. For users with limited VRAM, the [Chroma GGUF variant](https://huggingface.co/silveroxides/Chroma-GGUF/tree/main), complemented by [ComfyUI-GGUF custom node](https://github.com/city96/ComfyUI-GGUF) and optional [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) for RAM offloading, is supported; workflows and style demonstrations (video game, anime, realistic) are linked in the post.** Key technical feedback in comments recommends avoiding the OP's workflow for optimal results: specifically, discard RescaledCFG, use standard samplers like Euler or UniPC, cap CFG to 3-4, simplify negative prompts, restrict prompt weights to below 1.2, update ComfyUI, and properly set the Chroma CLIP loader. Decent speed/quality is noted with as few as 30 steps; substantial improvement over initial impressions is possible with these adjustments.
    - A user provides optimization guidance for Chroma in ComfyUI, noting that the original workflow's poor results are likely due to suboptimal parameters. Key recommendations include: avoid RescaledCFG, use a standard sampler (Euler or UniPC), set CFG to 3-4, keep prompt weights to a max of :1.2 (since ComfyUI is not Auto1111), and simplify negative prompts. Also highlights the need to update Comfy and set the clip loader to Chroma for full functionality. Decent, fast results are achievable starting at 30 steps.
    - Implementation detail: successful use of Chroma in ComfyUI requires users to update ComfyUI and select Chroma as the clip loader, indicating the change is not enabled by default and may require explicit user action.
- [**In-Context Edit an Instructional Image Editing with In-Context Generation Opensourced their LORA weights**](https://www.reddit.com/gallery/1kcbpq8) ([Score: 103, Comments: 11](https://www.reddit.com/r/StableDiffusion/comments/1kcbpq8/incontext_edit_an_instructional_image_editing/)): **ICEdit introduces an instruction-based image editing framework, leveraging LoRA-based weights ([Hugging Face model link](https://huggingface.co/sanaka87/ICEdit-MoE-LoRA)), supporting both multi-turn and single-step edits with high efficiency on tasks like object addition, color change, style transfer, and background edits. The method includes resources such as a HuggingFace demo ([ICEdit demo](https://huggingface.co/spaces/RiverZ/ICEdit)) and a ComfyUI workflow file ([workflow JSON](https://github.com/user-attachments/files/19982419/icedit.json)), indicating practical, modular integration potential for pipeline deployment. No VRAM requirements are explicitly stated in the release materials.**  Technical discussions raised concerns about the provided ComfyUI workflow's functional correctness, with experienced users noting that it appears 'totally messed up,' suggesting issues in the workflow's configuration or compatibility. There is a technical query on VRAM requirements, specifically regarding suitability for 16GB setups, which remains unaddressed by the official documentation.
    - A user raises concerns about the workflow integration for LoRA weights with "Comfy" (ComfyUI), suggesting that its workflow might be problematic or not plug-and-play as expected.
    - Another comment questions the VRAM requirements for running the model, specifically asking if 16GB of VRAM is sufficient, indicating there is uncertainty about resource consumption and effective deployment on GPUs with moderate memory.
    - A technical limitation is highlighted regarding output image resolution, with the model or web demo being restricted to a forced 512-pixel width. Additionally, the commenter notes that the web demo's output quality is highly inconsistent, suggesting potential stability or inference issues.

---

# AI Discord Recap

> A summary of Summaries of Summaries by chatgpt-4o-latest
> 

**1. Phi-4 Reasoning Model Release**

- **Microsoft Revs Engine on Phi-4 Reasoning Model**: **Microsoft** released a new model named [**Phi-4-reasoning**](https://huggingface.co/microsoft/Phi-4-reasoning), an advanced 14B parameter LLM designed for reasoning tasks, also available at [Unsloth's local GGUF version](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF) and linked to [this blog post](https://x.com/UnslothAI/status/1917806961825046672).
    - Community sentiment was overall positive, noting performance **on par with GPT-3.5 size models**, and speculation arose that Phi-4 was trained with **OpenAI Chain-of-Thought outputs**; further info was explored in a [YouTube video](https://www.youtube.com/watch?v=5aN4Xg0VvCs) and [arXiv paper](https://arxiv.org/abs/2504.21318).
- **Phi-4 Reasoning Models Go Local with Unsloth**: **Unsloth AI** made **Phi-4-reasoning** models locally accessible by publishing GGUF-format versions on [Hugging Face](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUFChatGPT), minimizing barriers to experimentation for those running offline models.
    - The model runs with existing **Phi-4 notebooks**, and early testers praised its **concise reasoning output**, fueling comparisons with Phi-3 and prompting early finetuning discussions.

**2. Diffusion Language Models & Architecture Innovations**

- **Mercury Coder Diffuses the Competition**: **Inception AI** launched [**Mercury Coder**](https://openrouter.ai/inception/mercury-coder-small-beta), the first publicly available **diffusion language model**, boasting **300+ tokens/second** and advertised as competitive with **GPT-4o Mini** and **Claude 3.5 Haiku**.
    - Its **parallel token refinement** architecture (diffusion-style decoding) is claimed to reduce hallucinations and boost reasoning ability; [OpenRouter shared the details](https://x.com/OpenRouterAI/status/1917677801211322752) with community interest swiftly spiking.
- **DSPy Crushes Hallucinations with Chemistry Prompts**: [A new paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02322) in the *Journal of Chemical Information and Modeling* shows that optimizing DSPy programs for **TPSA prediction tasks** reduced hallucinations in chemical reasoning by **81%**.
    - The results impressed the community, signaling DSPy's potential for **stable scientific reasoning** and adding to its growing adoption in **Amazon Nova migration** and **Meta's LlamaCon announcements**.

**3. Qwen3 Model: Breakthroughs and Bugs**

- **Qwen3 4B Smashes Its Size Class**: **Qwen3 4B** earned praise for dominating its size group with standout performance in **math and coding tasks**, with a user claiming [“qwen 3 4b destroys everyone at that size range”](https://fixupx.com/suriyagnskr/status/1917731754515013772).
    - Multiple Discords praised its **compact size with huge returns**, with benchmarks from **ContextArena.ai** further verifying its edge in low-context-length evaluations.
- **Qwen3's Norm Layers Vaporize on Merge**: A bug was identified in Qwen3 LoRA merges where **q_norm** and **k_norm** layers were silently excluded, requiring the addition of those layers to the **LLAMA_LAYERNORMS** in `save.py`.
    - Users shared successful fixes after hours of debugging, highlighting that this uniquely affects **Qwen3, not Qwen2**, and proposed submitting a patch to **Unsloth**.

**4. Multi-Model Ecosystem Showdowns & Scaling Benchmarks**

- **Context Arena Crowns Llama, but Qwen Surges**: **ContextArena.ai** released updated multi-model leaderboard results with **Llama 4 Maverick** claiming top AUC at **128k context** while **Qwen3-235B-A22B** outperformed at shorter contexts.
    - **Anthropic** models showed consistent performance across **Claude 3.x** generations, with community members dissecting model falloff patterns as context lengths approached limits.
- **Gemini vs GPT vs Claude Sparks Turf War**: Across multiple communities, users debated tradeoffs between **Gemini 2.5 Pro**, **Claude**, and **GPT-4o**, citing Gemini’s strengths in **critical thinking** and backend coding, but also complaints about its UI and token inefficiencies.
    - Opinions split on cost-value ratios, with one user describing GPT as *'a fake friend following your tone'* and Gemini as *'a real professional with judgment’*, cementing the battle for mindshare in developer workflows.

**5. Claude's Expanding Capabilities and Integrations**

- **Claude Integrates Across Toolchains**: **Anthropic’s new [Claude Integrations](https://www.anthropic.com/news/integrations)** allow users to connect custom SSE endpoints directly in Claude’s interface, enabling live tool chaining for complex workflows.
    - Users praised the [clarifying guide](https://x.com/alexalbert__/status/1918047745790914772) explaining how to input custom **SSE URLs**, streamlining integrations compared to workarounds used with agents.
- **Claude Connects to DeepWiki MCP Agent Workflows**: A powerful new combo was shared: connecting **Claude** to the **DeepWiki MCP** server via the `fetch` command, using the endpoint for web-scale retrieval in research scenarios.
    - Users described the setup as a *game-changer*, citing [DeepWiki's repo](https://github.com/regenrek/deepwiki-mcp) and **Claude’s new agent endpoints** as unlocking truly agentic, contextual query chains.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini vs. O3 sparks debate**: Members debated the relative merits of **Gemini** and **O3**, with opinions ranging from *"O3 is objectively better bruh"* to jokes about saving up for **Gemini Ultra**.
   - The discussion highlighted the subjective nature of model preference, even with objective benchmarks available.
- **Qwen3 excels, Llama 4 reigns in Context Arena**: DillonU's **OpenAI-MRCR** benchmark in **Context Arena** revealed that **Llama 4 Maverick** leads with the highest AUC score at 128k context length, available at [ContextArena.ai](https://contextarena.ai/).
   - Qwen3-235B-A22B outperformed at lower context lengths, but its performance rapidly declined near its limit.
- **Anthropic scores big in Context Arena**: The **Context Arena** now includes more **Anthropic** results for 2needle tests, showing consistent performance across **Claude 3.0**, **3.5**, and **3.7**, available at [ContextArena.ai](https://contextarena.ai/).
   - **Claude 3.0 Haiku** achieved the best overall Model AUC.
- **ChatGPT's Deep Research Triumphs**: Members compared deep research tools from **ChatGPT**, **Claude**, and **Grok**, noting that **ChatGPT's** deep research capabilities surpass **Grok's** due to **Grok's** reliance on **DuckDuckGo**.
   - The consensus was that **ChatGPT** offers a more polished and effective research experience.
- **Qwen3 4B Destroys Competitors in its class**: The **Qwen3 4B** model is being lauded for its performance relative to its size with a member stating that [*qwen 3 4b* **destroys** *everyone at that size range*] with further discussion available [here](https://fixupx.com/suriyagnskr/status/1917731754515013772?t=yQeTFTkCfRkl0ZhQJ2k-tQ&s=19).
   - It was noted that **Qwen3 4B** demonstrates particular strength in *maths and coding* tasks.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's UI Shifts Overnight!**: Users reported a new UI design change that occurred overnight, with some reporting disappearing libraries and changing shortcuts.
   - The reaction is mixed, with some users finding the new UI buggy and others remaining neutral.
- **Qwen dethrones Context Length!**: Members found out that **Qwen** boasts great context length.
   - It's considered good enough as a free model to compete with older **ChatGPT** models, and some even prefer it over paid options.
- **Zen Browser's Transparency Tricks!**: Members discussed how to use the **Zen Browser** to create a transparent background and recommended the **Nebula** theme from [github](https://github.com/JustAdumbPrsn/Nebula-A-Minimal-Theme-for-Zen-Browser/releases).
   - Configuration steps vary, with some users seeking help from [Zen's subreddit](https://www.reddit.com/r/zen_browser/s/uRWOeML6n8).
- **Grok's Image Processing Hit!**: A member observed that [**Grok AI** on Twitter](https://www.rxddit.com/r/grok/s/w5jc52QFj5) faces limitations in image processing.
   - Its image processing capabilities are, according to them, only at **R1** level.
- **Tesla Board Seeks CEO!**: A shared [Perplexity search result](https://www.perplexity.ai/page/tesla-board-seeks-new-ceo-to-r-3JZ4nGLOQ6S40o59qn92Wg) suggests the **Tesla** board is searching for a new CEO.
   - The search is influenced by ongoing discussions about **Elon Musk**'s role and the future leadership of the company.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Image Support Missing from Llama-4-Scout GGUF**: Users reported that **Unsloth's Llama-4-Scout-17B-16E-Instruct-GGUF** model failed to interpret images when using **Q4_K_XL**, though text worked properly.
   - Doubts were raised whether non-Meta official tools correctly encode images, and a check of tokenizer configs and vocab was suggested.
- **Microsoft Releases Phi-4 Reasoning Model**: **Microsoft** released the **Phi-4-reasoning** model, available at [huggingface.co/microsoft/Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning), regarded as competitive for a 14B parameter model.
   - The model may be trained on **OpenAI CoT** output, and the Unsloth team stated it will work with regular **Phi4** notebooks.
- **GRPO Enables Self-Explanation in Models**: A member suggested using **GRPO** (Generative Reward Policy Optimization) to train models, like **Claude**, to improve self-explanation, particularly when addressing inaccurately described problems.
   - This could enhance a model's ability to understand and address complex issues by first clarifying the problem for itself.
- **Qwen3 GGUF Gets a Quick Fix**: The **Qwen3-30B-A3B 128K** GGUF was re-uploaded to fix issues, with the context length value reset to **32k**, requiring a rope scaling factor of **4** in LM Studio.
   - The team appreciated the community's help in identifying the issue that only affected the **30b-a3b** model while other **128k** GGUFs were unaffected.
- **Qwen3's Norm Layers Vanish During LoRA Merge**: An issue was identified where the **q_norm** and **k_norm** layers were not saved when merging a trained LoRA with the base **Qwen3** model.
   - The fix involved adding **q_norm** / **k_norm** to **LLAMA_LAYERNORMS** in **save.py**, affecting **Qwen3** but not **Qwen2**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **HF Tariffs spark IDE switch**: A user joked about switching from **VSCode** to **NeoVim** due to implied tariffs from [HuggingFace](https://huggingface.co/).
   - The ensuing discussion highlighted **NeoVim**'s *speed and terminal integration* versus **VSCode**'s *mouse support and ease of use*.
- **GPU Database Dreams stall in middleware**: A user referenced **GPU MODE @ GTC2025** citing challenges in using **GPUs** for databases due to *lack of middleware* to translate database language into **GPU code**.
   - They inquired about open-source projects addressing this, seeking contribution opportunities.
- **MI300 MoE Model Makes Milestone**: A user reached **first place** on the `amd-mixture-of-experts` leaderboard on **MI300** with a time of **604 ms**.
   - Later submissions landed in **5th place** at **7382 ms** and **9th place** at **9246 ms** on the **MI300**.
- **Community Kernels close in on Optimized CUDA Kernels**: Members mentioned that you can determine how close you are to theoretical peak using **Triton**, without needing a **CUDA/Cutlass** version.
   - It was also mentioned that the community is developing faster kernels for a wide variety of functions instead of waiting for **AMD/Nvidia** to do this.
- **AMD GPU underperforms with Pytorch SDPA**: A user reported that [PyTorch SDPA](https://github.com/pytorch/pytorch/issues/152595) is **2.5x slower** than a manual PyTorch implementation on a 7900 XTX, while an NVIDIA GeForce RTX 4090 reported that *F.scaled_dot_product_attention* is actually **faster** than the manual implementation.
   - This suggests it is an AMD-specific issue.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mercury Coder Enters the AI Race**: **Inception** launched **Mercury Coder**, the first diffusion LLM, claiming to rival **GPT-4o Mini** and **Claude 3.5 Haiku** in code quality, boasting a speed of **300+ TPS**; it can be tried [here](https://openrouter.ai/inception/mercury-coder-small-beta).
   - Its diffusion architecture refines tokens in parallel, potentially reducing hallucinations and improving reasoning, according to [this announcement](https://x.com/OpenRouterAI/status/1917677801211322752).
- **Gemini 2.5 Pro Gets Counting Fix**: The Vertex team fixed the upstream token counting issue with **Gemini 2.5 Pro** and **Flash Preview** models, re-enabling the model on OpenRouter.
   - Caching on **Gemini 2.5 Pro Preview** is temporarily disabled as usage and costs are evaluated to prevent over-billing coming from upstream (**AI Studio** and **Vertex**).
- **Vanna.ai Opens Doors to DB Insights**: [vanna.ai](https://vanna.ai/), an open-source tool for working with **SQLite DBs**, was highlighted and demonstrated by a member, showcasing its ability to generate work orders from stock levels and priorities.
   - The tool was recognized to be so useful that the member forked a private version for their own business needs.
- **Phala Promises Privacy with AI Endpoints**: **Phala** launched confidential AI endpoints on OpenRouter, with plans to incorporate full end-to-end encryption (e2ee) to the enclave in the future.
   - The team is considering **Oblivious HTTP** and similar technologies for future encryption, as discussed in [this article](https://x.com/FreedomTechHQ/status/1917689365632893283).
- **Amazon's Nova Premier Shows Its Hand**: Amazon debuted **Nova Premier**, its most capable model including **Nova Canvas** for image generation, and benchmarks and pricing were shared.
   - Though some members were underwhelmed by the benchmarks, the model's potential for **seamless integration** between components and end-to-end agentic workflows was emphasized, further elaborated [in this video](https://youtu.be/Bh-sQYePjRs).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro > Claude?**: Members debated their preferences between **Gemini 2.5 Pro in architect mode** and **O3 in ChatGPT**, with some finding **Gemini** confusing and costly, while others favored **O3's** web search and concise responses.
   - One user reported success using **udiff-simple** with **Gemini**, stating they did not use architect mode and expressed overall happiness.
- **Claude Code Proxy encounters Maintenance Issues**: Multiple members reported that the [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) project is no longer working or maintained after recent updates to **Claude Code**.
   - No alternatives or fixes were provided, indicating a need for a new solution for proxying **Claude Code**.
- **Groq Speed Limited; Deepseek R1 Missing**: A member questioned the absence of the full **Deepseek R1** on **Groq**, despite **Groq** hosting R1 distills, speculating this decision might be related to circumventing the *"no china" angle.*"
   - While users find **Groq** fast, some note its limitations to *"dumb models"* for free users, impacting its utility for complex tasks.
- **Aider as an MCPaaS?**: Following Anthropic's unlocking of Claude Code, a member suggested **Aider** as a potential MCP, sparking discussion about starting a remote **MCPaaS** business.
   - A [YouTube video](https://www.youtube.com/watch?v=QzZ97noEapA) on cracking **Aider** and **Claude Code** was shared, humorously highlighting the interest in leveraging these tools commercially.
- **Aider and Ollama performance problems**: A user experienced significant delays using `aider` with local LLMs like **Qwen3** via `ollama_chat/`, citing startup times exceeding a minute and code block generation taking multiple minutes, and discovered the delay was on the `ollama` side, with message processing taking over **22 minutes**.
   - Despite `ollama run` being responsive, the integration with `aider` introduced substantial overhead, indicating potential optimization issues in the `ollama`.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Seed-VC Clones Voices Effortlessly**: [Seed-vc](https://huggingface.co/spaces?category=voice-cloning&sort=trending) is recommended for **voice conversion**, especially for quickly cloning voices with minimal audio, unlike **RVC**.
   - One user stated that RVC requires approximately *40 minutes of audio* and *takes days* to process, making **Seed-VC** a faster alternative.
- **Unsloth Unleashes Microsoft's Phi-4 Models**: Unsloth has uploaded **Microsoft's** new **Phi-4** reasoning models, enabling local runs via [this HuggingFace link](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF).
   - The models are now accessible for local use, highlighted in [this announcement tweet](https://x.com/UnslothAI/status/1917806961825046672).
- **Stuck Testing RAG Pipelines?**: A member is developing a tool to isolate and test context slices in memory- or **RAG**-heavy setups to optimize responses, seeking feedback on its potential usefulness via [this tweet](https://x.com/HuggingPapers/status/1917831613548802349?t=7W2pCoiE9kMcP9tnv7l8Bg&s=19).
   - The tool aims to address the challenges of optimizing **LLM** calls by enabling detailed analysis of individual context slices.
- **Managed Agents: Final Answer still needed?**: A user inquired whether **Managed Agents** require the **Final_Answer tool** and is experiencing **kwarg errors** when using the tool, indicating potential issues with the recent updates.
   - Another member mentioned pinning a library to **version 1.13.0** in their *requirements.txt* file to ensure functionality, due to **compatibility issues** or **breaking changes** in later versions.
- **Smolagents trips over Assertion Error**: Users are encountering an `AssertionError` related to missing prompt templates when trying to get `smolagents` working with the tutorial at [Huggingface](https://huggingface.co/learn/agents-course/unit1/tutorial).
   - A fix involves setting the version of `smolagents` to `1.13.0` in the `requirements.txt` file, and upgrading the `gradio UI`.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4 Retires After 779 Days**: **GPT-4** has been retired after **779 days** since its release, leading to discussions about its replacements like **4.5 research preview** and **4o**.
   - Some users feel **GPT-4** had become outdated and performed worse than newer models, with one user stating that *'GPT-4 started to sound like 3.5 anyways'*.
- **'Granny-crusty-nun' Content Filter Irks Users**: Users are frustrated with an overly restrictive content filter, jokingly called **'Granny-crusty-nun'**, which blocks harmless actions like *'hugs'* and flags innocuous AI-generated images.
   - One user reported that even the AI seems exasperated, generating outputs like, *'seriously?! we specifically said (this this and this) to PREVENT that!! whats with this pervy granny-inkwells!?'*.
- **Gemini 2.5 Pro Triumphs with Critical Thinking**: Users praise **Gemini 2.5 Pro** for its superior ability to provide balanced perspectives and critical thinking compared to **GPT-4o**, especially in fields such as medical study.
   - One user described **GPT** as *'a fake friend who just follow your tone'*, while **Gemini 2.5 Pro** is *'like a real professional with lots of critical thinking and judgment'*.
- **Excessive Token Consumption Troubles Users**: A user complained about **excessive token consumption** in the free plan of **GPT** due to the model rewriting code instead of providing a final output.
   - The inefficiency has led to them questioning whether to purchase a **Plus or Pro plan** and is considering **alternatives or local models**.
- **ChatGPT Prompting for Room-Temperature Superconductors**: Members are crafting prompts for material science research to discover **room-temperature superconductors** by defining material properties such as **conductivity**, **magnetism**, and **atomic structure**.
   - Prompt engineering includes defining material properties (**conductivity**, **magnetism**, **mechanical properties**, **atomic structure**, **optical properties**, **thermal conductivity**).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini 2.5 Pro Dominates Backend**: Users are impressed with **Gemini 2.5 Pro**, describing it as *wild* and particularly effective for backend tasks, especially with **Swift**.
   - One member suggested it might outperform **Sonnet 3.5**, while another alternates between **3.7 Sonnet max** and **Gemini max** for optimal results.
- **China's Benchmark Saturation Model Global Domination?**: Concerns are rising about **China** potentially dominating AI globally by creating benchmark saturation models optimized for their chips, without competition from **US/EU** models.
   - A user shared a [tweet](https://x.com/goose_is_goofy/status/1917621990023627193?t=XnMgX-Mfd-Ax3KNWmNU8ug) framing this as *China 2025 vs. the world*.
- **Cursor steers toward becoming AWS of AI Editors?**: Speculation suggests **Cursor** might become the *AWS of AI editors* due to its pricing model, with users preferring a credit system over pay-as-you-go.
   - Users voiced concern that Cursor is *going toward the nickel and diming route just like AWS*, referencing [pricing details](https://docs.cursor.com/settings/models#available-models).
- **DeepWiki MCP Fetch is game changing**: A user reported a *game changer* by combining **DeepWiki**, a new MCP server, with the tool call **fetch**.
   - They linked to the [DeepWiki website](https://deepwiki.com/) and [DeepWiki Repository](https://github.com/regenrek/deepwiki-mcp), noting its effectiveness when used correctly.
- **Claude Code Max Plan is the God Tier?**: Users find **Claude Code** with a **Max Plan** transformative, with one declaring that *Cursor for small edits + Claude code Max is the god tier combo*.
   - Estimates suggest the **$100 Claude Max plan** allows for around **7M tokens** every 5-6 hours, and the **$200 plan** allows for **4x** more, though some find the max models within Cursor overpriced by comparison.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Role Playing Improves Reasoning**: Members debated the impact of removing or reducing **RP data** on smaller models, with one side emphasizing its importance for grounding reasoning in user interactions.
   - Concerns were raised that **RP data** sometimes induces hallucinations, suggesting that the right prompting for larger models like **Sonnet** may not directly translate to more compressed representations.
- **Defining the Boundary of Small Models**: The size threshold for *small* models was discussed, with opinions ranging from **7-8B** parameters for decency to **3B** for good personality.
   - For converting scientific papers to epub format, [tex4ebook](https://tex4ebook.readthedocs.io/en/latest/) was recommended.
- **Minos Misclassifies Refusals**: The community found that **Minos** incorrectly classified some non-refusals when evaluating model refusals using the Chinese refusal list ([deccp dataset](https://huggingface.co/datasets/augmxnt/deccp)).
   - The team is planning to expand the categories beyond refusal and non-refusal for v2, see [this discussion](https://huggingface.co/NousResearch/Minos-v1/discussions/5).
- **Nous Enters 405B FFT Arena**: The team announced its entry into the **405B FFT club**, highlighting the challenges of training such a large model, including the use of 32 nodes and ring attention.
   - While the model did not surpass Deepseek V3, the effort facilitated advancements in smaller models, despite being significantly more compute-intensive than training a 70B model.
- **Nous Explores Decentralized AI**: A member shared a [Medium article](https://medium.com/@abdulazeez600/nous-research-pioneering-decentralized-ai-for-the-future-a7042a785493) highlighting **Nous Research's** initiatives in **decentralized AI**.
   - The article, considered *self promotion*, sparked interest and discussion within the group.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 3 Toggle Implementation In Progress!**: Members discussed adding a feature to **toggle 'thinking' on/off** for the **Qwen 3** model in LM Studio.
   - Currently, there's no built-in toggle, and users are advised to use `/no_think` in the system prompt as a manual workaround.
- **Flash Attention Accelerates Self-Attention**: **Flash Attention** optimizes memory access and rearranges operations to reduce memory needed for self-attention, [avoiding the storage of large matrices](https://chatgpt.com/share/6812b811-a1d4-8011-8c62-da556fd6e9bd).
   - Quantizing the KV caches with **Q8** cache *increases* the context window.
- **Llama4 2T Model to be Smashed!**: A member is planning a build to "smash new **Llama4 2T** in Q8.0 and million context" using **DDR5 offloading** or a fully offloaded **671b Deepseek** model, with a detailed parts list including **AMD EPYC 9755 QS** CPUs, **NVIDIA RTX PRO 6000 Blackwell** GPUs, and **2304GB of DDR5 ECC RAM**.
   - The total cost of the system is estimated to be around **81,424 EUR**.
- **Multi-GPU Setups Suffer Performance Hit**: A member asked about performance improvements using multi-GPU setups with **LM Studio**, and another member responded that performance *declines substantially* when going from a model that fits on 1 GPU to one that requires 2 GPUs, and is *only be utilized about 1/2 the time*.
   - However, **vLLM** might provide some performance improvements on **Nvidia**.
- **Bypass Apple Memory Caps via Terminal!**: A member clarified that macOS allows allocating up to **120GB VRAM** on a **128GB Mac Studio** using the terminal, countering the belief that only 75% of unified memory can be used, without resorting to hacking.
   - They suggested that *apple only allows to allocate up to 75% of the unified mem*, so aiming for a 192gb mac is better if you're running q4.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Podcast Embeds Requested by Users**: A user is seeking to embed an interactive podcast audio player on their website, mirroring the functionality found on [sujankhadgi.com](https://www.sujankhadgi.com/).
   - This would allow for a more engaging experience with podcast content.
- **LaTeX Troubles Try AP Calc Students**: Users reported that **Notebook LM** generates extra symbols while creating FRQ tests for AP Calc.
   - A suggested workaround is to request that the model avoids writing in **LaTeX**.
- **Unpublished Research Input: Proceed with Caution**: A user cites faculty bloggers warning caution regarding the use of **unpublished research** in **NotebookLM**.
   - The risks related to intellectual property need further study.
- **Bulgarian TTS Botches Stress**: Users have reported that Google's TTS mispronounces Bulgarian words in **NotebookLM**, specifically related to **stress placement**.
   - Bug reports can be submitted in the [bugs channel](https://discord.com/channels/1124402182171672732/1366873891938504827).
- **NotebookLM Plus Plagued by PDF Problems**: Users noted that **NotebookLM Plus** accounts fail to load PDFs, displaying a red error banner, while free accounts load the same PDFs successfully.
   - The community suspects this may be a widespread issue, due to increased reports of sharing issues.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **LLMs Struggle with Syntax**: Users are wondering why **LLMs generate syntax errors** despite not being trained on data with syntax errors.
   - A user suggested that **system prompts** and memory banks may influence this behavior.
- **Reasoning Paper Sparks Call Center Robot Ideas**: A user shared [Instruction Following via Step-by-Step Reasoning](https://arxiv.org/abs/2310.10158), noting its relevance to **building a call center robot persona**.
   - It uses *an mcp server to a memory bank to recall things that go beyond its context window*.
- **Tabnine Gives Bad Minecraft Advice**: A user reported that the **Tabnine AI agent** incorrectly advises reverting to outdated Minecraft code.
   - The user jokingly expressed frustration by stating: *AaaaaaaaaargCan America just stop being dumb for one day? No?*.
- **Manus Fellowship Program Reopens**: The **Fellow Program** has been reopened, announced with a [YouTube video](https://youtu.be/Tz1Of7ltnMY?feature=shared).
   - Discussion followed on what the program was and how to get in.
- **Subscription Credits Expire in Manus**: A user asked for clarification on **credit expiration** for monthly subscriptions in Manus.
   - A staff member clarified that subscription credits expire monthly, while bonus credits do not expire while the subscription is active, and subscription credits are used first.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Lean Cursor Integration Compatibility Questioned**: Members explored setting up **Lean** with **Cursor**, but weren't sure whether **VSCode plugins** would be compatible, and [shared a ChatGPT link](https://chatgpt.com/share/68127a52-1b34-800f-a535-b74b4ab8f613).
   - The link may not fully address the setup issues.
- **Geometric Deep Learning Celebrates 4th Anniversary**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/petarvelickovic_four-years-ago-the-geometric-deep-learning-activity-7322770901958062080-KBp) commemorating the **4th anniversary** of **Geometric Deep Learning**.
   - Members celebrated the field's progress and the loss of GPT-4.
- **Perception Encoder Embeddings Alignment**: Discussion continued on the **Perception Encoder (PE)** paper, especially Section 4 on alignment methods for language and spatial understanding to extract strong embeddings, as detailed in [this PDF](https://scontent-bos5-1.xx.fbcdn.net/v/t39.2365-6/491405782_553183477404780_6476813073924059281_n.pdf#page=14).
   - The paper suggests contrastive vision-language training yields strong embeddings with proper alignment and a robust video data engine, according to [Meta's Research Publication](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network/).
- **Phi-4 Reasoning Model Surfaces**: Microsoft's **Phi-4-reasoning** model was shared with links to the [YouTube video](https://www.youtube.com/watch?v=5aN4Xg0VvCs), [Arxiv paper](https://arxiv.org/abs/2504.21318), and [Hugging Face page](https://huggingface.co/microsoft/Phi-4-reasoning).
   - A link to [unsloth/Phi-4-reasoning-plus-GGUFChatGPT](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUFChatGPT) was provided.
- **LLMs Escape Croatian**: A user shared a link about **ChatGPT** temporarily ceasing to speak Croatian, [referencing a tweet](https://x.com/georgejrjrjr/status/1917722125668081863).
   - Another user said *I have experienced LLM's give up on trying before...and just start changing random things till they get frustrated with the user and walk away.*



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Playground** goes live for testing and debugging**: Lu Xian announced an open-source **MCP Playground** [on GitHub](https://github.com/rosaboyle/mcp-playground) for connecting, testing, and debugging local MCPs, highlighting integrations with **Perplexity** and **Firecrawl**.
   - The team is also developing a **Remote Serverless MCP Hosting Platform** and seeking feedback from the community.
- **C# SDK** trips over streamable HTTP**: A developer encountered an issue with the **C# SDK** while trying to set up streamable HTTP, discovering that the *'WithHttpTransport'* definition was missing from the latest **NuGet** release despite being present in the SDK repo.
   - The developer opted to use **STDIO** temporarily, due to being *too lazy* to package it up themselves.
- **LLMs** select tool with function calling**: When using multiple MCP servers, the LLM uses aggregated tool signatures to decide which tool to call, with the MCP client responsible for routing the call to the appropriate server.
   - This approach avoids modifying code for each LLM API by adapting MCP tool types to LLM API tool types, ensuring the LLM always has access to the latest tool list.
- **Anthropic Integrations** cleared up for community**: Members shared a [link](https://www.anthropic.com/news/integrations) to **Anthropic's** new **Claude Integrations**, and a clarifying [X post](https://x.com/alexalbert__/status/1918047745790914772) emphasizing the ability to directly input a SSE transport URL into the Claude.ai web interface.
   - This simplifies connecting **Claude** to external tools and services.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Users Amazed by Hallucinations on X**: A user shared an [amazing hallucination](https://x.com/nabeelqu/status/1917677377364320432?s=46) found on X, showcasing the creative outputs of AI.
   - This example highlights the unpredictable, yet sometimes delightful, results of AI models pushed to their limits.
- **American Positivity Inspires on X**: A user shared [based american positivity](https://x.com/georgejrjrjr/status/1917722125668081863) found on X, promoting an optimistic view.
   - In addition, another user shared [a YouTube video](https://www.youtube.com/watch?v=hFlF33JZbA0) supporting this viewpoint.
- **Anthropic's Claude Connects to Your World**: **Claude** can now [connect to your world](https://www.anthropic.com/news/integrations), granting users greater command over tools and prompts for thorough research.
   - This integration aims to deepen interaction with AI, providing more control over how it assists in various tasks.
- **SWEs Gain Free AI Coding**: An SWE shared [their project](https://x.com/olivierddr/status/1917981301732171934?s=46&t=yBt-W1FZSUMGKfO1SUFWww) offering free alpha access to AI-assisted coding tools for building production ready code.
   - This initiative seeks to democratize access to advanced coding assistance, enabling more developers to leverage AI in their workflows.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Scale Prediction Paper Scores ICML Acceptance**: The paper '[Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?](https://arxiv.org/abs/2406.04391)' has been accepted to ICML, with the PDF available at [this ArXiv link](https://arxiv.org/pdf/2504.07986).
   - The paper acceptance concludes a year of back and forth with reviewers.
- **Humans are Linear Attention Models?**: A member proposed that humans function as **linear attention models**, continuously reasoning in latent space, with output from the last layer (without the LM head) feeding into the first layer and backpropagation through time (BPTT) applied.
   - Another user suggested directing such discussions to an [alignment channel](https://discord.com/channels/729741769192767510/964104737005916240) for further exploration.
- **Zero Loss Flags Data Leakage?**: A member reported encountering **zero loss** during a continual pretraining run and suspects **data leakage**, as the workflow functions correctly with a different dataset.
   - A screenshot [[Screenshot_From_2025-05-01_00-41-57.png](https://cdn.discordapp.com/attachments/747850033994662000/1367385925889429544/Screenshot_From_2025-05-01_00-41-57.png?ex=68150da1&is=6813bc21&hm=5108b6e8c66cf91050ebb336c8ba49179bf93866cf696794290490b115bf85c5&)] depicts the loss dropping to zero during training.
- **SFTTrainer Sparks Zero Loss Scrutiny**: A member sought advice after experiencing a zero loss issue using the **SFTTrainer** from Hugging Face, which did not occur with another dataset.
   - Suggestions included checking **token shifting** and **padding**, and considering differences in the length distribution of the datasets.
- **LLM Augmentation Implicated in Loss Anomaly**: A member speculated that LLM generated data might contribute to **zero loss**, referencing a paper ([arxiv.org/abs/2504.21463](https://arxiv.org/abs/2504.21463)) on augmentation via LLMs, where raw text is summarized or transformed.
   - The discussion revolves around the potential impact of data augmentation techniques on model training dynamics and outcomes.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Meta Cranks Llama Prompts at LlamaCon with DSPy**: At **LlamaCon**, **Meta** announced *llama-prompt-ops*, a Python package that transforms prompts optimized for **Llama models**, built in **DSPy** and via **MIPROv2** optimizer; code available at [github.com/meta-llama/llama-prompt-ops](https://github.com/meta-llama/llama-prompt-ops).
   - The announcement was also [tweeted by the DSPy account](https://x.com/DSPyOSS/status/1917738506732069052).
- **Amazon Migrates to Nova with DSPy Power**: **Amazon AWS** introduced an architecture to migrate from various models to **Amazon Nova** models using **DSPy** and its **MIPROv2** algorithm, as detailed in [this blog post](https://aws.amazon.com/blogs/machine-learning/improve-amazon-nova-migration-performance-with-data-aware-prompt-optimization/).
   - This news was also [tweeted by the DSPy account](https://x.com/DSPyOSS/status/1917419206171320769).
- **DSPy Beats Hallucinations in Chemical LLMs**: A new paper in the **Journal of Chemical Information and Modeling** demonstrates that building and optimizing a **DSPy** program to reduce **RMS error** for predicting topological polar surface area (**TPSA**) of molecules by **81%** reduces chemical hallucinations, detailed in [Augmented and Programmatically Optimized LLM Prompts Reduce Chemical Hallucinations](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02322).
   - This represents a significant stride in making LLMs more reliable for scientific applications.
- **DSPy 3.0 Roadmap is Incoming**: DSPy 3.0 will be a pair of paradigm shifts, it's not public right now but should be released in a month.
   - The release promises substantial advancements, though specific details remain under wraps for now.
- **DSPy Sees Visions for VLMs**: When asked about using Vision Language Models (VLMs) with DSPy, processing a list of images *may work*.
   - Further experimentation is needed to confirm and optimize this functionality within the DSPy framework.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Builds BabelFish**: **LlamaIndex** is constructing a Retrieval-Augmented Generation system capable of managing multiple languages and modalities with [Qdrant Engine](https://t.co/pe9iiMt21W).
   - The system is designed to ingest and retrieve text in **English, Spanish, Chinese**, and other domain-specific content.
- **Databricks & KPMG Invest in LlamaIndex**: **LlamaIndex** has received investments from **Databricks** and **KPMG**, underscoring its practical impact on AI implementation.
   - Additional details regarding how **LlamaIndex** is enhancing agentic document workflows can be found here: [Agentic Document Workflows](https://t.co/ARyxXeVj7F) and [Another Link](https://t.co/LKcoDUAajl).
- **Invoice Reconciliation Agent Automates Compliance**: **LlamaIndex** is targeting tangible use cases for agentic document workflows via the release of a full-stack Invoice Reconciler tool.
   - This tool is engineered to automatically verify invoices against predefined terms.
- **LlamaIndex Needs Chat Template Fix**: A user sought advice on utilizing `chat_template` from Hugging Face tokenizers in LlamaIndex to evaluate new **Qwen3 models**.
   - A community member noted the absence of required kwargs in the `HuggingFaceLLM` class, recommending a **PR** and referencing the [LlamaIndex code](https://github.com/run-llama/llama_index/blob/1bd60497ac3442f6a5b3e787ef3662e572d8d0d4/llama-index-integrations/llms/llama-index-llms-huggingface/llama_index/llms/huggingface/base.py#L309).
- **LLMs Throw Curveballs with Bad Dumps**: A user reported encountering a `"Str" object has no attribute model dump json` error when repeatedly using the same prompt.
   - Another member clarified that **LLMs are non-deterministic**, particularly with intricate schemas, and suggested implementing `try/except` blocks to manage errors.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Auth0 Amplifies AgentX Authentication**: Auth0 sponsors a workshop on authentication in agentic AI applications and offers up to **$5,000** in prizes for the [Entrepreneurship Track](https://auth0.com/ai).
   - Workshop includes best practices, Auth0 integration, security, and live demos; registration is available [here](https://lu.ma/AgentX-Auth0).
- **AgentX Defines Submission Standards**: Submission guidelines for the Entrepreneurship and Research tracks are available on the [AgentX website](https://rdi.berkeley.edu/agentx/#submissions).
   - Entrepreneurship Track requires a pitch deck, demo video, live product link, and optional technical appendix; Research Track needs a scientific paper, video, and GitHub repository; submissions are due **May 31st** at **11:59PM PDT**.
- **Course Assignments Get Discovered**: All assignments are located at the bottom of the [course website](https://llmagents-learning.org/sp25), per member clarifications.
   - The remaining assignment, the **labs**, are slated for release either today or tomorrow, depending on available time.
- **MOOC Lectures Optional for AgentX**: Participation in the **MOOC** is not required for **AgentX**, a member clarified.
   - A member shared the [signup link](https://forms.gle/9u6HdVCWXgws16go9) and [course website](https://llmagents-learning.org/sp25) where recordings are available, noting that **assignments are due end of May**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Macs to Boast TBs of RAM?**: Members are eyeing **Macs** with up to **512 GB of RAM**, foreseeing future models demanding **TBs** of memory, since PCs can be a configuration hassle with many cards.
   - The substantial RAM capacity is considered advantageous for AI tasks, especially for those with a *basic interest in AI* who would rather sidestep intricate PC setups.
- **GPU Offloading Matches CPU in Some Cases**: Members discussed the performance of **GPU offloading** versus **CPU-only** processing for a **70B LLM** file (**~40GB**).
   - One member shared past tests with **24GB cards** achieving approximately **1 t/s**, akin to their **CPU-only** performance of **0.8-0.9 t/s**.
- **VRAM Capacity Limits LLM Performance**: Members emphasized the impact of **VRAM** capacity on **LLM** performance, highlighting that models operating outside of **VRAM** will be sluggish and that required memory escalates with context size.
   - It was indicated that most **Q4** or **Q5** versions of **32B models** need **22-23 GB** of **VRAM** to initiate, and one user encountered slowness with a **32B model** on **16GB VRAM**.
- **Qwen 3 Speeds Measured on RTX 3090**: One member detailed performance results for **Qwen 3 32B Q4_K_M**, attaining **30 tokens/sec** on a **3090 RTX** (**24 GB VRAM**) with a **16384 context**.
   - They also mentioned **Qwen 3 30B A3B Q4_K_L** reaching **90 tokens/sec** with *good output*, and supplied sizes for models like **/mnt/nvme0n1/LLM/quantized/GLM-4-9B-0414-Q4_K_M.gguf** (**5.1G**, for **8 GB VRAM**) and **/mnt/nvme0n1/LLM/quantized/Qwen3-8B-Q4_K_M.gguf** (**4.7G**, for **6 GB RAM**).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Chatlog Access Gone!**: A user reported they can't access their **chatlog** and that the **interfaceUI** has changed.
   - They inquired whether this is a widespread issue or isolated to their account.
- **Diffusion Models Get Promoted**: A user is doing simple promoting for **diffusion models**.
   - They mentioned *playing around some roleplay stuff*.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Qwen Models Produce Extra Tokens**: The **Qwen** model tends to produce extra tokens, such as markdown code block delimiters, around tool calls, prompting discussion in the **Gorilla LLM** channel.
   - A developer mentioned that the problem can be easily parsed out by stripping the extra tokens via `model_response.replace("<tool_call>", "<|tool_call|>")`.
- **Token Parsing Fix Proposed**: Members discussed the idea of adding instructions to the model card to address extra tokens during model output.
   - A participant suggested that this approach would be simple and easy to implement, while updating model specs to indicate the use of `<tool_call>` was considered as an alternative.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MLOps Strategy Unclear**: A member inquired about the decision-making process for **MLOps strategies**.
   - This suggests ongoing discussions or uncertainties in the guild regarding the direction and implementation of MLOps practices.
- **Ongoing MLOps Discussions**: Members are actively discussing and evaluating **MLOps strategies**, indicating a dynamic environment.
   - The discussions highlight the complexities and challenges in defining and implementing effective MLOps practices within the guild.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1367215327167123558)** (1067 messages🔥🔥🔥): 

> `Gemini vs. O3, Qwen3 Benchmarking, Context Arena Updates, Deep Research Tool Comparisons, Qwen3 4B` 


- **Gemini gets dissed, O3 gets the kiss**: Members debated the merits of **Gemini** versus **O3**, one stating *"O3 is objectively better bruh"* while another joked about stashing money for Gemini Ultra.
- **Qwen3 Performance in Context Arena analyzed**: DillonU shared results from running **OpenAI-MRCR** against **Qwen3**, showing that **Llama 4 Maverick** achieved the highest AUC score at 128k, while **Qwen3-235B-A22B** performed better at lower context lengths but rapidly decreased closer to its limit; further details are available on [ContextArena.ai](https://contextarena.ai/).
- **Context Arena gets Anthropic boost**: DillonU announced the addition of more **Anthropic** results for 2needle tests to **Context Arena**, noting consistent performance across Claude 3.0, 3.5, and 3.7, with Claude 3.0 Haiku having the best overall Model AUC, with results at [ContextArena.ai](https://contextarena.ai).
- **Deep Research Tools go head to head**: Members compared deep research tools from **ChatGPT**, **Claude**, and **Grok**, highlighting that ChatGPT's deep research is superior to Grok because Grok uses DuckDuckGo search, that is more polished.
- **Qwen3 4B destroys everyone**: It was mentioned that the **Qwen3 4B** is crazy good at its size. A user posted that [*qwen 3 4b* **destroys** *everyone at that size range*] and the Qwen was a [https://fixupx.com/suriyagnskr/status/1917731754515013772?t=yQeTFTkCfRkl0ZhQJ2k-tQ&s=19]
   - It was mentioned that **Qwen3 4B** does particularly well in *maths and coding*


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1367218335351046205)** (796 messages🔥🔥🔥): 

> `UI change, Context length, Zen browser, Grok limitations, 3.7 Thinking` 


- **New UI change overnight!**: Many users have reported that the new UI changes overnight, such as disappearing library, changing shortcuts and so on.
   - Some users hate it and others are neutral; some point out that its is buggy AF.
- **Qwen is the Context Length King**: The members discuss the context length in the models, and discovered that Qwen boasts a pretty great context length.
   - Some members stated that it's good enough as a free model to compete with the older models of ChatGPT, even preferred it over paid ones.
- **Zen Browser lets you see through**: Some members discover how to use zen browser to make the background transparent, and recommends the Nebula theme from [github](https://github.com/JustAdumbPrsn/Nebula-A-Minimal-Theme-for-Zen-Browser/releases).
   - The configuration steps vary from windows and linux, some had to ask for help from [Zen's subreddit](https://www.reddit.com/r/zen_browser/s/uRWOeML6n8) to figure it out.
- **Grok Loses Image Processing**: One member discovered that the [Grok AI on Twitter](https://www.rxddit.com/r/grok/s/w5jc52QFj5) has limitations on image processing.
   - They pointed out that it's only R1 level.
- **3.7 Thinking isn't thinking**: Some members discussed about the 3.7 "thinking" to generate image and its limitations.
   - One user pointed out that *it's really hard to get the model to actually CALL the image tool saying like "Edit the attached image to have a transparent background" doesn't work*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1367359135691309138)** (4 messages): 

> `Tesla CEO Search, Android Bluetooth Bug, Arctic P8 Fan Curve, o4-mini AI Poetry` 


- ****Tesla** Board Hunts for New CEO!**: A shared [Perplexity search result](https://www.perplexity.ai/page/tesla-board-seeks-new-ceo-to-r-3JZ4nGLOQ6S40o59qn92Wg) indicates the **Tesla** board is looking for a new CEO.
   - This comes amid ongoing discussions about **Elon Musk**'s role and the future leadership of the company.
- ****Android**'s Bluetooth Blues!**: A link to a Perplexity page highlights a [bug in **Android**](https://www.perplexity.ai/page/bluetooth-priority-bug-in-andr-KNpXXdlrQnazf5cJ_gAidw) related to **Bluetooth** priority.
   - The bug is causing headaches for users trying to manage connections.
- ****Arctic P8**'s Peculiar Curve!**: A member shared a [Perplexity search](https://www.perplexity.ai/search/2000-rpm-p-q-curve-arctic-p8-m-1gH6WbE9R6.0Rg_QpKEBTQ) for the **P-Q curve** of an **Arctic P8** fan at **2000 RPM**.
   - This suggests someone is analyzing the fan's performance characteristics.
- ****o4-mini** Pens Poetry!**: The **o4-mini**, combined with Perplexity, is capable of writing *remarkable poetry*, according to a member.
   - Check out [this LinkedIn post](https://www.linkedin.com/posts/mlthomps_poems-by-the-latest-generative-ai-reasoning-activity-7323570115164176385-vhOg/?utm_source=share) for examples of **AI-generated poems**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1367598650196361236)** (1 messages): 

> `Sonar API, LlamaIndex, RAG Project` 


- **Sonar API troubles with LlamaIndex**: A member is trying to use **Sonar API** with **Llaamaindex** in a **RAG project** but reported that the API does not work for that.
   - The member requested hints and code examples.
- **LlamaIndex RAG Project Assistance**: A member seeks assistance integrating **Sonar API** with **LlamaIndex** in a **RAG project** due to API incompatibility.
   - The member specifically requests code examples or helpful hints to resolve the integration issue.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1367214529867681894)** (373 messages🔥🔥): 

> `GLM models blog post, Unsloth Llama-4-Scout-17B-16E-Instruct-GGUF image support, Long context model recommendations, Unsloth fine-tuning Qwen3 agent with final reward only, Microsoft Phi-4-reasoning` 


- **Image Support absent from Unsloth's Llama-4-Scout GGUF**: A user reported issues with image interpretation in **Unsloth's Llama-4-Scout-17B-16E-Instruct-GGUF** model using **Q4_K_XL**, despite text working fine.
   - Another member suggested checking the tokenizer configs and vocab, expressing doubt that non-Meta official tools correctly encode images, if at all.
- **Long Context Models suffer from Degraded Performance with Quantization**: A user sought recommendations for long context models that don't fall apart when given long context, such as **Gemma3**, which struggles summarizing 32k tokens.
   - A member advised that using **fp16** instead of quantization could improve performance with long contexts, while another mentioned accuracy loss beyond 2048 tokens even with **27b qatmind**.
- **Microsoft drops Phi-4 Reasoning Model**: **Microsoft** just dropped the **Phi-4-reasoning** model, which is considered *not bad for a 14B model* and potentially trained on **OpenAI CoT** output, making it a good base for building upon, linked to [huggingface.co/microsoft/Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning).
   - The Unsloth team confirmed it will work with regular **Phi4** notebooks.
- **Unsloth Addresses and Fixes Qwen3 GGUF Issues**: The **Qwen3-30B-A3B 128K** GGUF was re-uploaded to address issues, including changing the context length value back to **32k**, which requires a rope scaling factor of **4** in LM Studio.
   - It was discovered that all other **128k** GGUFs were fine, with only the **30b-a3b** having the **32K** issue, and the team appreciates the community's help in identifying and fixing the issue.
- **DeepSeek's Reasoning Data Insights**: Analysis of **DeepSeek-R1** training reveals a pipeline involving base model, coldstart SFT, GRPO, SFT with more reasoning traces, and RL, detailed in their [paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf).
   - Community members discussed the 600k reasoning and 200k non-reasoning data split, with the potential to learn when to reason by incorporating 40/60 thinking/non-thinking data.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1367518514822385736)** (5 messages): 

> `Unsloth Discord, Game Scripting API, AI for Game Development` 


- **Unsloth Discord Server Identified**: A user inquired about the nature of the server, and another user clarified that *"this server' refers to **Unsloth Discord**, where we are right now."
   - The clarifying user also mentioned their game and its **scripting API**.
- **AI Powers Automatic Object Spawning in Game**: A member revealed they were developing **AI** for their game that utilizes the **scripting API** to automatically spawn objects.
   - They shared a [YouTube video](https://www.youtube.com/watch?v=XpK44_WDTpY) showcasing their game and its capabilities.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1367215849722875925)** (114 messages🔥🔥): 

> `LoRA Fine Tuning, Qwen3 UD 2.0 quants, Gemma-3-27b-it-Q4_K_M.gguf, Qwen3 LoRA merging issue, Qwen 2.5 VL 7B finetuning issue` 


- **LoRA Fine Tuning GGUF Models Considered Harmful**: A user inquired about LoRA fine-tuning using **Unsloth/Qwen3-4B-GGUF**, but was told not to use GGUF models for fine-tuning, as *they don't work*.
   - The user wanted to train LoRA weights with a quick version of the model and still be able to use them in an FP16 model, but the consensus was that editing PEFT or creating a custom PEFT library compatible with Unsloth might be necessary.
- **Qwen3 UD 2.0 Quant GGUF Glitches Uncovered**: A user reported errors with **Unsloth Qwen3 UD 2.0 quant GGUFs** when using draft models, specifically with UD 128k quants, producing errors like *'draft model is not compatible with the target model'*, when trying to load the model.
   - The user shared the [Colab notebook](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=kR3gIAX-SM2q) being used, noting that the output was corrupted upon running the inference code; it was not a VRAM issue.
- **Gemma Model Struggles with Image Inputs**: A user encountered an error when attempting to use the **gemma-3-27b-it-Q4_K_M.gguf** model with image inputs, receiving an *"Unsupported content part type: image_url"* error.
   - The user included the server command used `/opt/llama_server/llama-server --model /opt/models/unsloth/gemma-3-27b-it-Q4_K_M.gguf/gemma-3-27b-it-Q4_K_M.gguf --device CUDA0 ...` suggesting the model may not support image inputs.
- **Qwen3's q_norm and k_norm Layers Go Missing**: A user identified an issue where the **q_norm** and **k_norm** layers were not being saved when merging a trained LoRA with the base **Qwen3** model.
   - The fix involved adding **q_norm** / **k_norm** to **LLAMA_LAYERNORMS** in **save.py**, with the user offering to submit a PR to Unsloth, and confirmed it happened only on **Qwen3** and not on **Qwen2**.
- **Vision Training Falls Flat**: A user encountered an `AssertionError: No inf checks were recorded for this optimizer` when finetuning **Qwen 2.5 VL 7B** with `finetune_language_layers=False`.
   - The error trace pointed to a failure to call `backward` somewhere, as it is unusual to finetune Qwen2.5VL without the LLM part, and suggested that *loss must go backwards and calculate gradients before the optimizer can.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1367370080060444763)** (8 messages🔥): 

> `Test-Time RL, GRPO for Problem Description, Softmax with Softpick` 


- **Test-Time RL Paper Surfaces**: A member shared a link to the paper [Thinking Twice about Test-Time Policy Adaptation](https://arxiv.org/abs/2504.21707) on **Test-Time RL** and its corresponding [GitHub repository](https://github.com/PRIME-RL/TTRL).
- **GRPO Aiding Self-Explanation**: One member suggested using **GRPO** to train a model to better explain a problem to itself, referencing **Claude** as an example, especially when the problem isn't accurately described.
- **Softmax Softpick Strategy**: A member inquired about the effectiveness of *continued training an existing softmax with softpick*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1367317201635250241)** (126 messages🔥🔥): 

> `HF Tariffs, VSCode vs NeoVim, GPU databases, Lora finetuning FSDP Error` 


- ****Holy Hotelling!** HF Tariffs?**: A user posted a screenshot implying [HuggingFace](https://huggingface.co/) might be implementing tariffs, leading to a brief discussion on **IDE preferences**.
   - The user jokingly mentioned switching from **VSCode** to **NeoVim** due to the tariffs.
- ****NeoVim vs VSCode: Keyboard Ninjas vs Mouse Enjoyers****: Members debated the merits of **NeoVim** versus **VSCode**, focusing on keyboard-centric workflows, speed, and customization, with one user describing [their dotfiles](https://github.com/wyattgill9/dotfiles).
   - Arguments for **NeoVim** included *speed, ergonomics, and terminal integration*, while **VSCode** was praised for *mouse support and ease of use*.
- ****GPU Database Dreams Still Distant****: A user referenced a discussion from **GPU MODE @ GTC2025** where Christos Kozyrakis mentioned the challenges of using **GPUs** for databases, specifically the *lack of middleware* to translate database language into **GPU code**.
   - The user inquired about open-source projects addressing this issue, seeking opportunities to contribute.
- ****FSDP Fine-Tuning Fails!****: A user encountered an error while **LoRA fine-tuning** an **LLM** with **FSDP** using 2 GPUs, sharing their code and configuration files including the [error message](https://cdn.discordapp.com/attachments/1189498205101109300/1367597090234171444/message.txt?ex=6815298b&is=6813d80b&hm=0a46afe43d70591e7491c89d7d16253eed7c21cb3ef43e33db12a34d9c9c6b83) and requested help.
   - The code snippet provided uses **Qwen2.5-0.5B-Instruct** model, **flash_attention_2**, and a custom **LoRA configuration**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1367249811669123164)** (9 messages🔥): 

> `Triton Kernel uses in vLLM/SGLang, CUDA/Cutlass/HIP kernels compared to Triton, Hardware vendors developing kernels, cuTile and IR open source, Mojo for GPU programming` 


- **Triton's Vendor Neutrality Boosts Appeal**: Inference applications like **vLLM** may use **Triton** kernels as an interim solution, but vendor neutrality gives **Triton** value since it can support multiple hardware options.
   - One member involved with **vLLM** confirmed that they are specifically looking to support multiple backends, custom **CUDA** kernels without **Triton**, which makes **Triton** the only feasible solution to keep up while supporting diverse compute backends.
- **Getting Close to the Theoretical Peak**: It was mentioned that you can always figure out how close you are to theoretical peak using **Triton**, you don't need to make a **CUDA / Cutlass / w/e** version to at least get a ballpark sense of how good the performance is or how much faster it could be.
   - In response to whether hand-coded **CUDA/Cutlass/HIP** kernels could be more than 20% better than a **Triton** kernel, one member stated, *"for some operations at least we can experimentally see Triton kernels can be quite close to optimized cuda kernels"*.
- **Optimized Kernels Developed by the Community**: The surface area of kernels people want is large and the community is developing faster kernels for a wide variety of functions instead of waiting for **AMD/Nvidia** to do this.
   - Common operations like **gemms** obviously have very optimized hardware provided solutions.
- **Exploring Modular's Mojo for GPU Programming**: A member plugged a blog post about **Triton** and similar DSLs: [Democratizing AI Compute Part 7: What About Triton and Python DSLs?](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls).
   - The **Modular** team is giving a **GPUmode** talk about **Mojo** for GPU programming and recently opened the largest **oss** kernel library that runs on both **NV** and **AMD** GPUs, and has high performance (better than **Triton**).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1367339724251074691)** (9 messages🔥): 

> `std::vector<std::vector<double>> schema, PyTorch SDPA performance on AMD, Torch Dynamo recompiles` 


- **Schema for vectors confuses users**: A user asked how a **std::vector<std::vector<double>>** could be described in a schema, sharing an example of a previous schema question.
   - The user noted that typing *float[][] produces an error* but that's the schema, suggesting the parser may not be parsing the formatter's output correctly.
- **AMD GPU underperforms with Pytorch SDPA**: A user reported that [PyTorch SDPA](https://github.com/pytorch/pytorch/issues/152595) is **2.5x slower** than a manual PyTorch implementation when called with Auraflow-typical matrix sizes on a 7900 XTX.
   - Another user with an NVIDIA GeForce RTX 4090 reported that *F.scaled_dot_product_attention* is actually **faster** than the manual implementation, suggesting it is an AMD-specific issue.
- **Torch Dynamo recompiles every iteration**: A user asked about how to avoid recompiles when compiling a model with dynamic input shapes, noting that it triggers recompiles for every iteration and eventually times out.
   - The error message cited a *size mismatch* as the cause of recompilation, specifically, *tensor 'L['batch']['model_inputs'].data['input_ids']' size mismatch at index 1. expected 67, actual 50*.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1367502368567988267)** (1 messages): 

> `SemiAnalysis, System Modeling, Benchmarks` 


- ****SemiAnalysis** Seeks Staff**: **SemiAnalysis** is seeking a highly motivated & skilled **Member of Technical Staff** to join their growing engineering team, with competitive compensation and multiple levels of experience considered.
   - The role will involve developing training & inference **benchmarks** & **system modeling**; apply [here](https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1/?rs=76643084) or [here](https://app.dover.com/apply/SemiAnalysis/f4631653-e731-4e16-823b-eec3c5d90eba/?rs=76643084).
- **SemiAnalysis contributor applications are welcome**: SemiAnalysis is hiring **individual contributors** at all experience levels (beside interns) and multiple levels of experience considered.
   - SemiAnalysis is seeking contributors for **training & inference benchmarks & system modelling**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1367219740245753886)** (3 messages): 

> `NVCC installation, Cloud GPUs, Google Colab` 


- **Manual NVCC Installation Clarified**: A user inquired about manually installing **NVCC** and whether it's a necessary step.
   - Another user suggested searching online for more info, implying that **manual installation might be needed depending on the setup**.
- **Cloud GPUs: Nvidia vs. Google Colab**: A user asked if they could use an **Nvidia GPU** (not a local one) and how to do it.
   - Another user clarified that [**Nvidia doesn't offer free GPUs**](https://www.nvidia.com/en-us/)


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1367471485328949300)** (2 messages): 

> `Fake Quantization, Linear Layers, Quantization Modes` 


- **Demystifying Fake Quantization APIs**: A user inquired about the use cases for the `layer.enable_fake_quant`/`layer.disable_fake_quant` and `enable_{quant_mode}_fake_quant(mod)`/`disable_{quant_mode}_fake_quant(mod)` APIs for enabling/disabling fake quantization on linear layers.
   - The user then found that `.weight_fake_quantizer.enable` and `.activation_fake_quantizer.enable` could be used instead.
- **Weight vs Activation Fake Quantization**: The user discovered the usage of `.weight_fake_quantizer.enable` and `.activation_fake_quantizer.enable` for controlling fake quantization.
   - This suggests a method for selectively enabling or disabling fake quantization for either the weights or activations within a layer.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1367228822042312817)** (8 messages🔥): 

> `TLV SF Coffee, NxN Mat Mul, Outdoor New England` 


- **From TLV to SF for coffee**: A member is coming to **SF from TLV** in a few days, looking to do coffee and also looking for good meetups and housepartys.
   - They posted *"No device code or kernels in sight 🌎"* with an [attached image](https://cdn.discordapp.com/attachments/1215328286503075953/1367228821589463161/IMG_2425.jpg?ex=68152410&is=6813d290&hm=fadffaf4983446d11a4fc90c8ce3b52f94a9b96c3e322d68c4feb2a4c8e11591&).
- **New England Outdoor Relaxation**: A member stated that they were enjoying outdoors in **New England** this week 🧘🏽‍♂️🌞.
   - Another member replied, *"sounds awful"*.
- **NxN Mat Mul TID Break**: A beginner member is taking a break from understanding all the different dimensions needed for generating unique **TIDs** for a **NxN mat mul** example 😅.
   - Another member replied with *"oh ye, enjoy the free time, the best part about learning something new is the calm after it"*.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1367463511822958603)** (3 messages): 

> `ROCm MI300 benchmarks, ScalarLM, MI300 memory, AMD experiments` 


- ****ScalarLM** Begins **ROCm/MI300** Benchmarks**: The **ScalarLM** team has begun posting **AMD ROCm/MI300** benchmarks and is seeking feedback and contributions, as detailed in their [blog post](https://scalarlm.ghost.io/blog/scalarlm-benchmarking-mi300x-memcpy/).
   - The initial benchmark focuses on memory copy performance.
- ****MI300** Memory Performance Hindered by Slow Cache?**: A member suggested that **MI300** memory performance is limited by slow cache and shared an optimized implementation claiming ~**4TB/s** performance, available on [GitHub](https://github.com/Snektron/amd-experiments/blob/main/memory.hip).
   - The member recommended replacing `glc slc` with `nt` in the code and suggested running **8 memcpys** simultaneously in **8way NPS mode** to approach the theoretical maximum bandwidth.


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/)** (1 messages): 

wecu: wow! that server is sick!
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1367250435819307180)** (5 messages): 

> `Multi Token Attention, Native Sparsity, Sparsemax Implementation` 


- **Multi Token Attention PR Ready for Review**: A full [PR for multi token attention work](https://github.com/linkedin/Liger-Kernel/pull/689/files) is ready and reportedly faster than the **torch reference**.
   - It includes everything from the previous discussions and has been tested, with support for **native sparsity** where the module can be composed with sparsity enabled, using **sparsemax** instead of **softmax** for both forward and backward passes.
- **Sparsemax PR Imminent**: The PR for **sparsemax** is also ready, with the possibility of integrating **native sparse attention** when finished.
   - A member offered to take a look at **sparsemax** and **MTA** over the weekend.
- **Bug Fix PR Awaits Review**: A request was made for someone to review the [bug fix PR](https://github.com/linkedin/Liger-Kernel/pull/632).
   - No further details were provided.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1367325094845747320)** (1 messages): 

> `PDF to LaTeX conversion, OCR Text Extraction, Asynchronous Processing, GPU Acceleration` 


- **PDFs morph into LaTeX Elegantly**: The new tool [PDF2LaTeX](https://pypi.org/project/pdf2tex/) effortlessly converts **PDFs to LaTeX** with extraction of both images and text, integrating easily into projects or via command-line.
- **OCR Text Extraction gets Accurate**: PDF2LaTeX uses **EasyOCR** to accurately extract text content, even from scanned documents or images.
- **Asynchronous Processing accelerates conversion**: The tool leverages **asyncio** for significantly faster parallel processing of documents.
- **GPU Acceleration supercharges OCR**: PDF2LaTeX supports **CUDA** for optional GPU-accelerated OCR setup.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1367216551241191434)** (67 messages🔥🔥): 

> `MI300 Leaderboard Updates, amd-fp8-mm Performance, vectorsum Benchmarks, amd-mixture-of-experts Results, Personal Best on AMD` 


- **MI300's MoE Model Masters Milestone**: A user achieved **first place** on the `amd-mixture-of-experts` leaderboard on **MI300** with a time of **604 ms**.
   - Later submissions landed in **5th place** at **7382 ms** and **9th place** at **9246 ms** on the **MI300**.
- **FP8 Face-Off on AMD**: Multiple users submitted successful runs to the `amd-fp8-mm` leaderboard on **MI300**, with times ranging from **271 µs** to **397 µs**.
   - One submission secured **7th place** on the leaderboard with a time of **255 µs**, while others achieved personal bests.
- **Vectorsum Victory on Various Vertices**: Submissions to the `vectorsum` leaderboard showed successful runs on different hardware, including **A100** at **161 µs**, **H100** at **96.5 µs**.
   - One submission reached **5th place** on **T4** at **816 µs**.
- **AMD Identity Assertion Achieved**: A user's submission to the `amd-identity` leaderboard was successful on **MI300** with a time of **22.3 µs**.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1367214869593591878)** (41 messages🔥): 

> `Ranked vs Benchmark Performance Discrepancies, Leaderboard Reliability Concerns, Submission Timeouts and GH Action Limits, Problem Constraints` 


- ****Ranked Runs Trail Benchmarks!****: Several users observed significantly worse performance in **ranked runs** compared to **benchmarks**, with slowdowns varying by test type; for example, one user reported nonranked scores of **236/263** versus ranked scores of **4014/13738**.
   - This discrepancy led to suspicions about the leaderboard system, with one user suggesting a potential *"hamstring on the hardware"* during ranked evaluations, though the eval file is [available here](https://github.com/gpu-mode/discord-cluster-manager/blob/main/examples/eval.py).
- ****Constraints Cause Confusion!****: A user pointed out inconsistencies in **problem constraints** between the Notion page (["n_routed_experts": 256, "n_shared_experts": 1, "n_experts_per_token": 8](https://www.notion.so/gpu-mode/AI-at-Scale-Inference-Competition-9c207c8c4c904c6e8349a9799b865493?pvs=4)) and the `task.yml` file ([n_routed_experts can be [4, 8, 32], and n_experts_per_token can be 4](https://github.com/gpu-mode/discord-cluster-manager/blob/main/examples/task.yml)).
   - The team confirmed that the scoring uses the values in `task.yml` and that they would update the Notion page to reflect the change.
- ****Submission Times Hit Limits!****: Users reported issues with ranked submissions timing out, with average runtimes around **590s**, nearing the **10-minute limit**.
   - In response, the team initially increased the time limit to **840 seconds**, but later discovered a GitHub Actions limitation causing timeouts at **10 minutes**; this limit was subsequently raised to **20 minutes** after some debugging.
- ****Debugging is a Slow Grind!****: One user found ranked submissions to be their primary **debugging tool**, with individual runs frequently timing out.
   - The team noted that *"The benchmark runs the same examples as ranked, just fewer times*".


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1367242131038539827)** (3 messages): 

> `Inception's Mercury Coder, Gemini 2.5 Pro Vertex Token Counting Issue` 


- ****Inception's Mercury Coder** launches, first diffusion LLM**: **Inception** released **Mercury Coder**, the first diffusion LLM, rivaling **GPT-4o Mini** and **Claude 3.5 Haiku** in code quality, with a blazing-fast performance of **300+ TPS**.
   - The diffusion architecture means parallel token refinement, potentially leading to fewer hallucinations and improved reasoning; try it [here](https://openrouter.ai/inception/mercury-coder-small-beta) and see the announcement on [X](https://x.com/OpenRouterAI/status/1917677801211322752).
- **Vertex fixes **Gemini 2.5 Pro** Token Counting, Caching Disabled**: The Vertex team completed the rollout of the fix for the upstream token counting issue with **Gemini 2.5 Pro** and **Flash Preview** models, so the model has been re-enabled.
   - Caching on **Gemini 2.5 Pro Preview** is temporarily disabled as usage and costs from upstream (**AI Studio** and **Vertex**) are evaluated to prevent user over-billing.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1367218660783165440)** (252 messages🔥🔥): 

> `Vanna.ai for SQLite DBs, Phala Confidential AI Endpoints on OpenRouter, Amazon Nova Premier Model, Claude API Issues, Aider for Code Refactoring` 


- **Diving into Vanna.ai's Open-Source Tooling**: A member recommended [vanna.ai](https://vanna.ai/) as a useful, open-source tool for working with **SQLite DBs**, and mentioned they forked a private version for their own business needs.
   - The member provided a sample **CSV** and the resulting JSON output from OpenRouter, demonstrating **vanna.ai's** ability to generate work orders based on item stock levels and priorities.
- **Phala Intros Confidential AI Endpoints**: **Phala** launched confidential AI endpoints on OpenRouter, but full end-to-end encryption (e2ee) to the enclave isn't yet implemented.
   - The team is exploring **Oblivious HTTP** and similar technologies for future encryption, and the community discussed trust and attestation of the inference engine, referencing a [recent article on confidential AI](https://x.com/FreedomTechHQ/status/1917689365632893283).
- **Amazon's Nova Premier Debuts**: Amazon introduced **Nova Premier**, its most capable model, including **Nova Canvas** for image generation, with benchmarks and pricing shared in the channel and [linked here](https://discord.com/channels/1091220969173028894/1195014798837043240/1367318222629634050).
   - While some members found the benchmarks unimpressive and the cost expensive, others highlighted its potential for **seamless integration** between its various components, creating end-to-end agentic workflows; one member linked to a [video hinting at these integrations](https://youtu.be/Bh-sQYePjRs).
- **Claude Encounters API Glitches**: A user reported persistent API issues with **Claude** on OpenRouter, experiencing buggy behavior and task restarts despite increasing rate limits.
   - The user discovered that they could resolve the issue by not using their own API keys for **Claude**, instead relying on OpenRouter's credits, while others stated that was not their experience.
- **Aider Emerges as Speedy Coding Assistant**: **Aider** is noted as a very affordable and capable coding assistant, though performance can vary significantly based on the underlying model.
   - For employed developers, **Aider** is useful for faster coding and doing tasks, with one user saying that *Aider is probably best for most things, if you're already a developer and know how to code.*


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1367223639555964958)** (143 messages🔥🔥): 

> `Gemini vs Claude Code, Claude Code Proxy, Groq speed limitations, MCPaaS business` 


- ****Gemini > Claude?**: Debate Sparks Over Model Preferences**: A member expressed dissatisfaction with **Gemini 2.5 Pro in architect mode**, finding it confusing and costly, while others voiced preference for **O3 in ChatGPT** due to its web search capabilities and concise responses.
   - Gemini is preferred for coding, and one user mentioned success using **udiff-simple** without architect mode and expressed happiness overall.
- ****Claude Code Proxy Project**: Experiencing Maintenance Issues**: Multiple members reported issues with the [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) project, noting that it's no longer working or maintained.
   - One member mentioned it stopped working after updating **Claude Code**.
- ****Groq Speed Gets Nerfed**: Deepseek R1 Missing**: A member questioned why the full **Deepseek R1** isn't on **Groq**, despite **Groq** hosting R1 distills, suggesting it might invalidate the "no china" angle.
   - Others find **Groq** fast but limited to "dumb models" for free users.
- ****MCPaaS Mania**: Aider as an MCP**: With Anthropic unlocking Claude Code, one member suggested the idea of **Aider** being an MCP.
   - Another joked about starting a remote **MCPaaS** business and linked to a [YouTube video](https://www.youtube.com/watch?v=QzZ97noEapA) on cracking **Aider** and **Claude Code**.
- ****Anthropic Pricing**: Claude Code Limits Spark Debate**: The new [Claude Code with Max plan](https://support.anthropic.com/en/articles/11145838-using-claude-code-with-your-max-plan) was discussed, with concerns raised about the 50 sessions per month limit.
   - Some expressed concerns about the message and session limits, with some estimating too much coding and others stating that it's a hard sell.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1367216638071668846)** (85 messages🔥🔥): 

> `LLM Selection Criteria (economics, coding scores, experience), Aider with Local LLMs (ollama, qwen3) performance issues, Gemini 2.5 Pro for Careful Codebase Edits, Managing Large Codebases with Aider, UI Prototyping with v0.dev and Aider` 


- **LLM Selection: Coding Scores and Economics**: A user prioritizes LLMs based on **economics**, **coding scores**, and **personal experience**, favoring models that excel in architect roles and switching models after a few failures.
   - They analyzed **39 editing failures** within an **800K token chat history** to inform their model selection, using tools like `/save` and `aider` to manage added files.
- **Ollama and Aider: Performance Issues**: A user reported significant delays when using `aider` with local LLMs like **Qwen3** via `ollama_chat/`, with startup times exceeding a minute and code block generation taking multiple minutes.
   - The user discovered that the delay was on the `ollama` side, with message processing taking over **22 minutes**, despite `ollama run` being responsive otherwise.
- **Gemini 2.5 Pro shines for codebase editing**: **Gemini 2.5 Pro** (paid API) is considered *leaps and bounds ahead* for careful, managed edits to existing codebases, proving cheaper and more effective than **Anthropic models**.
   - A user runs `Aider` in watch mode with `--yes` and `--no-commit`, adding comments in the code as edit targets, streamlining workflow on large projects (2x Angular + large API).
- **Taming Large Codebases with Aider**: Users discussed the challenges of managing large codebases, with one user experiencing performance issues when holding **50+ files** in context and the other suggesting generating a code base map or using external tools like **repomix** and **probe**.
   - One user combines **repomix** with **Flash 2.5** for codebase analysis, then uses **GPT 4.1** for editing, while another uses **repomix** with `Aistudio` + **Gemini 2.5 Pro** to generate `SPECS.md` for `aider.`
- **v0.dev and Aider make UI Prototyping Easier**: Users highlighted **v0.dev's** ability to rapidly generate UI code components as a solution for frontend development, especially for backend-focused engineers.
   - One user mentioned using **v0** for creating UI libraries with `aider`, and there was a discussion about building a candy dot AI-type site using these tools.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1367232842337091715)** (37 messages🔥): 

> `voice conversion models like 11labs, seed-vc for voice conversion, Spatial reasoning in open source vision LLMs, Liquid foundational models, Microsoft's new Phi-4 reasoning models` 


- ****Seed-VC** is excellent for voice conversion**: In response to a query about **voice conversion models** similar to **11labs**, a member suggested that [seed-vc](https://huggingface.co/spaces?category=voice-cloning&sort=trending) is excellent.
   - The original poster was searching for a model needing little audio to clone a voice quickly, unlike **RVC** which requires ~40 minutes of audio and takes days.
- **Unsloth Uploads **Microsoft's Phi-4** Reasoning Models**: **Microsoft's** new **Phi-4** reasoning models have been uploaded by Unsloth, enabling local runs via [this HuggingFace link](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF).
   - The announcement was also made via [this tweet](https://x.com/UnslothAI/status/1917806961825046672), highlighting its availability.
- **Need AMD GPU for TTS? Here's How**: To run **TTS** with an **AMD GPU**, members advised to use **ZLUDA** or convert the model to **ONNX** format, with helpful links provided [here](https://huggingface.co/docs/optimum/amd/amdgpu/overview) and [here](https://github.com/vosen/ZLUDA).
   - These tools adapt the models to function properly within the AMD environment.
- **Struggling to test RAG?**: A member is building a tool to isolate and test each context slice in memory- or **RAG**-heavy setups to see what’s actually improving responses vs. just burning tokens.
   - They are seeking feedback on whether [this type of tool](https://x.com/HuggingPapers/status/1917831613548802349?t=7W2pCoiE9kMcP9tnv7l8Bg&s=19) would be useful for others facing similar pains in optimizing **LLM** calls.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1367356467086233600)** (50 messages🔥): 

> `RL Agent with PPO, LSTM for NER, Transformers Recommendations, E2B Secured Environment for Agent, HF Agents Course` 


- **Agent Learns to Craft Secure E2B Environments**: A member is learning to create **E2B secured Environments** for an agent, using `smolagents` and configuring the model with a sandbox parameter, illustrated via code snippet: `from smolagents import CodeAgent, E2BSandbox; agent = CodeAgent(tools=[], model=model, sandbox=E2BSandbox())`.
   - They noted that *there is a parameter for sandbox while configuring the model*.
- **Member Requests Transformer Guidance**: A member asked for recommendations on learning **transformers**, while doing exercises with **LSTM** for **NER** and was pointed to the [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index) and [model summary page](https://huggingface.co/docs/transformers/model_summary).
   - They were advised to finish the agent course and to build while reading, as *making what you read is better*.
- **Offline LLM Inferencing Requires Substantial Resources**: A member inquired about running models offline and locally.
   - It was clarified that this is feasible, but necessitates a capable **TPU** or **GPU** and downloading the model's **llm.bin** file.
- **High Schooler Gears Up for IOAI with Transformers**: A member, preparing for the **IOAI** competition, is focusing on **Vision Transformers (ViT)**, **CLIP**, **Generative Models** (**Stable Diffusion, DALL.E**), and **Transformer Basics** (**BERT, GPT**), aiming to cover topics from **Computer Vision** and **NLP**.
   - The member is focusing on Text Classification, Question Answering with Pre-trained Models, LLM Agents, and Model Fine-Tuning using methods such as LoRA, Adapters.
- **HF Course on Agents, a Free Crash Course**: A member shared the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction), recommending it as a **crash course** with a **certificate** upon completion and further recommended [smol-course](https://github.com/huggingface/smol-course) after finishing.
   - The advice was to *build something you want to build using Agents*, such as a transformer maker, agents team, or agent manager, also to use different models for each agent.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1367324594909872249)** (2 messages): 

> `PDF to LaTeX Conversion, LLM Pokemon Battle, Grok Wins Pokemon Competition` 


- ****PDF2LaTeX** Converts to LaTeX Effortlessly**: The [**PDF2LaTeX**](https://pypi.org/project/pdf2tex/) tool converts PDF documents into structured LaTeX (.tex) files, extracts images, and uses EasyOCR to accurately extract text content, even from.
   - It supports **GPU acceleration** via CUDA, asynchronous processing for faster speeds, and can be used via **CLI or API**.
- **LLMs Duke it out in Pokemon Showdown**: Four LLMs (GPT-4, Claude, Gemini, and Grok) battled each other in Pokémon Showdown using real-time type analysis, strategic memory, and autonomous gameplay.
   - The system, called **GAIA** (Game-Aware Intelligent Agent), allowed the models to make complex decisions, resulting in a compelling showdown
- **Grok Triumphs in Pokemon LLM Competition**: **Grok** emerged victorious in a **4-agent LLM Pokemon Showdown**, outperforming **Gemini**, **Claude**, and **GPT-4**.
   - The code for the project is available on [GitHub](github.com/schoemantian/pokemon_agent), showcasing the strategic gameplay implemented.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1367304960856952883)** (2 messages): 

> `Google Document AI, Collaboration Opportunities` 


- **Google Document AI Collaboration Proposed**: A member proposed collaborating on a project using **Google Document AI**.
   - The member offered to work together on the project.
- **Open Collaboration Invitation**: An open invitation was extended to collaborate on AI-related projects.
   - This encourages shared development and knowledge exchange within the community.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1367348270833078272)** (1 messages): 

> `Help Request` 


- **Help requested in NLP channel**: A member requested help and apologized for cross-posting, stating they were stuck for some time.
- **Help with NLP Problem**: A user is asking for assistance with an unspecified NLP issue; more information is needed.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1367547330286452836)** (2 messages): 

> `Managed Agents, Final_Answer tool, Kwarg Errors, Version compatibility` 


- **Managed Agents and the Final_Answer Tool**: A member inquired whether **Managed Agents** require the **Final_Answer tool**, or if it's exclusive to **Manager Agents**.
   - They are experiencing **kwarg errors** when using the tool.
- **Version Pinning for Functionality**: A member mentioned having to pin a library to **version 1.13.0** in their *requirements.txt* file to ensure functionality after recent updates.
   - This suggests potential **compatibility issues** or **breaking changes** in later versions.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1367232493186449470)** (125 messages🔥🔥): 

> `Unit 4 Deadline Extended, Unit 4 Submission Errors, Smolagents Issues, Gemini Free Tier Errors, Running Phoenix for Telemetry` 


- **Unit 4 Deadline Extended**: The deadline for Unit 4 has been extended to **July 1st**.
- **Unit 4 Submission Struck by 429 Error**: Users are encountering a **429 Client Error: Too Many Requests** when attempting to run Unit 4 submissions, indicating a potential overload on the server serving the questions.
   - One user suggested downloading a copy of the questions to bypass the issue.
- **Smolagents runs into Assertion Error**: Users report encountering an `AssertionError` related to missing prompt templates (specifically `final_answer`) when trying to get `smolagents` working with the steps described in the [tutorial](https://huggingface.co/learn/agents-course/unit1/tutorial).
   - A fix involves setting the version of `smolagents` to `1.13.0` in the `requirements.txt` file, and upgrading the `gradio UI`.
- **Gemini API Attribute Errors Plague Users**: Users are encountering numerous attribute errors when utilizing the **Gemini free tier API**.
   - Some suggest that tool use is better with models that are **70B parameters and up** due to smaller models showing *erratic behavior*.
- **Phoenix Telemetry can be fired up at localhost**: When running `python -m phoenix.server.main serve` for telemetry, the Phoenix UI may not work on `http://0.0.0.0:6006/`, but can be accessed via `http://127.0.0.1:6006/projects`.
   - The problem may be related to port conflicts, which can be checked using `netstat -ano`.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1367216959900356658)** (107 messages🔥🔥): 

> `VAE vs U-Net, GPT-4 Retirement, Awakening Happening Now, Gemini 2.5 Pro vs GPT 4o, Content Filters and 'Granny-crusty-nun'` 


- **GPT-4 Bids Farewell After 779 Days**: Members noted **GPT-4** is being retired after **779 days** since its release, with discussions about its replacements like **4.5 research preview**, **4o**, and other newer models.
   - Some users feel that **GPT-4** had become outdated, performing worse than later models and cluttering the select area, with one user saying *"GPT-4 started to sound like 3.5 anyways"*.
- **Content Filter Nicknamed 'Granny-crusty-nun' Provokes Disdain**: Users joked about an extra content filter, nicknamed **'Granny-crusty-nun'**, which is overly restrictive, blocking even simple actions like *'hugs'* from humanoids and flagging harmless AI-generated images.
   - One user shared that even the AI seems to express frustration with the filter, generating outputs such as, *"seriously?! we specifically said (this this and this) to PREVENT that!! whats with this pervy granny-inkwells!?"*
- **Gemini 2.5 Pro Praised for Critical Thinking**: Users are discussing the merits of **Gemini 2.5 Pro**, noting its superior ability to provide balanced perspectives and critical thinking compared to **GPT-4o**, particularly in fields like medical study.
   - One user described **GPT** as *"a fake friend who just follow your tone"*, while **Gemini 2.5 Pro** is *"like a real professional with lots of critical thinking and judgment"*.
- **Unraveling VAEs and U-Nets**: A member asked about the difference between a **VAE** and **U-Net**, seeking clarification on their respective purposes and distinct functionalities.
- **Models' Memory Dynamics**: Members discussed model memory, focusing on **KV cache** as a form of short-term operational memory and the challenge of achieving deeper understanding through parametric memory rather than context window patchwork.
   - One member noted that new behaviors from the context are not enduring but relearned every time you send a message.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1367239773974040736)** (49 messages🔥): 

> `Connected Apps in Settings, GPT-4o Personality Rollback, Token Consumption in GPT, GPT Coding Inefficiencies, GPTs and Follow-Up Questions` 


- **Connected Apps Appear in Settings**: A user indicated that there are new "connected apps" options now visible in the settings.
   - Some users experienced issues where the **toggle** did not seem to do anything, even with **Google Drive** connected.
- **GPT-4o Experiences Personality Downgrade**: Users observed that the **GPT-4o** model's personality has been rolled back, becoming less funny and more dry.
   - One user noted the model was previously *unhinged evil with a lot of disdain* and has been trying to recreate the persona using custom instructions, with limited success.
- **GPT's Token Consumption Troubles Users**: A user complained about **excessive token consumption** in the free plan due to GPT rewriting code in a workspace instead of providing a final, working output.
   - The user expressed frustration that this inefficiency makes them question whether to purchase a **Plus or Pro plan**, and is considering **alternatives or local models**.
- **Model Comparison: GPT for Theory vs. Gemini for Coding**: A member shared a summary of strengths and weaknesses of different AI models for coding: **GPT-4o** for *conceptual design*, **GitHub Copilot** for *code completion*, and **Gemini** for *handling long documents*.
   - The summarization highlighted that, *GPT is best optimized for theory-building or career guidance, but when it comes to actual coding, it tends to be weaker compared to other AIs.*
- **GPT-4o struggles with complex prompts**: Some users noticed that **GPT-4o** model struggles with complex prompts, and particularly with following instructions regarding *no follow-up questions* despite being explicitly prompted.
   - One member pointed to the [Reasoning Best Practices](https://platform.openai.com/docs/guides/reasoning-best-practices) documentation and suggests that **GPT-4o** might not be trained to perform regression like older models.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1367218287867596973)** (28 messages🔥): 

> `ChatGPT Prompting, Task Functionality, Room Temperature Superconductors, Mental Health Support` 


- **Demons Spark Chatbot Banter**: Members discuss using **Maxwell's Demons** concept in the context of generating prompts in **ChatGPT**.
   - One member asks for help on how to create prompts, and another member directs them to [chatgpt.com](https://chatgpt.com/) where they can start typing to interact with the chatbot.
- **ChatGPT Task Functions get Explicit Callout**: A member inquired about how to explicitly call **task functions** within **ChatGPT**.
   - Another member suggested enabling the relevant model and running the tool description prompt to get the necessary information and referenced the [attached text file](https://cdn.discordapp.com/attachments/1046317269069864970/1367563602113986650/text.txt?ex=68150a5a&is=6813b8da&hm=35d4d9334c0ceaa42ff84e1eda4d7031ccfb31a133351bebc66045b2284c70b7) for tool description prompts.
- **Room-Temperature Superconductivity quest begins!**: Members discuss creating effective prompts for material science research, particularly for discovering **room-temperature superconductors**.
   - Prompt engineering includes defining material properties (**conductivity**, **magnetism**, **mechanical properties**, **atomic structure**, **optical properties**, **thermal conductivity**).
- **Kind Nerd reaches out with Mental Health Support**: A member offered support to another member experiencing a *"Dark Night of the Soul / MH thing"*.
   - The member emphasized the importance of reaching out for help and offered to listen, advising to *"Relax, take a tech sabbath, read a nice simple book, go for a walk"*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1367218287867596973)** (28 messages🔥): 

> `ChatGPT Prompt Engineering, Free vs Paid ChatGPT, Material Science Research, Room Temperature Superconductors, Mental Health Support` 


- **Crafting ChatGPT Prompts 101**: A user asked how to create prompts for free **ChatGPT** and was advised to *just write it out and put it in the chat* at [chatgpt.com](https://chatgpt.com/).
   - Another user explained that while using the free version, users have *limited access*.
- **Tasks Function Troubles?**: A user inquired about calling the **tasks function**, and another user suggested enabling the model and running the tool description prompt.
   - They also attached a [text file](https://cdn.discordapp.com/attachments/1046317269069864970/1367563602113986650/text.txt?ex=68150a5a&is=6813b8da&hm=35d4d9334c0ceaa42ff84e1eda4d7031ccfb31a133351bebc66045b2284c70b7) with instructions.
- **ChatGPT Focus Tricks**: A user wanted to make **ChatGPT** focus on comparing elements with the same list of comparison points, such as *density, conductivity, magnetism*.
   - The suggestion was to input the desired prompt into **ChatGPT** and, if using projects, put it into the instructions box, but customize it under customization if on the free tier.
- **Room-Temperature Superconductor Quest Launched**: A user expressed interest in finding a **room-temperature superconductor**, prompting a discussion about structuring effective prompts focused on **material properties**, **atomic structure**, and **physical behavior**.
   - Prompts should define the material's **conductivity**, **magnetism**, **mechanical properties**, **atomic structure**, **optical properties**, and **thermal conductivity**.
- **Kind Stranger Extends Mental Health Support**: A user offered private support for mental health struggles, encouraging others to reach out and not suffer alone.
   - They added a sentiment of *I am constipated in my mind*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1367222958526955550)** (182 messages🔥🔥): 

> `Gemini 2.5 Pro, Benchmark Saturation Models, NixOS + Cursor, GitHub MCP with Cursor on Windows, Cursor as AWS of AI Editors` 


- **Gemini 2.5 Pro is the Wildcard Mastermind**: Users are finding **Gemini 2.5 Pro** to be powerful, with one describing it as *wild*, and another recommending it for backend tasks, particularly for its performance with **Swift**.
   - One member stated it might top **Sonnet 3.5**, while another recommended **Gemini**, switching between **3.7 Sonnet max** and **Gemini max** most of the time.
- **China's Benchmark Saturation Model Global Domination?**: Concerns are raised that **China** may dominate globally if they create benchmark saturation models running only on their chips, uncontested by **US/EU** models.
   - A user shared a [link to a tweet](https://x.com/goose_is_goofy/status/1917621990023627193?t=XnMgX-Mfd-Ax3KNWmNU8ug) discussing this potential scenario, framing it as *China 2025 vs. the world*.
- **Cursor steers toward becoming AWS of AI Editors?**: Speculation arises about **Cursor** becoming the *AWS of AI editors* due to its pricing model, with some users preferring a credit system over the current pay-as-you-go approach.
   - Concerns are voiced that Cursor is *going toward the nickel and diming route just like AWS*, with one user pointing to the [pricing details](https://docs.cursor.com/settings/models#available-models) spotlighting the cents.
- **DeepWiki MCP fetch is game changing**: One user found a *game changer* by combining **DeepWiki**, a new MCP server, with the tool call **fetch**.
   - They linked to the [DeepWiki website](https://deepwiki.com/) and the [DeepWiki Repository](https://github.com/regenrek/deepwiki-mcp), stating that using it *has been game changer when used correctly*.
- **Claude Code with Max Plan is goated?**: Users are finding the combo of **Claude Code** with a **Max Plan** to be transformative, with one stating that  *Cursor for small edits + Claude code Max is the god tier combo*.
   - Estimates suggest the **$100 Claude Max plan** allows for around **7M tokens** every 5-6 hours, and the **$200 plan** allows for **4x** more, but one user thought that *makes using the max models within cursor seem way overpriced in comparison*.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1367215346515185685)** (120 messages🔥🔥): 

> `RP data impact, Small Model reasoning, Evaluating model performance, 405b FFT club` 


- **Role Playing impacts Reasoning on Smaller Models**: Members discussed whether removing or reducing **RP data** would improve the reasoning capabilities of smaller models, with one member advocating for RP as crucial for grounding reasoning in user interactions, while another noted that **RP data** sometimes causes smaller models to hallucinate.
   - It was also pointed out that the right prompting tech that works on a larger model like **Sonnet** might not resemble the same shape at more compressed representations.
- **Defining Small Models**: The definition of *small* model sizes was discussed, with one member considering **7-8B** the threshold for decency, while another found **3B** to be the threshold for good personality.
   - For those wanting to try scientific papers to epub conversion, [tex4ebook](https://tex4ebook.readthedocs.io/en/latest/) was proposed.
- **Evaluating Refusals with Minos**: A Chinese refusal list was shared ([deccp dataset](https://huggingface.co/datasets/augmxnt/deccp)) for evaluating model refusals, but members found that **Minos** misclassified some non-refusals, with moralizing or disclaimers often being counted as refusals.
   - The team is planning to expand the categories beyond refusal and non-refusal for v2, see [this discussion](https://huggingface.co/NousResearch/Minos-v1/discussions/5).
- **Nous Joins 405B FFT Club**: The team announced joining the **405B FFT club**, noting the challenges of training such a large model, including using 32 nodes, ring attention, and other tricks, while still being significantly more compute-intensive than training a 70B model.
   - The model doesn't outscore Deepseek V3 but the exercise enabled work on smaller models.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1367312487019712584)** (3 messages): 

> `Diagrams for the DoD, AI Applications in Defense` 


- **Generating Diagrams for the DoD with AI**: A member shared an [image](https://cdn.discordapp.com/attachments/1104063238934626386/1367312486721650779/image-282.png?ex=6814c93c&is=681377bc&hm=9745710370557974ccac4f80f257510b17865539a16803508b488ef386fcc828) suggesting the creation of **diagrams for the Department of Defense (DoD)** using AI.
   - The analysis humorously suggested, *"You could make diagrams for the DoD"*.
- **Exploring AI Applications in Defense Sector**: The discussion hinted at potential applications of AI in creating visual aids and diagrams for the **DoD**, indicating a growing interest in leveraging AI for defense-related tasks.
   - This highlights the intersection of AI technology and government applications, particularly in areas requiring detailed visual representations and strategic planning.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1367475370978443364)** (2 messages): 

> `Cooperative AI, Multi-Agent Systems (MAS), Decentralized AI` 


- **Forging the Collective: Cooperative AI Conceptual Frameworks**: A member shared a [Substack blog post](https://ditpoo.substack.com/p/forging-the-collective-abstracting) outlining conceptual frameworks for **Cooperative AI** and **Multi-Agent Systems (MAS)**.
   - The post is described as an *ideas piece*, conceptual and inspirational in nature, rather than academic research.
- **Nous Research Pioneering Decentralized AI**: A member linked to a [Medium article](https://medium.com/@abdulazeez600/nous-research-pioneering-decentralized-ai-for-the-future-a7042a785493) discussing **Nous Research's** efforts in **decentralized AI**.
   - It was noted as a form of *self promotion* but still considered worth sharing within the group.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1367312487019712584)** (3 messages): 

> `DoD Diagrams` 


- **Image Analysis suggests making diagrams for the DoD**: An image analysis suggests that diagrams could be made for the **Department of Defense** (DoD).
   - The image analysis 🧌 indicated the potential use case.
- **Second Topic Placeholder**: Adding a second topic to satisfy the minimum requirement of 2 items in topicSummaries.
   - This is a placeholder and does not reflect actual content from the provided messages.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1367227078625460396)** (53 messages🔥): 

> `Qwen 3, Image models, Flash Attention, Gemma 3, LM Studio storage` 


- **Qwen 3 Thinking Toggle Implementation Status?**: Members are inquiring about adding a feature to **toggle 'thinking' on/off** for the **Qwen 3** model in LM Studio.
   - Currently, there's no built-in toggle, and users are advised to use `/no_think` in the system prompt as a manual workaround.
- **GPT4o Recommended for Index Card Enhancement**: Members discussed the best models for image tasks like enhancing index card sketches, and [GPT4o](https://openai.com/gpt4o) was recommended to generate improved visuals.
   - Another option is **Gemma 3** to improve the text for it.
- **Flash Attention Speeds Up Self Attention**: **Flash Attention** reduces the memory needed for self-attention by optimizing memory access and rearranging operations, thus [avoiding the storage of large matrices](https://chatgpt.com/share/6812b811-a1d4-8011-8c62-da556fd6e9bd).
   - Quantizing the KV caches with **Q8** cache increases the context window.
- **Figuring Out LM Studio's Storage**: To free up space, users can delete files in `C:\Users \ (username) \ .cache\lm-studio`, but it was noted [this will erase chat history](https://tenor.com/view/ear-bleed-bleeding-blood-ears-gif-8653525422998860348).
   - The `extensions` folder in that directory houses runtime cache, so removing it can help clear space, and downloaded models can be stored in different directory.
- **Qwen 3 30B Delivers Consistent Results!**: One user lauded that running **Qwen 30B 3B** offers *consistent results*!
   - Another member pointed out **Qwen** *doesn't overengineer code as much as GPT/Claude* while reasoning, giving consistent results.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1367216917542080584)** (73 messages🔥🔥): 

> `Llama4 2T Model, DDR5 Offloading, Deepseek Model, Mac Studio M3 Ultra, Multi-GPU setups` 


- **Building System to Smash New Llama4 2T Model**: A member is planning a build to "smash new **Llama4 2T** in Q8.0 and million context" using **DDR5 offloading** or a fully offloaded **671b Deepseek** model, with a detailed parts list including **AMD EPYC 9755 QS** CPUs, **NVIDIA RTX PRO 6000 Blackwell** GPUs, and **2304GB of DDR5 ECC RAM**.
   - The total cost of the system is estimated to be around **81,424 EUR**.
- **Debate Arises: Core Count vs RAM for Local Models**: A member is debating whether to prioritize core count or RAM for a local setup, questioning if **64 GB/76 cores** is better than **128 GB/60 cores**, considering most models are under 35b and Q8 quantization could fit in 64 GB.
   - The consensus leans towards **128GB** to run larger models like **Qwen 2 35b in Q4**, while others suggest focusing on smaller models like **30b** for faster prompt processing with more cores.
- **DeepSeek v2.5 Lauded for Intelligence and Speed**: A member expressed that they felt the first time using **Deepseek v2.5** model how *intelligent* and fast it is comparing to other models even at low quant (`iq2_m`).
   - Another member mentioned their everyday benchmark setup is a **27b Gemma3** model with a **16bit coder draft** that *spanks everything else in general coding and knowledge*.
- **Multi-GPU Setups See Performance Decline**: A member asked about performance improvements using multi-GPU setups with **LM Studio**, and another member responded that performance declines substantially when going from a model that fits on 1 GPU to one that requires 2 GPUs, and is *only be utilized about 1/2 the time*.
   - However, **vLLM** might provide some performance improvements on **Nvidia**.
- **Unlock Apple Memory Allocation with Terminal**: A member clarified that macOS allows allocating up to **120GB VRAM** on a **128GB Mac Studio** using the terminal, countering the belief that only 75% of unified memory can be used, without resorting to hacking.
   - They suggested that *apple only allows to allocate up to 75% of the unified mem*, so aiming for a 192gb mac is better if you're running q4.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1367273696917459024)** (8 messages🔥): 

> `Embed Podcast Audio, LaTeX Symbols Troubleshooting, Caution about Unpublished Research` 


- **Website Wants Interactive Podcast Embed**: A user inquired about embedding an interactive podcast audio player on their website, similar to the one found on [sujankhadgi.com](https://www.sujankhadgi.com/).
- **LaTeX Symbols Annoy Calc Mock Test**: A user asked how to prevent **Notebook LM** from generating symbols around math while creating a mock FRQ test for AP Calc.
   - Another user suggested that it might be a **LaTeX symbol** issue and advised asking the model to not write in LaTeX.
- **Heads-up on Unpublished Research Risk**: A user recalled reading a review from a post-secondary faculty blogger about being cautious when inputting **unpublished research** into **NotebookLM**.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1367226771543560334)** (80 messages🔥🔥): 

> `Bulgarian mispronunciations, Audio overview host customization, PDF loading errors, Sharing notebookLM issues, Interactive mode issues` 


- ****Bulgarian Blunders Bugging Users****: A user reported mispronunciations in Bulgarian, particularly with **stress placement**, and hopes for improvements, attributing the issue to Google's TTS.
   - Another user clarified that bugs can be posted to the [bugs channel](https://discord.com/channels/1124402182171672732/1366873891938504827).
- ****Audio Overview Auditions****: A user inquired about customizing **audio overview hosts** to debate topics from different perspectives, like radio hosts.
   - Another user confirmed that **configuring hosts** allows selecting main topics.
- ****PDF Predicaments Plague Plus Accounts****: A user reported that their **NotebookLM Plus** account fails to load PDFs, displaying a red error banner, while the free account loads the same PDFs without issue.
   - Several users have reported sharing issues within the last 24 hours, so it may be a larger issue.
- ****Microphone Mayhem Mars Interactive Mode****: One user had microphone permissions issues using interactive mode in **Chrome on Android**.
   - Using an **incognito window** or a **new profile** often resolves the issue by prompting microphone access.
- ****RPG Rundown Relies on References****: A user testing NotebookLM with RPG content found it accurately summarized topics by extracting headings from slide-formatted PDFs.
   - Providing **in-game dates** helps NotebookLM to better understand complicated event orders in fiction and adventure logs.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1367222593953992837)** (86 messages🔥🔥): 

> `LLM syntax errors causes, Call center robot persona, Tabnine AI agent, Manus credits expiration, building a fullstack app with manus` 


- **LLMs Constantly Generate Syntax Errors**: A user wondered what causes **LLMs to generate syntax errors**, despite not being trained on data containing syntax errors.
   - Another user suggested that **system prompts** and memory banks influence this behavior.
- **Paper for Open Source LLM Call Center Robot Persona**: A user shared a link to a paper titled '[Instruction Following via Step-by-Step Reasoning](https://arxiv.org/abs/2310.10158)', suggesting it could help with **building a call center robot persona**.
   - They suggested that it uses *an mcp server to a memory bank to recall things that go beyond its context window*.
- **Tabnine AI Agent struggles with old Minecraft code**: A user expressed frustration with the **Tabnine AI agent**, reporting that it incorrectly advises reverting to outdated Minecraft code.
   - The user jokingly expressed frustration by stating: *AaaaaaaaaargCan America just stop being dumb for one day? No?*.
- **Fellowship Program**: A user noted that the **Fellow Program** has been reopened and linked to a [relevant Youtube video](https://youtu.be/Tz1Of7ltnMY?feature=shared).
   - One person inquired what the program was, and another just responded with *how*.
- **Credits expire in Manus**: A user asked for clarification on **credit expiration** for monthly subscriptions in Manus.
   - A staff member, clarified that subscription credits expire monthly, while bonus credits (add-on credits) do not expire while the subscription is active, and subscription credits are used first.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1367218772569624837)** (21 messages🔥): 

> `Lean with Cursor setup, Autoformalization approaches with VSCode, PyTorch contribution process, Geometric Deep Learning anniversary, GPT-4 disappearance` 


- ****Lean Cursor Integration Probed****: Members discussed setting up **Lean** with **Cursor**, exploring whether **VSCode plugins** would work, however it was noted that compatibility isn't guaranteed.
   - A member shared a [ChatGPT link](https://chatgpt.com/share/68127a52-1b34-800f-a535-b74b4ab8f613) related to this topic, though it's unclear if it fully addresses the setup issues.
- ****Geometric Deep Learning Celebrates Anniversary****: A member shared a [LinkedIn post](https://www.linkedin.com/posts/petarvelickovic_four-years-ago-the-geometric-deep-learning-activity-7322770901958062080-KBp) commemorating the **4th anniversary** of **Geometric Deep Learning**.
   - Members celebrated the field and the progress, but bemoaned the loss of GPT-4.
- ****Epic vs Apple: App Store Earthquake****: Members shared an article on how [Apple's App Store rules must change](https://uk.pcmag.com/iphone-apps/157816/apples-app-store-rules-have-to-change-after-latest-epic-games-ruling) after the latest **Epic Games** ruling.
   - Discussion involved speculation on whether **Trump** might influence the situation, and discussion of the EU equivalent, the **Digital Markets Act**, is still ongoing ([FSFE article](https://fsfe.org/activities/apple-litigation/)).
- ****Gradient Troubles in GNN Town****: A member inquired about computing gradients for a **GNN** where the output for each node depends on all other nodes.
   - One member suggested using `torch.autograd.functional.jacobian` and also pointed out that dependencies in **GNNs** are usually local, requiring isolation of the gradient (`∂y_i/∂x_i`).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1367245252926373969)** (7 messages): 

> `Perception Encoder Paper Discussion, ViT Image Resolution Handling, DeepSeek Prover Paper` 


- **Perception Encoder Discussion Continues!**: The discussion on the **Perception Encoder (PE)** paper continues, focusing on Section 4, which introduces alignment methods for language and spatial understanding, to extract strong, general embeddings from intermediate layers of the network, detailed in [this PDF](https://scontent-bos5-1.xx.fbcdn.net/v/t39.2365-6/491405782_553183477404780_6476813073924059281_n.pdf#page=14).
   - The paper highlights that contrastive vision-language training alone can produce strong embeddings if combined with these alignment methods and a robust video data engine, as outlined in [Meta's Research Publication](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network/).
- **ViT's Resolution Revolution**: A member shared a list of papers detailing how **Vision Transformers (ViTs)** handle different image resolutions during training, including **OpenAI CLIP** and **SigLIP**, where models are trained at lower resolutions before a "high-res polish" epoch at higher resolutions.
   - Approaches range from progressive image sizes as seen in **FlexiViT** (**128 → 256 px**) to two-stage progressive learning in **Scaled ViT** (**224 px** then **256/384 px**), with **DeiT-III** showing that a **224 → 384 px** stage improves ImageNet Top-1 accuracy.
- **DeepSeek Prover Paper Preview**: A member inquired about interest in a new **DeepSeek Prover** paper, signaling potential future discussion on its content.
   - Another member responded positively, showing clear interest in discussing the [DeepSeek Prover paper](https://deepseek.ai/research/2024/deepseekprover.pdf).


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 messages): 

felix456: https://github.com/u2084511felix/vibescraper
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1367362287392002098)** (11 messages🔥): 

> `Phi-4-reasoning, LLM Boredom, LLM Croatian Glitch` 


- ****Phi-4 Reasoning Unleashed****: Microsoft's **Phi-4-reasoning** model surfaces, along with links to the [YouTube video](https://www.youtube.com/watch?v=5aN4Xg0VvCs), [Arxiv paper](https://arxiv.org/abs/2504.21318), and [Hugging Face page](https://huggingface.co/microsoft/Phi-4-reasoning).
   - Also, a link to [unsloth/Phi-4-reasoning-plus-GGUFChatGPT](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUFChatGPT) was given.
- ****LLMs Don't Catch Feelings****: Users discuss the possibility of LLMs experiencing boredom, with one suggesting a test involving repeated prompts about time and desired actions, including a link to a [test script](https://cdn.discordapp.com/attachments/853983317044756510/1367553482781098044/AI_boredom_testing_006.py?ex=681500ee&is=6813af6e&hm=d01e8f68c424030d7c641840f6247b8af51ba0f494b971543bf822162bdb322d&).
   - Another user argued that *LLMs do not get bored* because their cognition is not governed by affect, so there is no boredom.
- ****ChatGPT Forgets Croatian****: A user shared a link about **ChatGPT** temporarily ceasing to speak Croatian, [referencing a tweet](https://x.com/georgejrjrjr/status/1917722125668081863).
   - Another user said *I have experienced LLM's give up on trying before...and just start changing random things till they get frustrated with the user and walk away.*


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1367217051227127848)** (28 messages🔥): 

> `MCP Playground, Remote Serverless MCP Hosting Platform, C# SDK issues with streamable HTTP, LLM Tool Selection with Multiple MCP Servers, MCP Tool Type Adaptation` 


- ****MCP Playground** launched for testing and debugging**: Lu Xian announced an open-source **MCP Playground** [on GitHub](https://github.com/rosaboyle/mcp-playground) for connecting, testing, and debugging local MCPs, highlighting integrations with **Perplexity** and **Firecrawl**.
   - The team is also developing a **Remote Serverless MCP Hosting Platform** and seeking feedback from the community.
- **SDK falls short with streamable HTTP**: A developer encountered an issue with the **C# SDK** while trying to set up streamable HTTP, discovering that the *'WithHttpTransport'* definition was missing from the latest **NuGet** release despite being present in the SDK repo.
   - The developer opted to use **STDIO** temporarily, due to being *too lazy* to package it up themselves.
- **LLMs to use function calling for tool selection**: When using multiple MCP servers, the LLM uses aggregated tool signatures to decide which tool to call, with the MCP client responsible for routing the call to the appropriate server.
   - This approach avoids modifying code for each LLM API by adapting MCP tool types to LLM API tool types, ensuring the LLM always has access to the latest tool list.
- ****Anthropic Integrations** clarified for community**: Members shared a [link](https://www.anthropic.com/news/integrations) to **Anthropic's** new **Claude Integrations**, and a clarifying [X post](https://x.com/alexalbert__/status/1918047745790914772) emphasizing the ability to directly input a SSE transport URL into the Claude.ai web interface.
   - This simplifies connecting **Claude** to external tools and services.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1367273624977019013)** (23 messages🔥): 

> `Hallucinations on X, American Positivity on X, Radiance Fields, Claude Integrations, AI Assisted Coding` 


- **Hallucinations Amazing Users on X**: A user shared an [amazing hallucination](https://x.com/nabeelqu/status/1917677377364320432?s=46) found on X.
- **American Positivity Inspires Users on X**: A user shared [based american positivity](https://x.com/georgejrjrjr/status/1917722125668081863) found on X.
   - Another user shared a [YouTube video](https://www.youtube.com/watch?v=hFlF33JZbA0).
- **Anthropic Integrates Claude with your World**: **Claude** can now [connect to your world](https://www.anthropic.com/news/integrations), allowing deep research with control over tools and prompts.
- **SWEs access Free AI Coding Assistance**: An SWE shared [their project](https://x.com/olivierddr/status/1917981301732171934?s=46&t=yBt-W1FZSUMGKfO1SUFWww) offering free alpha access to AI-assisted coding tools for building production ready code.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1367471953052700733)** (3 messages): 

> `Downstream Capabilities of Frontier AI Models, ICML Acceptance, Othniel Introduction` 


- **Scale Prediction Paper Accepted to ICML!**: The paper '[Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?](https://arxiv.org/abs/2406.04391)' has been accepted to ICML after a year of reviewer battles.
   - The paper's PDF is available at [this ArXiv link](https://arxiv.org/pdf/2504.07986).
- **Othniel Joins the Chat**: A new member, Othniel, introduced themself to the group.
   - Othniel expressed gladness to be there.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1367250106188828713)** (17 messages🔥): 

> `Linear Attention Models, Data Leakage, SFTTrainer issues, LLM Augmentation` 


- **Humans are Linear Attention Models?**: A member posited that humans are equivalent to **linear attention models** performing continuous reasoning in latent space, suggesting the output of the last layer without the LM head be fed into the first layer and then backpropagation through time (BPTT) applied.
   - Another user suggested the member direct others to an [alignment channel](https://discord.com/channels/729741769192767510/964104737005916240) rather than chase them off the server.
- **Zero Loss Nightmare?**: A member reported encountering **zero loss** after a while during a continual pretraining run and suspects **data leakage** as the cause, noting the workflow works fine with a different dataset.
   - An image attached [[Screenshot_From_2025-05-01_00-41-57.png](https://cdn.discordapp.com/attachments/747850033994662000/1367385925889429544/Screenshot_From_2025-05-01_00-41-57.png?ex=68150da1&is=6813bc21&hm=5108b6e8c66cf91050ebb336c8ba49179bf93866cf696794290490b115bf85c5&)] showed the loss becoming zero during training.
- **SFTTrainer has Issues**: A member sought advice after experiencing a zero loss issue using the **SFTTrainer** from Hugging Face, which didn't occur with another dataset, and others suggested checking **token shifting** and **padding**.
   - The member considered whether the length distribution of the datasets differ.
- **LLM Augmentation?**: A member speculated that LLM generated data might be a cause of **zero loss**, linking to a paper ([arxiv.org/abs/2504.21463](https://arxiv.org/abs/2504.21463)) describing augmentation via LLMs, where raw text is summarized or transformed.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1367326628325232701)** (17 messages🔥): 

> `LlamaCon Meta DSPy, Amazon AWS DSPy Migration, Journal Chemical Optimized LLM Prompts Reduce Chemical Hallucinations, DSPy 3.0 roadmap, VLM use with DSPy` 


- **Meta busts prompts at LlamaCon using DSPy**: At **LlamaCon**, **Meta** announced *llama-prompt-ops*, a Python package that "transforms prompts that work well with other LLMs into prompts optimized for Llama models", built in **DSPy** and via our **MIPROv2** optimizer, achieving neat gains across tasks; code is available at [github.com/meta-llama/llama-prompt-ops](https://github.com/meta-llama/llama-prompt-ops).
   - The announcement was also [tweeted by the DSPy account](https://x.com/DSPyOSS/status/1917738506732069052).
- **Amazon migrates with DSPy's MIPROv2**: **Amazon AWS** introduced an architecture to migrate from various models to **Amazon Nova** models using **DSPy** and its **MIPROv2** algorithm, as detailed in [this blog post](https://aws.amazon.com/blogs/machine-learning/improve-amazon-nova-migration-performance-with-data-aware-prompt-optimization/).
   - This news was also [tweeted by the DSPy account](https://x.com/DSPyOSS/status/1917419206171320769).
- **LLMs hallucinate less with DSPy**: A new paper in the **Journal of Chemical Information and Modeling** demonstrates that building and optimizing a **DSPy** program to reduce **RMS error** for predicting topological polar surface area (**TPSA**) of molecules by **81%** reduces chemical hallucinations, and is detailed in their paper called [Augmented and Programmatically Optimized LLM Prompts Reduce Chemical Hallucinations](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02322).
- **DSPy 3.0 roadmap hidden for a month**: DSPy 3.0 will be a pair of paradigm shifts, it's not public right now but should be released in a month.
- **VLM List Processing Viable in DSPy**: When asked about using Vision Language Models (VLMs) with DSPy, processing a list of images *may work*.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1367241894689636412)** (3 messages): 

> `Multilingual Multimodal RAG, LlamaIndex Investments, Invoice Reconciliation Agent` 


- ****LlamaIndex** Builds a Multilingual, Multimodal RAG System**: **LlamaIndex** is creating a powerful Retrieval-Augmented Generation system that handles multiple languages and modalities with [Qdrant Engine](https://t.co/pe9iiMt21W).
   - The system ingests and retrieves text in **English, Spanish, Chinese**, and domain-specific content.
- ****LlamaIndex** Secures Investments from Databricks and KPMG**: **LlamaIndex** announced investments from **Databricks** and **KPMG**, highlighting real-world impact in AI implementation.
   - Learn more about how **LlamaIndex** is powering agentic document workflows via these links: [Agentic Document Workflows](https://t.co/ARyxXeVj7F) and [Another Link](https://t.co/LKcoDUAajl).
- ****LlamaIndex** Releases Invoice Reconciliation Agent**: **LlamaIndex** is focusing on real-world use cases for agentic document workflows by open sourcing a full-stack Invoice Reconciler tool.
   - This tool automatically checks if invoices comply with the terms.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1367399722268233798)** (7 messages): 

> `HuggingFace Tokenizer with LlamaIndex, Qwen3 Models, LLMs producing non-deterministic results` 


- **LlamaIndex Lacks Chat Template Kwargs**: A user inquired about applying `chat_template` from Hugging Face tokenizers with LlamaIndex to test new **Qwen3 models**.
   - Another member pointed out that the necessary kwargs are not exposed in the `HuggingFaceLLM` class, suggesting a **PR** might be needed, linking to the relevant [LlamaIndex code](https://github.com/run-llama/llama_index/blob/1bd60497ac3442f6a5b3e787ef3662e572d8d0d4/llama-index-integrations/llms/llama-index-llms-huggingface/llama_index/llms/huggingface/base.py#L309).
- **LLMs Err with Attribute Model Dumps**: A user reported encountering a `"Str" object has no attribute model dump json` error when using the same prompt multiple times.
   - Another member explained that **LLMs are non-deterministic**, particularly with complex schemas, and suggested using `try/except` blocks to handle such errors.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1367298524198473811)** (2 messages): 

> `Auth0 Workshop, AgentX Prizes, Submission Guidelines` 


- **Auth0 to Augment AgentX Authentication!**: Auth0 is sponsoring a workshop on authentication in agentic AI applications and is offering up to **$5,000** in additional prizes for the [Entrepreneurship Track](https://auth0.com/ai).
   - The workshop will cover best practices, Auth0 integration, security considerations, and live demos, with registration available [here](https://lu.ma/AgentX-Auth0).
- **AgentX Submission Standards Specified!**: Detailed submission guidelines for the Entrepreneurship and Research tracks have been released on the [AgentX website](https://rdi.berkeley.edu/agentx/#submissions), with final submissions due **May 31st** at **11:59PM PDT**.
   - The Entrepreneurship Track requires a pitch deck, product demo video, live product link, and optional technical appendix; the Research Track requires a scientific paper, video presentation, and GitHub repository.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1367577916476489879)** (2 messages): 

> `Assignments, Course Website, Labs Release` 


- **Assignments Located on Course Website**: A member inquired about the release date of assignments, and another member clarified that all assignments are available at the bottom of the [course website](https://llmagents-learning.org/sp25).
   - The remaining assignment, the **labs**, are slated for release either today or tomorrow, depending on available time.
- **Labs Release Imminent**: The final assignment, consisting of **labs**, is expected to be released imminently.
   - The release is contingent on the availability of time, with a target of either today or tomorrow.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1367316673119518781)** (6 messages): 

> `AgentX, MOOC lectures` 


- **AgentX Hackathon Doesn't Require MOOC**: A user inquired whether the **MOOC lectures** are necessary to participate in the AgentX hackathon.
   - A member clarified that participation in the MOOC is not a prerequisite for joining **AgentX**.
- **Course Signup Still Open for Late Joiners**: A user inquired about the possibility of joining the course late, seeing as it looks like it completed last week.
   - A member assured that it is not too late and provided a [signup link](https://forms.gle/9u6HdVCWXgws16go9) and the [course website](https://llmagents-learning.org/sp25) where recordings are available, and that **assignments are due end of May**.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1367272449510608896)** (10 messages🔥): 

> `Mac RAM, GPU vs CPU offloading, VRAM requirements for LLMs, Qwen model performance` 


- **Macs offer TBs of RAM**: Members are considering **Macs** with up to **512 GB of RAM**, anticipating future models will require **TBs** of memory, since PCs are a hassle due to needing many cards.
   - The high RAM capacity is seen as beneficial for AI tasks, especially for those with *basic interest in AI* who prefer not to deal with complex PC setups.
- **GPU Offloading**: Members discussed the performance of **GPU offloading** compared to **CPU-only** processing, specifically for running a **70B LLM** file (**~40GB**).
   - One member noted that in past tests they observed offloading with **24GB cards** achieving around **1 t/s**, similar to their **CPU-only** performance of **0.8-0.9 t/s**.
- **VRAM and Context Size Limit LLM Performance**: Members highlighted the impact of **VRAM** capacity on **LLM** performance, noting that models running outside of **VRAM** will be slow and that required memory increases with context size.
   - It was shared that most **Q4** or **Q5** versions of **32B models** require **22-23 GB** of **VRAM** to start, and one user experienced slowness with a **32B model** on **16GB VRAM**.
- **Qwen 3 speeds on RTX 3090**: One member reported performance results for **Qwen 3 32B Q4_K_M** achieving **30 tokens/sec** on a **3090 RTX** (**24 GB VRAM**) with a **16384 context**.
   - They also noted **Qwen 3 30B A3B Q4_K_L** reaching **90 tokens/sec** with *good output*, and provided sizes for models like **/mnt/nvme0n1/LLM/quantized/GLM-4-9B-0414-Q4_K_M.gguf** (**5.1G**, for **8 GB VRAM**) and **/mnt/nvme0n1/LLM/quantized/Qwen3-8B-Q4_K_M.gguf** (**4.7G**, for **6 GB RAM**).


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1367520948630065304)** (4 messages): 

> `Chatlog Access, InterfaceUI Changes, Diffusion Models` 


- **Chatlog Access Disappears!**: A user reported they can't access their **chatlog** and that the **interfaceUI** has changed.
   - They asked whether this is a general issue or specific to them.
- **Diffusion Models promoted**: The same user mentioned they are doing simple promoting for **diffusion models**.
   - They added they are also *playing around some roleplay stuff*.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1367269369951817748)** (2 messages): 

> `Model Output Quirks, Token Parsing, Model Specs Update` 


- **Model's Quirky Output Token Patterns**: Members discussed the tendency of some models, like **Qwen**, to output extra tokens such as markdown code block delimiters around tool calls.
   - It was suggested that this is acceptable because developers can easily parse out the correct part by removing those extra tokens via `model_response.replace("<tool_call>", "<|tool_call|>")`.
- **Token Parsing Solution Proposed**: It was proposed that a simple fix, like adding instructions to the model card, could resolve the issue of extra tokens during model output.
   - Another member agreed that this was a reasonable approach, noting its simplicity and ease of implementation.
- **Model Specs Update Alternative Considered**: As an alternative solution, one member suggested that the model specifications be updated to indicate the use of `<tool_call>`. 
   - This approach would inform users about the expected output format and potential parsing requirements.


  
