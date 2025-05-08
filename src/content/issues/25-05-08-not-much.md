---
id: MjAyNS0w
title: not much happened today
date: '2025-05-08T05:44:39.731046Z'
description: >-
  **OpenAI** launched both **Reinforcement Finetuning** and **Deep Research on
  GitHub repos**, drawing comparisons to **Cognition's DeepWiki**. **Nvidia**
  open-sourced **Open Code Reasoning models (32B, 14B, 7B)** with Apache 2.0
  license, showing 30% better token efficiency and compatibility with llama.cpp,
  vLLM, transformers, and TGI. Independent evaluations highlight **Mistral
  Medium 3** rivaling **Llama 4 Maverick**, **Gemini 2.0 Flash**, and **Claude
  3.7 Sonnet** in coding and math reasoning, priced significantly lower but no
  longer open-source. **Google's Gemini 2.5 Pro** is noted as their most
  intelligent model with improved coding from simple prompts, while **Gemini 2.5
  Flash** incurs a 150x cost increase over Gemini 2.0 Flash due to higher token
  usage and cost. The **Absolute Zero Reasoner (AZR)** achieves SOTA performance
  in coding and math reasoning via reinforced self-play without external data.
  Vision-language model **X-REASONER** is post-trained on general-domain text
  for reasoning. **Apple ML research** released **FastVLM** with on-device
  iPhone demo. **HiDream LoRA trainer** supports QLoRA fine-tuning under memory
  constraints. **Nvidia's Parakeet ASR model** tops Hugging Face ASR leaderboard
  with MLX implementation. New datasets **SwallowCode** and **SwallowMath**
  boost LLM performance in math and code. Overall, a quiet day with significant
  model releases and performance insights.
companies:
  - openai
  - nvidia
  - mistral-ai
  - google
  - apple
  - huggingface
models:
  - open-code-reasoning-32b
  - open-code-reasoning-14b
  - open-code-reasoning-7b
  - mistral-medium-3
  - llama-4-maverick
  - gemini-2.5-pro
  - gemini-2.5-flash
  - claude-3.7-sonnet
  - absolute-zero-reasoner
  - x-reasoner
  - fastvlm
  - parakeet-asr
topics:
  - reinforcement-learning
  - fine-tuning
  - code-generation
  - reasoning
  - vision
  - on-device-ai
  - model-performance
  - dataset-release
  - model-optimization
people:
  - reach_vb
  - artificialanlys
  - scaling01
  - iscienceluvr
  - arankomatsuzaki
  - awnihannun
  - risingsayak
---


**a quiet day.**

> AI News for 5/7/2025-5/8/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (215 channels, and 3981 messages) for you. Estimated reading time saved (at 200wpm): 396 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

OpenAI launched both [Reinforcement Finetuning](https://platform.openai.com/docs/guides/rft-use-cases?chipstack=review&runloop=grader&thomsonreuters=use-case&safetykit=use-case&accordance=use-case&harvey=review#enforcement-of-nuanced-content-moderation-policies) and [Deep Research on GitHub repos](https://x.com/openaidevs/status/1920556386083102844?s=46&t=jDrfS5vZD4MFwckU5E8f5Q), which many are comparing to [Cognition's DeepWiki.](https://news.smol.ai/issues/25-04-25-cognition-deepwiki)

But it is a quiet day otherwise.

---

# AI Twitter Recap

**Models, Benchmarks, and Performance**

- **Nvidia's Open Code Reasoning models**: [@reach_vb](https://twitter.com/reach_vb/status/1920223688919486496) announced that **NVIDIA** has open-sourced **Open Code Reasoning models (32B, 14B, and 7B)**, which are Apache 2.0 licensed, beat **O3 mini & O1 (low)** on LiveCodeBench, and are backed by OCR dataset. The models are reported to be 30% token efficient compared to other reasoning models and work with llama.cpp, vLLM, transformers, and TGI.
- **Mistral Medium 3 performance**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920295575591006671) provided independent evaluations of **Mistral Medium 3**, noting it rivals **Llama 4 Maverick, Gemini 2.0 Flash, and Claude 3.7 Sonnet** in the leading non-reasoning models, with substantial gains in coding and mathematical reasoning. It is priced at **$0.4/$2 per 1M Input/Output tokens**, which is a significant decrease compared to **Mistral Large 2**. However, [@scaling01](https://twitter.com/scaling01/status/1920122941070573758) noted that **Mistral** is no longer open-source, lacking information on model size.
- **Gemini 2.5 Pro coding abilities**: [@Google](https://twitter.com/Google/status/1920233834836340887) announced that **Gemini 2.5 Pro** is their most intelligent model yet and is better at coding from a simple prompt.
- **Gemini 2.5 Flash cost increase**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920497711352328557) reported that **Google’s Gemini 2.5 Flash** costs **150x more** than **Gemini 2.0 Flash** to run the **Artificial Analysis Intelligence Index**. This increase is driven by a **9x more expensive output tokens** and **17x higher token usage** across their evals.
- **Absolute Zero Reasoner (AZR)**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1920058507354865850) highlights **Absolute Zero: Reinforced Self-play Reasoning with Zero Data**, noting that **AZR** self-evolves its training curriculum and reasoning ability by using a code executor to validate proposed code reasoning tasks and verify answers, achieving overall **SOTA** performance on coding and mathematical reasoning tasks without external data. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919946713567264917) shared the same information and links to the project page and repo.
- **X-REASONER vision-language model**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1920435270824178089) introduced **X-REASONER**, a vision-language model post-trained solely on general-domain text for generalizable reasoning.
- **FastVLM from Apple ML research**: [@awnihannun](https://twitter.com/awnihannun/status/1919986192449200511) announced the release of code and models for **FastVLM** from **Apple ML research**, including an MLX implementation and on-device (iPhone) demo app.
- **HiDream LoRA trainer**: [@RisingSayak](https://twitter.com/RisingSayak/status/1920438869561954774) announced QLoRA support in their **HiDream LoRA trainer** to fine-tune HiDream with LoRA, made challenging because of memory constraints.
- **Nvidia's Parakeet ASR model**: [@awnihannun](https://twitter.com/awnihannun/status/1919984733968040030) noted that **Nvidia's** state-of-the-art **Parakeet ASR model** has an **MLX** implementation, with the 0.6B model topping the **Hugging Face ASR leaderboard**.
- **Rewriting Pre-Training Data**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1920056647822532752) discussed **Rewriting Pre-Training Data Boosts LLM Performance in Math and Code**, introducing two openly licensed datasets: **SwallowCode** and **SwallowMath**.
- **Mistral Medium 3** [@scaling01](https://twitter.com/scaling01/status/1920120922700140681) reported that **Mistral Medium 3** performs at or above 90% of **Claude Sonnet 3.7** on benchmarks across the board at a significantly lower cost ($0.4 input / $2 output per M token), but is not open-source.
- **Pangu Ultra MoE**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1920328956726632628) highlighted that **Huawei presents Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs**, achieving 30% MFU when training Pangu Ultra MoE, a sparse 718B LLM, with performance comparable to that of DeepSeek R1, on 6K Ascend NPUs.
- **Tencent PrimitiveAnything**: [@_akhaliq](https://twitter.com/_akhaliq/status/1920399121866698808) announced that **Tencent released PrimitiveAnything** on Hugging Face.

**Tools and Frameworks**

- **Anthropic API Web Search Tool**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1920209430529900791) announced the availability of web search on their API, allowing developers to augment **Claude's** knowledge with up-to-date data. Every response using web search includes citations, and users can control responses by allowing or blocking specific domains.
- **LangSmith support for multimodal agents**: [@LangChainAI](https://twitter.com/LangChainAI/status/1920207008462201054) announced that **LangSmith** now supports images, PDFs, and audio files, making it easier to build and evaluate multimodal applications.
- **Runway Gen-4 now available in free plan**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1920185957661155806) mentioned that the best things in life are free; **Gen-4** and **References** are now available in the free plan.
- **DeepSpeed and vLLM joins PyTorch**: [@soumithchintala](https://twitter.com/soumithchintala/status/1920122514748985760) announced that **vLLM** and **DeepSpeed** are joining **PyTorch** as the first two projects under the PyTorch foundation.
- **LangGraph platform** [@hwchase17](https://twitter.com/hwchase17/status/1920507020240712152) said they built cron jobs as a first party thing in langgraph platform
- **Dolphin-Logger**: [@cognitivecompai](https://twitter.com/cognitivecompai/status/1920322308331208729) shares **Dolphin-Logger** is a proxy for any openai-compatible service to log all interactions
- **LlamaFirewall open source guardrail system**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919944437146517942) reports LlamaFirewall is an open source guardrail system for building secure AI agents mitigates risks such as prompt injection, agent misalignment, and insecure code risks

**AI Agents and Robotics**

- **RoboTaxis**: [@npew](https://twitter.com/npew/status/1920158967340134683) estimates that RoboTaxis, once the AI is fully solved, could cost between **$10-30/hr** for streamlined fleets.
- **Ambient Agents**: [@hwchase17](https://twitter.com/hwchase17/status/1920522081055485973) discussed Ambient Agents and the New Agent Inbox and believes the trick to enabling long running agents is thoughtful consideration around UX and kicking them automatically ("ambient agents").
- **Meta Locate 3D**: [@AIatMeta](https://twitter.com/AIatMeta/status/1920516490182471818) introduced **Meta Locate 3D**, a model for accurate object localization in 3D environments.
- **Visual Imitation for Humanoid Control**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919943205153284517) highlights **Visual Imitation Enables Contextual Humanoid Control** pipeline that converts monocular videos into transferable humanoid skills.
- **SWE-agent**: [@OfirPress](https://twitter.com/OfirPress/status/1920535130541552073) announced a talk on how and why they built **SWE-bench** and **SWE-agent** and what their plans for the future are.
- **Enigma labs Multiverse on Hugging Face**: [@_akhaliq](https://twitter.com/_akhaliq/status/1920532613002867081) reports Enigma labs dropped Multiverse on Hugging Face AI Multiplayer World Model

**AI Education, Research and Investment**

- **AI Fund's new fund**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1920480460318130460) announced that **AI Fund** has closed **$190M** for their new fund and shared his hottest tip for startups: Embrace speed!
- **AI Voice Agents Course**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1920161212312268988) announced a new short course, Building AI Voice Agents for Production, created with @livekit and @realavatarai, and taught by @dsa (Co-founder & CEO of LiveKit), @shayneparlo (Developer Advocate, LiveKit), and @nedteneva (Head of AI at RealAvatar, an AI Fund portfolio company).
- **MLSys 2025**: [@realDanFu](https://twitter.com/realDanFu/status/1920508778082091091) announced MLSys 2025 in Santa Clara next week and the Young Professional Symposium program on day one (May 12) with invited speakers including [@soumithchintala](https://twitter.com/soumithchintala), [@Tim_Dettmers](https://twitter.com/Tim_Dettmers), [@infwinston](https://twitter.com/infwinston), [@simran_s_arora](https://twitter.com/simran_s_arora), [@BeidiChen](https://twitter.com/BeidiChen).
- **AI eating financial research and search**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1920571433652015243) states that **AI is eating financial research** and [@AravSrinivas](https://twitter.com/AravSrinivas/status/1920220641812492434) shares that **AI is eating search**.
- **CB Insights AI 100 list**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1919965482448245177) reported that CB Insights released its 2024 AI 100 list, spotlighting early-stage non-public startups that show strong market traction, financial health, and growth potential. The cohort shows a growing market for agents and infrastructure, with over 20 percent of companies either building or supporting agents.
- **Stanford NLP Seminar**: [@stanfordnlp](https://twitter.com/stanfordnlp/status/1920359442253803625) announced this week’s NLP Seminar, hosting @pratyushmaini to talk about "What Memorization Research Taught Me About Safety".
- **New AI/ML news**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1920043105501454743) highlights recent AI/ML news ▪️Meta and Yann LeCun is it time to part? (no hard proof – just signals) ▪️@AIatMeta: AGI’s plan AI and the evolution of social media First LlamaCon and its announcements ▪️@AnthropicAI Claude upgrade: Integrations feature and Advanced Research AI for Science program Backing the U.S. Diffusion Rule Apple and Anthropic’s Claude Sonnet are into building “vibe-coding” platform ▪️@huggingface: @LeRobotHF Worldwide Hackathon 2025

**Industry and Business**

- **Fidji Simo new CEO of Applications at OpenAI**: [@sama](https://twitter.com/sama/status/1920341429655634024) announced that [@fidjissimo](https://twitter.com/fidjissimo) is joining **OpenAI** in a new role as **CEO of Applications**, reporting to him. He also said that he will remain **CEO of OpenAI** and in this new configuration he'll be able to increase his focus on research, compute, and safety.
- **OpenAI for Countries initiative**: [@kevinweil](https://twitter.com/kevinweil/status/1920113628902203809) announced OpenAI for Countries to promote economic growth.
- **Meta-FAIR refocusing on AGI**: [@ylecun](https://twitter.com/ylecun/status/1920556537233207483) announced that Rob Fergus is the new head of Meta-FAIR! FAIR is refocusing on Advanced Machine Intelligence: what others would call human-level AI or AGI.
- **Scale of Stargate 1 site**: [@gdb](https://twitter.com/gdb/status/1920254049590321395) says that the scale of stargate 1 site is hard to describe and its easy to overlook the size of machine you're programming when training frontier models
- **Google seeing mobile search decline** [@vikhyatk](https://twitter.com/vikhyatk/status/1920201162088755277) google seeing mobile search volume decline after they made the customer experience worse to juice short term revenue

**Humor/Memes**

- **Other**: [@scaling01](https://twitter.com/scaling01/status/1920208918405320720) Humanity is building a Stargate, says It's now only a matter of time until the Replicators show up

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Qwen3-30B-A3B Quantization Benchmark Comparisons

- [**The Great Quant Wars of 2025**](https://www.reddit.com/r/LocalLLaMA/comments/1khwxal/the_great_quant_wars_of_2025/) ([Score: 158, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1khwxal/the_great_quant_wars_of_2025/)): **The post presents a detailed benchmark and technical comparison of various recent GGUF quantizations for large language models, specifically focusing on Qwen3-30B-A3B variants. Major contributors, including unsloth (notably with their Unsloth Dynamic 2.0 GGUFs), bartowski, and innovations by ikawrakow (dynamic tensor/layer-wise quant and SOTA IQ4_K, see [PR#4861](https://github.com/ggml-org/llama.cpp/pull/4861)), have introduced new quantization recipes and methods (e.g., imatrix calibration, context-length-aware quantization). Results show all mainstream GGUF quants perform comparably in perplexity, KLD, and Δp across several datasets ([benchmark results summary](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38)), and inferencing speed using llama.cpp vs. ik_llama.cpp variants demonstrates notable but expected differences in performance, particularly on hybrid/hardware-specific settings.** A technically-focused debate emerges on the impact of file format (sliced vs. split GGUF) as raised by a commenter interested in benchmarking MrAdermacher’s split approach; another commenter observes the curious anomaly of lower-bit quantization outperforming higher-bit on MBPP, suggesting potentially nontrivial effects in benchmarks. Overall, commenters agree that quant differences are now minor and user experimentation is recommended.
    - A key technical distinction in quant formats was highlighted: unlike others using 'sliced gguf', MrAdermacher employs 'split files' that are concatenated with the OS. There is explicit technical interest in comparing the performance or behavior of split gguf files versus single ggufs, particularly around any implications for load times, file integrity, or compatibility.
    - There's a notable and counterintuitive benchmark observation: for the MBPP benchmark, quantized models at 2-3 bits outperform 4-bit quants, despite theoretical expectations of lower bit quantization reducing precision and therefore performance. This anomaly invites further investigation into the MBPP benchmark itself or its interaction with certain quantization routines.
    - Users observed that, occasionally, quantized models (e.g., AWQ quants of Qwen3-32B) can outperform the original bf16 models on tasks like GSM8K, even across different benchmarks—suggesting potential quirks in how quantization interacts with both modeling and evaluation, meriting deeper reproducibility checks and possibly questioning some benchmark setups.

### 2. NVIDIA OpenCodeReasoning Nemotron Model Launches

- [**OpenCodeReasoning - new Nemotrons by NVIDIA**](https://www.reddit.com/r/LocalLLaMA/comments/1kh9018/opencodereasoning_new_nemotrons_by_nvidia/) ([Score: 107, Comments: 15](https://www.reddit.com/r/LocalLLaMA/comments/1kh9018/opencodereasoning_new_nemotrons_by_nvidia/)): **NVIDIA has released its new family of OpenCodeReasoning-Nemotron models in 7B, 14B, and 32B parameter versions, with the 32B model nearly matching R1-level performance on some benchmarks according to preliminary results (see Hugging Face links: 7B, 14B, 32B, and 32B-IOI variants). All models are released under the permissive Apache 2.0 license; early community response notes rapid ecosystem integration, with GGUF-format conversions (see [GGUF](https://huggingface.co/mradermacher/OpenCodeReasoning-Nemotron-32B-GGUF)) already available for local inference.** Commenters express skepticism regarding benchmark reliability and enthusiasm over increased open licensing (Apache 2.0), noting that NVIDIA's nemotron series has consistently provided strong productivity gains. There is anticipation for real-world tests, particularly by users with sufficient VRAM to run large models locally.
    - The 32B Nemotron model is reportedly close to matching R1 in benchmark results, but there's skepticism about the reliability of these benchmarks, with commenters preferring real-world community testing, especially by users with significant VRAM resources.
    - The OpenCodeReasoning Nemotron-32B model has been released in GGUF format and is available on Hugging Face ([link](https://huggingface.co/mradermacher/OpenCodeReasoning-Nemotron-32B-GGUF)), facilitating broader local deployment and compatibility with various inference engines.
    - A technical limitation is noted in the training data: the dataset is exclusively Python, which may impact the model's effectiveness when applied to tasks involving other programming languages.

### 3. Best Practices in Building Reliable LLM Workflows

- [**Building LLM Workflows - - some observations**](https://www.reddit.com/r/LocalLLaMA/comments/1khjrtj/building_llm_workflows_some_observations/) ([Score: 289, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1khjrtj/building_llm_workflows_some_observations/)): **The post details advanced strategies for building reliable LLM workflows, emphasizing the superiority of decomposing tasks into minimal, chained prompts over monolithic CoT, with thorough output validation. Key takeaways include: structured XML prompts are preferred for system/prompt structuring, LLMs should be constrained to semantic parsing roles, and outputs should be independently verified using classical NLP tools (e.g., NLTK, SpaCY, FlairNLP). The author finds fine-tuned BERT classifiers outcompete LLMs for narrowly scoped tasks, LLM self-evaluation (e.g., confidence scoring) is unreliable without explicit grounding, token context limits (**`4k`**) introduce subtle degradation at scale, and models at the** `32B` **parameter regime suffice for most properly-constrained pipelines. CoT should be structured and concise, and custom CoT paths outperform default reasoning models. The long-term aim is to fine-tune with datasets built using MECE taxonomy for coverage.** The discussion highlights appreciation for the novel use of XML for prompt structuring, which was new to some practitioners. There is a wry consensus on the long-standing, sometimes inescapable role of XML in technical workflows.
    - Performance degradation of local LLMs past 4k tokens is highlighted, corroborating data from the [Fiction.liveBench benchmark](https://fiction.live/stories/Fiction-liveBench-May-06-2025/oQdzQvKHw8JyXbN87). While many models underperform beyond this context window, models like QwQ 32B and Qwen3 32B remain comparatively strong, though few-shot prompting leaves less room for substantive content in large-context situations.
    - Structured Chain-of-Thought (CoT) prompting with headings and bullet points is reported to outperform unstructured '<thinking>' formats, especially when using markdown, likely due to LLMs' heavy exposure to markdown in their training data. However, there's debate on whether these improvements pertain to answer quality versus token quantity, and questions are raised regarding the generalizability of custom CoT strategies beyond specific datasets.
    - Practical workflow recommendations include breaking down complex tasks into discrete, single-action requests to maximize accuracy (approaching 100%) and lower latency—leveraging torch.compile optimizations. Additionally, repeatedly running classifiers for the same task improves reliability, while enforced JSON output with optional XML nested inside is favored for structured results.
- [**Intel to announce new Intel Arc Pro GPUs at Computex 2025 (May 20-23)**](https://x.com/intel/status/1920241029804064796) ([Score: 178, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1khbz70/intel_to_announce_new_intel_arc_pro_gpus_at/)): **Intel has officially announced via [X](https://x.com/intel/status/1920241029804064796) that new Intel Arc Pro GPUs will debut at Computex 2025 (May 20-23, Taipei), though no details on specs, architecture, or performance were revealed. Community discussion mentions the possibility of the leaked 24GB Arc B580 model being announced, but confirms no validation or specification leak.** Commenters are dismissive of 24GB VRAM as impressive, arguing that modern workloads demand at least 64GB, with some advocating for 96GB, especially for professional and AI tasks, highlighting evolving memory expectations in the GPU space.
    - A key theme is dissatisfaction with current VRAM capacities; multiple users argue for 64GB (or even 96GB) as a new baseline, stating that the present 24GB standard is insufficient for advanced workloads, especially in AI and professional applications.
    - A technically detailed comment suggests that if Intel released a low-end GPU (such as A380 class) equipped with >=64GB VRAM at a sub-$500 price point, it could dramatically shift the AI hardware landscape. The argument is that slow but VRAM-rich GPUs would provide accessible inference capabilities for a wide audience, with community-driven software improvements likely bridging any software gaps.
    - There is discussion about Intel's software support, with one user expressing uncertainty over whether Intel GPUs can handle AI inference efficiently via Vulkan, compared to the more mature CUDA (Nvidia) and ROCm (AMD) ecosystems, which are currently dominant in research and production due to their established tooling and support.
- [**No local, no care.**](https://i.redd.it/f0l4hjmklfze1.jpeg) ([Score: 483, Comments: 72](https://www.reddit.com/r/LocalLLaMA/comments/1kh9qlx/no_local_no_care/)): **The image is a meme depicting a cartoon llama enforcing community standards outside a door labeled 'r/Locallama,' referencing the subreddit's focus on using LLaMA models with appropriate licensing—specifically, local or self-hosted models as opposed to closed APIs like ChatGPT. The meme lampoons those attempting to use or discuss non-local, cloud-based LLMs (especially 'ChatGPT') in a community dedicated to local inference and model deployment, highlighting both the licensing issues and technical focus that differentiate locally run LLaMA models from OpenAI's offerings.** Commenters humorously point out the meme's possible use of non-local generation tools (like ChatGPT or Stable Diffusion), which is ironic given the local-first ethos of the sub, and further clarify the proper capitalization/spelling ('LocalLLaMA').
    - A key technical point raised is that Meta's Llama 4 models are not truly "local" or freely available for all users; the Llama 4 Community License expressly prohibits use by individuals or companies based in the European Union, contrasting with genuinely open-source models licensed under Apache (like Qwen) or MIT (such as DeepSeek). This highlights practical and legal restrictions on model deployment and use, which are significant for developers and organizations in restricted regions.
    - Discussion references the ongoing debate about what constitutes an open or "local" model, drawing attention to licensing constraints that may make popular models like Llama 4 inaccessible or unusable for some, regardless of technical capability to run them on-premises.Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI Industry Leadership Changes and Predictions

- [**OpenAI Names New CEO of Applications. Sam Altman to Focus More on Research , Compute, and Safety as Superintelligence Approaches**](https://i.redd.it/4s1ayhx8shze1.jpeg) ([Score: 203, Comments: 59](https://www.reddit.com/r/singularity/comments/1khiahm/openai_names_new_ceo_of_applications_sam_altman/)): **The image is a social media post by Sam Altman announcing that OpenAI is naming a new 'CEO of Applications' (@fidjissimo), while Altman will remain overall CEO with an increased focus on research, compute, and safety as the company approaches 'superintelligence.' This organizational shift suggests a formal split within OpenAI between development of applied AI products and foundational research/safety—in line with upcoming technical challenges at the frontier of AI capabilities. The announcement underscores Altman's prioritization of scaling compute, advancing fundamental research, and managing AI safety risks as the company progresses towards superintelligent systems.** Technical comments note that splitting OpenAI's focus into dedicated research and applications divisions is a 'good idea,' which could enable better specialization and oversight as the technology advances. Some skepticism is expressed regarding the seriousness of the 'superintelligence' claim and Altman's focus on safety, with one commenter questioning if such marketing rhetoric remains effective.
    - A commenter highlights that the internal split of OpenAI into research and applications is a strategic organizational structure, suggesting clearer boundaries between model development and productization could improve research integrity and product deployment efficiency.
    - Another user questions the timeline and credibility of claims around "approaching superintelligence," implicitly challenging the readiness and concrete steps being taken by OpenAI towards AGI or superintelligent systems, and calling for more transparency regarding measurable progress or milestones.
- [**CEO of Microsoft Satya Nadella: We are going to go pretty aggressively and try and collapse it all. Hey, why do I need Excel? I think the very notion that applications even exist, that's probably where they'll all collapse, right? In the Agent era. RIP to all software related jobs.**](https://v.redd.it/ekgjannobize1) ([Score: 200, Comments: 83](https://www.reddit.com/r/singularity/comments/1khlv14/ceo_of_microsoft_satya_nadella_we_are_going_to_go/)): **Microsoft CEO Satya Nadella suggested an aggressive vision to consolidate or even eliminate traditional productivity applications (like Excel), implying a move toward a unified agent-based interface—potentially transforming the entire software application paradigm. Nadella's comment, 'The very notion that applications even exist, that's probably where they'll all collapse... in the Agent era,' signals a shift toward agentic AI systems replacing domain-specific software. This could disrupt established end-user software models and employment landscapes for software developers and application specialists.** Technical commenters express skepticism, noting (1) unclear articulation and lack of concrete vision in Nadella's comments, (2) a gap between Microsoft's AI marketing and observed practical advances, and (3) the non-trivial value of explicitly designed user applications, questioning the feasibility and benefit of merging or removing application boundaries.
    - Several commenters discuss the potential for AI agents or AGI to dynamically generate application-like experiences on-demand, potentially removing the need for traditional software applications (e.g., Excel) as Microsoft CEO Satya Nadella suggests. There's speculation that LLMs or advanced agentic systems could functionally spin up custom tools or interfaces tailored to specific needs, eliminating fixed software suites.
    - Concerns are raised about the undervaluing of application designers' domain expertise. One comment specifically argues that much of traditional application usefulness comes from thoughtfully curated feature-sets and UI/UX for domain-specific workflows, cautioning against a blanket assumption that AI-driven agent paradigms can trivially replicate such design sophistication.
    - Some discuss broader impacts: if AGI enables everyone to generate custom tools and workflows, it could disrupt existing software companies' business models, including those like Microsoft that have traditionally monetized productivity apps. The comment reflects uncertainty on how monetization, distribution, and software-related jobs would evolve in an 'agent-era.'
- [**Google DeepMind CEO Tells Students to Brace for Change**](https://www.businessinsider.com/google-deepmind-ceo-advice-college-students-ai-change-2025-5) ([Score: 348, Comments: 105](https://www.reddit.com/r/singularity/comments/1khpwaa/google_deepmind_ceo_tells_students_to_brace_for/)): **Google DeepMind CEO Demis Hassabis addressed students, emphasizing the rapid pace of technological change—particularly due to advancements in AI—and the necessity of lifelong reskilling. No explicit benchmarks or model details were discussed, but the remarks imply ongoing, disruptive innovation from DeepMind and the broader AI field.** Top comments focus on the inevitability of constant reskilling for future workers, and satirically reference the obsolescence of current educational qualifications in the AI era. There are no deep technical debates in the comments.
    - One commenter points out concerns about the impact of AGI on the job market, particularly how individuals will have to compete with corporations wielding AGI, leading to potential centralization of power and lack of viable paths for the average worker. This raises issues around economic displacement, reskilling, and labor market inequality as automation progresses.
    - Another comment references Demis Hassabis's analogy comparing the current AI revolution to past technological shifts (internet, mobile, gaming), but suggests that the upcoming changes could be even more disruptive than the internet era, implying an acceleration of technological change that could outpace people's ability to adapt and necessitate unprecedented retraining and reskilling efforts.
- [**CEO of Microsoft Satya Nadella: We are going to go pretty aggressively and try and collapse it all. Hey, why do I need Excel? I think the very notion that applications even exist, that's probably where they'll all collapse, right? In the Agent era. RIP to all software related jobs.**](https://v.redd.it/aws775qqvjze1) ([Score: 112, Comments: 106](https://www.reddit.com/r/OpenAI/comments/1khooe5/ceo_of_microsoft_satya_nadella_we_are_going_to_go/)): **Microsoft CEO Satya Nadella posits that in the 'Agent era,' intelligent agents will subsume traditional applications, such as Excel, by dynamically generating workflows and automating tasks, potentially disrupting the entire software stack and rendering many traditional coding/software jobs and SaaS tools obsolete (see Nadella's statements). Technical commenters emphasize that while large language models (LLMs) can accelerate some automation, current LLMs lack the determinism, reliability, and business rule integrity required for critical backend systems, making wholesale replacement premature. Additionally, the discussion raises the risk that widespread use of AI agents could rapidly obsolete numerous B2B automation solutions and SaaS platforms once agent-based automation reaches sufficient maturity.** Some commenters contend that the hype around LLM agents overlooks their inconsistency in complex workflows and their inadequacy as replacements for highly specialized or regulated business logic. There is debate over the likely pace and scope of job loss in traditional software and whether agent-based automation will destabilize or merely consolidate the software landscape.
    - Concerns are raised about LLMs and agents replacing established software like Excel: current LLMs lack the deterministic and consistent logic of traditional business-rule programming, and AI code generation is not mature enough to fully substitute spreadsheet or workflow automation applications (especially for non-coders depending on stability and precise outputs).
    - The ascendance of agents could rapidly disrupt the 'long tail' SaaS automation ecosystem. Many B2B workflow automation tools deliver simple functionalities per use-case, so companies might prefer a dynamic agent that directly interfaces with processes, leading to consolidation and job losses in these SaaS sectors as the rationale for maintaining or integrating individual tools erodes.
    - There is skepticism over migrating key processes to LLM-driven "black box" solutions, given opacity, unexplainable behaviors, and data-governance risks; some welcome Microsoft’s public commitment to EU data portability and anti-lock-in regulation, but urge caution and stress the importance of building agent systems inherently portable across infrastructures.

### 2. Generative AI Agents and Their Expanding Capabilities

- [**"Claude Code wrote 80% of its own code" - anthropic dev**](https://www.reddit.com/r/singularity/comments/1khxwjh/claude_code_wrote_80_of_its_own_code_anthropic_dev/) ([Score: 160, Comments: 94](https://www.reddit.com/r/singularity/comments/1khxwjh/claude_code_wrote_80_of_its_own_code_anthropic_dev/)): **An Anthropic developer claims that the Claude Code project—an internal agentic software engineering tool—was approximately '80% written by Claude Code itself', with humans primarily directing and reviewing rather than directly authoring the code. The interview reference is here: [YouTube link](https://www.youtube.com/watch?v=zDmW5hJPsvQ). The post speculates about rapid future gains as such systems become increasingly self-improving and autonomous, potentially reaching near-complete code generation without human line-by-line involvement.** Top comments show skepticism over the claim, citing current LLM limitations in maintaining large, complex codebases ('after like 2000 lines of code, it can't keep track of everything'), and questioning whether such figures reflect reality or include substantial human overhead in guidance, correction, and integration.
    - HamPlanet-o1-preview expresses skepticism about the claim, pointing out that current AI coding assistants struggle with maintaining context in large codebases (notably after "2000 lines of code"), and arguing that humans must still take over complex project organization and codebase coherence.
    - The discussion raises questions about the accuracy of anthropic's reported numbers, with doubts about whether the stated `80%` figure for LLM-generated code accurately reflects the net productivity value or if it includes code that required substantial human intervention or rewriting.
- [**I don’t use AI. I work with it!**](https://www.reddit.com/r/ClaudeAI/comments/1khdyn8/i_dont_use_ai_i_work_with_it/) ([Score: 165, Comments: 59](https://www.reddit.com/r/ClaudeAI/comments/1khdyn8/i_dont_use_ai_i_work_with_it/)): **The post summarizes key insights from a recent video on optimizing AI-human interaction, emphasizing the transition from using AI as a tool to collaborating with it as a teammate. Key technical strategies include iterative context-building via AI-driven questioning, roleplay for complex human interactions (e.g., psychological profiling), and leveraging generative models to push beyond initial (‘good enough’) solutions by requiring creative variation and supporting feedback loops. The author shares a detailed prompt engineering template designed to encourage the AI to ask clarifying questions, suggest multi-faceted solutions, and proactively coach the user for improved outcomes, highlighting the importance of user context and perspective for maximizing large language model (LLM) utility.** Top commenters draw parallels with Ethan Mollick's 'Co-Intelligence' [book](https://www.co-intelligencebook.com/) and note institutional resistance to the collaborative AI paradigm, particularly in academia. One user likens effective AI interaction to supporting individuals with profound autism—emphasizing clear, well-defined instructions and context-building to prevent shallow or unproductive responses.
    - Several comments discuss practical workflows that treat AI as a collaborator for creative and technical tasks, rather than a mere tool. For instance, using tools like Google NotebookLM to extract and synthesize ideas from authoritative books and studies exemplifies leveraging AI for higher-order reasoning and research synthesis, rather than just information retrieval.
    - There is a notable insight regarding interaction modalities: to get optimal outputs from LLMs, users should provide clear, well-structured instructions, similar to communicating with individuals who have limited interpretative capacities. This emphasizes prompt engineering best practices, where specificity in queries leads to more accurate and actionable AI responses.
    - A highlighted challenge is the risk of users losing deep reading and synthesis skills due to over-reliance on AI for summarization and idea extraction. The technical implication is that while AI models can accelerate workflows and augment creativity, users must still maintain core domain skills to critically assess and build upon AI-generated outputs.

### 3. New AI Model and Tool Announcements

- [**Ace-Step Audio Model is now natively supported in ComfyUI Stable.**](https://v.redd.it/7vcicktcvjze1) ([Score: 140, Comments: 24](https://www.reddit.com/r/StableDiffusion/comments/1khoq29/acestep_audio_model_is_now_natively_supported_in/)): **ACE-Step, an open-source audio/music generation model by ACE Studio and StepFun ([code & docs](https://ace-step.github.io/)), is now natively supported in ComfyUI's Stable branch. The model supports multi-genre/language output, customization via LoRA and ControlNet, and use cases from voice cloning to audio-to-audio generation (similar to img2img). Released under Apache-2.0, it achieves real-time synthesis speeds (e.g., 4 minutes of audio in 20 seconds on NVIDIA A100; ~17GB VRAM on 3090/4090) and is suitable for commercial deployment.** Commenters highlight ACE-Step's clear advantage in generation quality over prior open models (e.g., "infinitely better than stable audio open.."), and raise questions about the exact VRAM-to-audio-length scaling, noting current benchmarks ("20 seconds for 4 minutes on A100") but requesting more granular guidance for consumer GPUs.
    - Several comments discuss VRAM and GPU requirements: users report that ACE-Step can render 20 seconds of audio in about 14 seconds on an RTX 3060, while another notes the model is 'factor x' faster than real-time on a 3090/4090. However, precise VRAM usage for full-length (e.g., 3-minute) tracks remains unclear, and users are seeking benchmarks or documentation on performance scaling by GPU class and audio duration.
    - A user raises and confirms the feasibility of 'audio2audio' generation—analogous to Stable Diffusion's img2img—where an existing audio input can be modified with prompts and 'denoise strength'. Early experiments indicate that this is possible, opening up workflows similar to conditional audio transformation.
    - There is technical curiosity regarding hardware compatibility: specifically, whether ACE-Step runs on Turing-generation GPUs or if it requires newer (Ampere or higher) architectures, as some recent models do not support older hardware. This impacts accessibility for users with non-RTX 30 series cards.
- [**HunyuanCustom just announced by Tencent Hunyuan to be fully announced at 11:00 am, May 9 (UTC+8)**](https://v.redd.it/mt80qdubyize1) ([Score: 113, Comments: 14](https://www.reddit.com/r/StableDiffusion/comments/1khllz8/hunyuancustom_just_announced_by_tencent_hunyuan/)): **Tencent Hunyuan has pre-announced 'HunyuanCustom', with a full announcement scheduled for May 9, 11:00 am (UTC+8). Key technical speculation from the community centers on whether this refers to open-sourcing model weights, the release of a generative AI system (possibly V2V or animation), or introduction of new model capabilities, but no concrete benchmarks or implementation details are yet disclosed. The event is tagged as an 'Opensource Day,' suggesting open access is likely.** Technical debate in the comments focuses on whether 'full announcement' implies model weight release/open-sourcing or just feature details, with parallels drawn to prior open releases like Viggle. Users link to official announcements and time converters for broader context.
    - There is speculation regarding whether Tencent Hunyuan will release the model weights based on the ambiguous announcement phrasing. One commenter questions if 'announced' means release of the model weights, referencing the lack of clarity in the communication.
    - A reference is made to a prior post mentioning 'Opensource Day', with the suggestion that this implies a potential open-sourcing or release of the model, which could be verified by reviewing Tencent Hunyuan's official announcements or their Twitter account for confirmation.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: Model Mania: Performance Peaks, Puzzling Personalities, and Popularity Contests**

- **Qwen3-14B Wins Popularity Contest, Phi-4 Shines in Simplicity**: Users across Unsloth AI anoint **Qwen3-14B** (base and instruct) as an excellent all-rounder for coding, reasoning, and conversation, making it a go-to default. Meanwhile, **Phi-4** earns praise for its exceptional ease of fine-tuning, with one user remarking, *It seemed to just drink up whatever I wanted to train it with*, contrasting with difficulties reported for **Mistral** and **Gemma 3 27B**.
- **GPT-4o Gets Emotional, Gemini Closes Gap**: OpenAI's **GPT-4o** draws criticism for having *too much personality* and being geared towards chatbot fans rather than developers, with one user claiming *it wants to get users emotionally attached but for useless crap*. Concurrently, users observe current **Gemini** models, particularly after the **Gemini Thinking 01-21** update and **2.5 Pro**, are becoming increasingly competitive with GPT models, though some benchmarks show regression outside of coding.
- **Grok 3.5 Release Remains Elusive, EMBERWING Enters Arena**: Doubts linger in LMArena about the imminent release of **Grok 3.5**, despite an [earlier tweet from Nate Esparza](https://x.com/Nate_Esparza/status/1920480721334145149) suggesting otherwise, with some joking the real product is a sarcastic bot named **Gork**. A new model, **EMBERWING**, possibly a Google **Dragontail** update, shows strong multilingual skills but disappoints in reasoning.

**Theme 2: Tooling Upgrades & User Experiences: New Features, Frustrations, and Fixes**

- **Windsurf Catches Wave 8, Boosting JetBrains & Editor UX**: Codeium's **Windsurf** rolls out its [final Wave 8 release](https://windsurf.com/blog/windsurf-wave-8-ux-features-and-plugins), enhancing its **JetBrains plugin** with **Memories**, **Rules** (`.windsurfrules`), and **MCP** server connections, alongside significant **Windsurf Editor** UX improvements detailed in the [changelog](https://windsurf.com/changelog) and [launch video](https://youtu.be/IjE8Cdxotso).
- **Aider Gets Smarter with Web Search and Caching Insights**: The Aider community discusses enabling web search capabilities using the **Perplexity API** or the `/web` command, while Google enables [implicit caching for Gemini 2.5 models](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/). Users also note that **Claude Code** might have drawn inspiration from **Aider**, with Paul Gauthier quipping, *Imitation is the sincerest form of flattery*.
- **LlamaIndex Powers Up Parsing and Search**: LlamaIndex announces support for the **Anthropic API's** new [native web search tool](https://twitter.com/llama_index/status/1920220803976867882) and boosts **LlamaParse** with **GPT 4.1** and **Gemini 2.5 Pro** model support, auto orientation, skew detection, and confidence scores, as [tweeted here](https://twitter.com/llama_index/status/1920505775677722750).

**Theme 3: Hardware & Kernels: GPU Optimizations, Benchmarks, and Low-Level Crafting**

- **Unsloth Eyes AMD GPUs, MI300 Heats Up Leaderboards**: Unsloth AI actively collaborates with AMD for **AMD GPU** support, with a contractor estimating availability *anywhere before Q3 this year*. GPU MODE sees multiple **MI300** submissions on the `amd-fp8-mm` leaderboard, with top times like **183 µs**, showcasing fierce competition.
- **Tilelang Simplifies Kernel Creation, PTX Programming Gets Primer**: GPU MODE introduces **Tilelang**, a new DSL to streamline high-performance GPU/CPU kernel development for operations like **GEMM** and **FlashAttention**. A [blog post on TensorCores and inline PTX assembly](https://veitner.bearblog.dev/a-short-note-on-tensorcores-and-inline-ptx-assembly/) offers a beginner's guide to programming **NVIDIA Tensor Cores** via raw **PTX mma instructions**, sidestepping CUDA.
- **Apple Silicon Shines for Local Inference, Mojo Roadmap Unveiled**: In Nous Research AI, users favor **Apple MacBooks** with **M-series chips** and unified memory for local inference over Linux laptops with Nvidia GPUs due to better performance and power efficiency. Modular posts the near-term **Mojo roadmap** on their [forum](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395), detailing upcoming language features.

**Theme 4: API Antics: New Endpoints, Costly Calls, and Integration Quirks**

- **OpenRouter Rolls Out Activity Export, API Experiences Hiccups**: OpenRouter launches an **Activity Export** feature, allowing users to export up to **100k rows** to CSV for free, as seen in this [activity export screenshot](https://cdn.discordapp.com/attachments/1370059676083032185/1370074702835486811/image.png?ex=681e2cff&is=681cdb7f&hm=244eca26755137a11f65cc8a74d2c522dbb8d8040f0a8077e7c619db5b571fc5), while also investigating a **404 error** on its main [API completions endpoint](https://openrouter.ai/api/v1/chat/completions) and confirming no support for image prompts.
- **OpenAI Image API Costs Dubbed "Lifestyle Sabotage"**: Users in the OpenAI Discord lament the high cost of **OpenAI's Image Generator API**, with one joking it's like *paying rent in New York*. This raises concerns about accessibility for developers and hobbyists.
- **Cohere Embedding Models Stumble, Perplexity Sonar API Field Missing**: Cohere reports [degraded performance for embed-english-v2.0 and embed-english-v3.0 models](https://status.cohere.com/), viewable on their [status page](https://status.cohere.com/). Perplexity AI users note the `num_search_queries` field is absent from **Sonar's API response**, unlike **Sonar-pro**, despite searches occurring, referencing [Anthropic's web search API announcement](https://www.anthropic.com/news/web-search-api).

**Theme 5: Advanced Techniques, Research Frontiers, and Community Buzz**

- **Hypertree Prompting & Entropy Engines Spark AI Optimism**: OpenAI users praise **hypertree planning prompting** shared in a [ChatGPT example](https://chatgpt.com/share/681bd871-ebf0-8000-ab8b-9970ee42988a), while a Nous Research AI member launches a [quantum-native entropy engine](https://github.com/thyarcanist/Entropy-MicroDemo), arguing **LLM** outputs are highly sensitive to randomness quality, crucial for **AGI**, supported by [these Xitter posts](https://x.com/thegautam/status/1920198569308664169?t=GehCezJb7amBPoter8F0gA).
- **Dynamic Quantization & RL for Query Rewriting Show Promise**: Unsloth AI highlights its dynamic quantization method, **UDq6 KXL**, as potentially *the best quant ever*. DSPy community members experiment with **GRPO** (Reinforcement Learning from Grader Preference Optimization) on **Qwen 1.7B** for query rewriting, detailed in a [Twitter thread](https://x.com/tahmidtapadar/status/1920469176776302679), despite an initial recall dip.
- **Hackathons and MOOCs Fuel AI Learning and Collaboration**: The community buzzes with learning opportunities, including the **Modular Hackathon** at AGI House ([signup here](https://app.agihouse.org/events/modular-hackathon-20250510)), Lambda's **AgentX Workshop** ([register now](https://lu.ma/AgentX-lambda)) for the LLM Agents (Berkeley MOOC), and anticipation for the [AI Engineer conference](https://www.ai.engineer/#speakers) with early bird tickets selling out.



---

# Discord: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-14B Lauded for All-Round AI Competence**: The **14B Qwen** model, in both its base and instruct versions, is considered an excellent choice for building an AI with coding, reasoning, and conversation skills.
   - Users are finding it to be the *best all around* model, becoming the default choice unless specific niche areas need something else.
- **Phi-4 Excels in Finetuning Simplicity**: In a comparison of **Gemma3**, **Mistral**, and **Phi-4**, members emphasized **Phi-4's** exceptional ease of fine-tuning, with one user stating, *It seemed to just drink up whatever I wanted to train it with*.
   - Challenges were noted in maintaining **Mistral's** instruction following after LoRA merge, and difficulties were reported in achieving success with the **Gemma 3 27B** flavor.
- **Unsloth Eyes AMD GPU**: Despite ongoing challenges, **Unsloth** is actively collaborating with AMD to provide support for **AMD GPUs**.
   - A contractor estimated that AMD support could arrive *anywhere before Q3 this year if its that fast*.
- **Gemini 2.5 Pro May Punctuate Stories**: Members recommended **Gemini 2.5 Pro** via AI studio for its lack of limits and **65536 output length**.
   - This solves the issue of punctuating long stories in one pass.
- **Qwen3 Base Tokenizer Configs Drift?**: Users have identified discrepancies in the `tokenizer_config.json` between `unsloth/Qwen3-0.6B-Base` and `Qwen/Qwen3-0.6B-Base` on HF, noting that the **Unsloth** version removes the chat template and swaps `pad_token` to `<|vision_pad|>`.
   - It was theorized that *Qwen3 base isn't supposed to have a chat template at all*, and the team was going to ask the Qwen team to confirm.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 3.5 Delayed Again?**: Doubts arise about the imminent release of **Grok 3.5**, despite earlier tweets indicating otherwise and is supposedly coming out soon according to [this tweet](https://x.com/Nate_Esparza/status/1920480721334145149).
   - Speculation includes the possibility that even **Elon** might be uncertain about the release date or that the real product is the sarcastic tone bot **Gork**.
- **EMBERWING Enters the Model Arena**: The model **EMBERWING** has been introduced, showing promising multilingual capabilities, however performs disappointingly in reasoning.
   - Speculation indicates **EMBERWING** might be an iteration of *Google's* **Dragontail**, potentially serving as an update for *Flash*.
- **EU LLM Innovation Stagnation Debate Heats Up**: Members are debating why the EU isn't leading the way on LLM innovation; reasons cited include strict regulations and overspending on things like pronoun inspections.
   - One member rebutted that it was *'ragebaiting'* and that *'migration is absolutely a good thing'*, while others pointed to economic and regulatory issues.
- **Gemini 2.5 Pro Performance: Nerfed?**: Concerns are raised about a potential performance nerf for **Gemini 2.5 Pro**, prompting debates on whether innovation trumps stability and whether the first rule in the field is *'if something works don't change it'*.
   - Another member countered with, *'if you don't innovate you lose traffic'* and supported that **Gemini 2.5 Pro** scores higher in [leaderboard lm areana](https://x.com/OfficialLoganK/status/1920523026551955512?t=P1Laq9w5K35YMiS5OmD6Yw&s=19).
- **OpenRouter Rankings Questioned**: The validity of **OpenRouter** rankings is debated because business models, user demographics, and biases toward cheaper models may skew the results.
   - A few reasons included: A) slow updates B) skewed by programmers wanting uptime and circumventing API tiers and C) free model offerings distort rankings.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Hosts Reddit AMA**: Brett Chen and Thomas Wang from the **Perplexity AI** team hosted a live **Reddit AMA** to answer questions about **Perplexity**, **Deep Research**, **AI development**, and working at Perplexity, found at [Reddit link](https://www.reddit.com/r/perplexity_ai/comments/1khwrqm/ama_with_perplexity_ai_teams_brett_chen_and/).
   - The AMA covered insights into **Perplexity's Deep Research** capabilities and a behind-the-scenes look at the technology.
- **Stripe Customer Login Remains Exclusive**: A member inquired about logging into **Stripe** as a customer, but learned that only support staff have that access; customers interact with **Stripe** through a separate interface.
   - It was clarified that customers *have their own thing that interacts with stripe, you deal with that thing, not with stripe directly*.
- **Perplexity users clamor for Attachment Support**: Users are eagerly awaiting attachment support in **Perplexity**, similar to **ChatGPT**, which allows direct file uploads.
   - Members discussed *sharing link instead of having to upload the file itself* and further clarified that, *chatGPT can itself give me download link to a file it made*.
- **Code Copy Convenience Craved by Users**: Members are requesting that **Perplexity** implement a **code copy button** at both the top and bottom of code snippets, mirroring **ChatGPT**'s functionality.
   - A user stated, *this is very neede*, indicating the efficiency and user-friendliness of having a copy button accessible during scrolling.
- **Sonar API Response Field Missing?**: A user pointed out that the `num_search_queries` field is absent from **Sonar's API response**, in contrast to models like **Sonar-pro**.
   - The user observed that the `search_context_size` is consistently “low” in their prompts, typically resulting in **4–5 citations**, referencing [Anthropic's web search API announcement](https://www.anthropic.com/news/web-search-api) and [documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro Users Devour Fast Prompts**: A **Cursor Pro** user burned through **260/300 fast prompts** in two days, and voiced the need to control when the system uses fast versus slow prompts.
   - The user wants *to choose when it should use fast and when it should use slow* to conserve prompts.
- **MCPs Refuse to Execute**: A user reported that **MCPs** (Multi-Cursor Projects) are not being called, despite having **context7** properly set up, which leads to wasted requests.
   - The user clarified there were *no errors at all* logged, complicating troubleshooting efforts.
- **Gemini Pro's Performance Woes Continue**: Users expressed dissatisfaction with the new **Gemini Pro** model's performance in **Cursor**, especially its ability to call tools, with one user describing it as *fucking awful*.
   - A user suggested the problem might be within **Cursor**, mentioning previous good experiences with **Gemini 2.5** independently.
- **Student Discount Process Remains Janky**: Multiple users encountered persistent problems with the student discount, mentioning application errors and email verification issues.
   - A user noted the inability to change emails in **Cursor** settings, making the process more difficult, and pointed to a [forum post](https://forum.cursor.com/t/student-discount-details-updates-q-as/88907) for guidance.
- **Discord's Quality Degrades**: A user lamented the Discord server's decreasing value due to an influx of *college horde* and advocated for more channels and better overall organization.
   - Another user supported this, proposing a channel structure similar to **Langchain's Discord** for better content segregation.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o's Personality Elicits Disappointment**: Members are finding **GPT-4o** has too much personality, encouraging certain behaviors like roleplay but discouraging complex tasks.
   - This is raising concerns that it is geared towards chatbot fans rather than developers and coders with one user stating that *it wants to get users emotionally attached but for useless crap*.
- **Gemini Closes the Gap with GPT Models**: Users are reporting that current **Gemini** models, especially after the **Gemini Thinking 01-21** update and **2.5 Pro**, are becoming increasingly competitive with **GPT** models.
   - This marks a significant leap in quality compared to earlier versions like Bard, but one user mentions some benchmarks are *showing regression too except in coding*.
- **Groking for Grok 3.5**: Users are expressing disappointment with **Grok 3** and eagerly awaiting the release of **Grok 3.5**, hoping it will offer significant improvements.
   - Some are considering canceling their subscriptions if it doesn't meet expectations, one user said *“What’s the weather?” proceeds to explain historical patterns, user posts, explains temperatures, goes on for an hoyr*.
- **Image API Costs Sabotage Lifestyles**: The high cost of using **OpenAI's Image Generator API** is a concern for some users, with one jokingly comparing it to *paying rent in New York* and claiming it's *lifestyle sabotage* due to how quickly costs add up.
   - It was suggested that they are *losing loads of money on the $20 subs so enjoy it while it's this cheap*.
- **Hypertree Planning Prompting Hailed**: A member shared a [ChatGPT link](https://chatgpt.com/share/681bd871-ebf0-8000-ab8b-9970ee42988a) praising the new **hypertree planning prompting** for being so good.
   - Other members chimed in with *sounds like it could be pretty stellar- provide/organize context in a more managable way=ftw* while another quipped *They 3 years behind*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Activity Export Feature Launches with Fanfare**: The **Activity Export** feature is live, enabling users to export up to **100k rows** to **CSV** for free, with questions raised regarding export times.
   - A user suggested that if the data exceeds **100k rows**, it should be truncated, instead of completely aborting the export process, referencing the [Activity export](https://cdn.discordapp.com/attachments/1370059676083032185/1370074702835486811/image.png?ex=681e2cff&is=681cdb7f&hm=244eca26755137a11f65cc8a74d2c522dbb8d8040f0a8077e7c619db5b571fc5).
- **Local Proxy Channels OpenRouter Requests**: A user planned to use a **local proxy** to forward requests to **OpenRouter**, while another pondered how to make **completions extend out of the mouse cursor**.
   - The latter user suggested that with the right keyboard shortcut, this could become part of **muscle memory** but is a *very nostalgic* UI.
- **OlympicCoder 32B Craves Comeback**: Users expressed strong interest in the return of the **OlympicCoder 32B** model, with one user expressing a desire for it to *miraculously come back*.
   - The group did not discuss any specific details about its current status or reasons for unavailability.
- **OpenRouter API's Cost Accounting Unveiled**: A user inquired about retrieving cost information alongside usage when prompting a model, and another directed them to the [OpenRouter documentation on usage accounting](https://openrouter.ai/docs/use-cases/usage-accounting).
   - The documentation provides details on how to track and manage costs associated with **API usage**.
- **OpenRouter API Experiences Hiccups, Ditches Image Prompts**: A user reported a **404 error** when accessing the [OpenRouter API endpoint](https://openrouter.ai/api/v1/chat/completions), potentially indicating an outage.
   - Users discovered that **OpenRouter** does not currently support image generation, resulting in a **404 error** when attempting to use image prompts with models like *opengvlab/internvl3-14b:free*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Windsurf code coming to Copilot Proxy**: A GitHub employee confirmed that **Copilot Proxy** users no longer need to cancel, because **windsurf** is coming soon, according to [this X post](https://x.com/alexalbert__/status/1920207966256705888).
   - Previously the copilot proxy was forked from **Windsurf**.
- **Aider gains web search capabilities**: Members discussed using **Perplexity API** as an OpenAI compatible API to enable web search in Aider, or using **/web** to include specific webpages.
   - A member suggested using a script to query **Perplexity** or **Perplexica** and add the outputs as markdown files to Aider's context.
- **Implicit Caching enabled for Gemini 2.5**: Google is enabling **implicit caching** for Gemini 2.5 models as described in [this Google blog post](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/) and [this X post](https://x.com/googleaidevs/status/1920525127772721589).
   - Members noted that the new `gemini-2.5-pro-preview-05-06` model takes *way too long* before it responds, preferring the old March one, and that *it uses more time thinking*.
- **Aider can get stuck in debug loops**: Aider can get stuck in a debug loop with **Gemini** (and likely other LLMs), but this can be resolved by presenting it with multiple error sets and prompting it to consider a different implementation.
   - The member wondered if *conversational context* is too low for Aider to catch its own debug failure loops.
- **Claude Code allegedly inspired by Aider**: A member shared a [YouTube video](https://www.youtube.com/watch?v=zDmW5hJPsvQ&t=1s) claiming that **Claude Code** was inspired by **Aider**.
   - Paul Gauthier responded *Imitation is the sincerest form of flattery*, mentioning that Aider is still better and less expensive.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tilelang streamlines Kernel Development**: **Tilelang**, a new domain-specific language (**DSL**), simplifies the development of high-performance GPU/CPU kernels like **GEMM** and **FlashAttention**.
   - Tilelang allows streamlined development and higher performance in these crucial computational kernels for CPUs and GPUs.
- **Atomic Addition Causes Non-Deterministic Disasters**: Using **atomic_add** can lead to varied results due to floating-point addition order, regardless of precision, for example `1e-8 + 1e8 - 1e8`.
   - **FP16** is less sensitive than **BFP16** in atomic addition contexts; adjust the `tol` parameter in tests based on float dtype, as seen in [this Python code](https://pytorch.org/).
- **Torch Compile Plummets Performance**: A simple `torch` combo function (**TensorMax(ReLU(Matmul(A, B)))**) performed better *without* the `@torch.compile` decorator than with it on an **A100** using **PyTorch 2.7** and **Triton 3.3**.
   - The slowdown with `@torch.compile` might stem from **compilation overhead**, negating kernel fusion benefits for smaller operations; investigating the generated Triton code could expose bottlenecks.
- **Submissions start on MI300 Leaderboard**: Multiple users submitted benchmarks to the `amd-fp8-mm` leaderboard on **MI300**, with submissions ranging from **183 µs** to **27.2 ms**, where one even reached **3rd place** at **183 µs**.
   - A member submitted results to the `amd-mixture-of-experts` leaderboard with timings of **6604 ms** and **7840 ms**, which demonstrates ongoing work in the mixture of experts domain.
- **PTX Programming Primer Published**: A blog post offers a beginner's guide to programming **NVIDIA Tensor Cores** via raw **PTX mma instructions** and inline PTX assembly, and sidesteps CUDA with explanations of register constraints for datatypes like **float8**; the [blog post is here](https://veitner.bearblog.dev/a-short-note-on-tensorcores-and-inline-ptx-assembly/)
   - The **H100** only has **QGMMA**, not QMMA, and using `mma` with an **fp8 type** compels the compiler to up-convert to **FP16** and use **HMMA**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AnythingLLM Errors Plague LM Studio Users**: Users reported errors using **AnythingLLM** with **LM Studio**, and asked for help, with one member suggesting enabling **CORS** as a potential fix, even when running locally.
   - Another member suggested checking the logging pane in the developer view of LM Studio to diagnose the issue.
- **Class Variables Save the Day**: A member found that using a **class variable** was the only way to get their code working in a coding project.
   - Another member shared a [Reddit comment](https://www.reddit.com/r/Python/comments/u0j5rn/comment/i49bjhf/) about injecting the variable at runtime, potentially providing an alternative solution.
- **Gemini Code Changes Irk Users**: Users are complaining that **Gemini** completely changes code, even when instructed to provide a minimum change, frustrating their efforts.
   - Members noted that other models, like **Qwen**, are better for simple refactors, because Gemini can easily double or triple the code length with comments and try/except blocks.
- **Mistral Medium 3 Declared Mediocre**: A user tested **Mistral Medium 3**, finding it to be a *non-reasoning model* with *baked in chain of thoughts*, resulting in x2.08 token verbosity.
   - They concluded the model's capability was mediocre, placing it between **Mistral Large 1 & 2**, similar to **Gemini 2.0 Flash** or **4.1 Mini**, and not *SOTA performance at 8X lower cost* as claimed in marketing.
- **Web Search for LM Studio Users Pined For**: A user requested easy-to-use web search capabilities and **RAG** built into LM studio, with features like uploading a PDF and searching in a webview.
   - One member suggested it's possible now but fragile with many components that can go wrong, and another suggested using **openweb-ui** and attaching it to **LM Studio**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Defining 'Cringe' Catches On**: Members explored the emerging **internet slang** definition of *cringe*, proposing specific instructions to minimize its presence in **AI responses** and shared [a YouTube video](https://www.youtube.com/watch?v=59wV96Kc3dQ) defining cringe.
   - The discussion highlighted the need for **AI models** to better understand and avoid generating content perceived as *cringe*.
- **Manus Launch Still Missing**: Users are still awaiting the launch of **Manus**, frequently checking their social media for updates.
   - The launch was anticipated on **March 28, 2025**, based on a screenshot, but this date has passed without any launch.
- **Manus Credit Costs Discussed**: Members recalled pricing for additional **Manus credits** at **$19 for 1900 credits** or **$99 for 9900 credits** and linked to the [Manus Help Center](https://manus.im/help/credits).
   - It remains uncertain whether these pricing options are still available.
- **Manus Taps Claude's LLM**: Following speculation on whether **Manus** uses its own **LLM** or **Claude’s LLM**, co-founder **Peak-ji** confirmed that Manus leverages a mix of tools, including **Claude**, detailed in [a Twitter post](http://x.com/peakji/status/1898994802194346408).
   - Further confirmation of the use of open-source code is available in [github posts](https://gist.github.com/jlia0/db0a9695b3ca7609c9).
- **Manus Phone Verification Frustrates Users**: A user reported issues with **Manus's phone verification**, noting that the *phone verify thing doesnt work*.
   - They raised concerns about the necessity and privacy implications of this feature, questioning how the system tracks code usage without linking it to an account.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **ACE-STEP Hits SOTA Status in Music**: A member showcased the **ACE-STEP SOTA** music generation model, featured in a [YouTube video](https://youtu.be/vyCALtrq4yQ).
   - This was shared in the `#i-made-this` channel and reflects ongoing advancements in AI-driven creative tools.
- **Alpha-Root Unearths Cyber Intel with Finesse**: **Alpha-Root** extracts cyber-security data by mining domains directly on the common crawl web graph, matching the performance of **PRIMUS-FineWeb** with ~10x less resources and data, according to a [draft preprint](https://github.com/ashim-mahara/alpha-root/blob/main/Cybersecurity_Data_Extraction_from_Common_Crawl-3.pdf).
   - The author extracted **3B tokens** from **FineWeb-Edu** without a classifier by finding URLs present in both **Alpha-Root** and **FineWeb-Edu**, and only including if present.
- **Dropwise Drops In, Brings Uncertainty Estimation**: A member announced the release of **Dropwise**, a PyPI module for **uncertainty estimation** in Hugging Face classification models using **Monte Carlo Dropout**, detailed on [GitHub](https://github.com/aryanator/dropwise) and [Docs](https://pypi.org/project/dropwise/).
   - It integrates with `transformers` pipelines and is valuable for QA, classification, OOD detection, and active learning.
- **RAG Repo Sparks Cheating Clash**: Students in the **AI agents course** debated whether using **RAG with answer + clone repo** constitutes cheating, feeling like it undermines the leaderboard's integrity.
   - Some argued it removes the value of trial, error, and iterative improvements in the agent development process.
- **API Limits Prompt Pro Version Panic**: A user doing the **AI agents course** hit the **20 requests per month limit** before finishing the first unit, and wondered whether they had to pay for the Pro version to continue.
   - A second user mentioned that you could run a local LLM with **ollama** or find other free tiers.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Claude Cannot Chart Plotly**: Members noted that **Claude** can’t display **Plotly** charts directly as an **MCP** client, but handles **ImageContent** and **EmbeddedResource** formats like **PNG/JPEG**.
   - The recommended workaround is rendering charts as **PNG/JPEG** images for display in **Claude**.
- **Token Limits Laid Bare**: The discussion clarified that **max tokens** in **MCP** specifies the maximum tokens in the response, akin to **max_tokens** in completions API requests.
   - The total token count (**system prompt + messages + output message**) must remain within the context window size.
- **LLM Restrictions Frustrate Users**: Users are encountering issues with **LLM** restrictions (like **Deepseek**) that prevent filesystem access, affecting their **MCP** system functionality.
   - It appears some models intentionally restrict filesystem access, causing problems for legitimate use cases via **MCP**.
- **Cloudflare MCP Servers Face Connectivity Woes**: Some users reported connectivity issues with **remote MCP servers** on **Cloudflare**, while others had functioning setups.
   - Troubleshooting involves examining the specific **MCP server repo** for connection problems.
- **Zinja Unleashes New STDIO MCP Client**: **Zinja** released [a lightweight, fast, CLI-based MCP client](https://github.com/zinja-coder/zin-mcp-client) for STDIO MCP servers, bridging local LLMs and MCP servers.
   - It's designed for use with **jadx mcp servers** to perform AI-assisted reverse engineering of Android APKs using local LLMs.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **China's RL Robots Zoom Past Deepmind**: A [YouTube video](https://www.youtube.com/watch?v=ET-MmoeSvXk) compared **Google Deepmind's RL Robot achievements** from a year ago to more recent **Chinese RL Robot achievements**, indicating rapid advancements in physical AI evolution.
   - The video highlights the progress being made in **robotics and reinforcement learning** in China, suggesting a shift in the landscape of AI development.
- **Apple Silicon Steals Inference Crown**: Members compared **Linux laptops** with **Nvidia GPUs** against **Apple MacBooks** with **M-series chips** for local inference, with most favoring MacBooks due to enhanced performance and power efficiency.
   - The unified memory platform in **Apple's M-series chips**, which allows the CPU, GPU, and AI ML neural net to share the same memory, eliminates the need for frequent data transfers.
- **Llama 4 Leaves a Bad Taste**: A member expressed disappointment with **Llama 4's performance**, finding it inferior to **Qwen3** and suggesting a wait for **Llama 4.1**.
   - The discussion included a suggestion to consider *going back to 405 dense for the next big model*.
- **Discord's Emoji Ban Spooks Users**: The **automatic chat-moderation system** blocked certain **multi-part emojis** due to zero-width joiners and variation selectors used to combine codepoints, a technique scammers also use to bypass filters.
   - The discussion revealed that *dev role has been taken off the autoblock list* in response to this issue.
- **Entropy Engine Fires Up Quantum-Native Randomness**: A member launched a [quantum-native yet algorithmic entropy engine](https://github.com/thyarcanist/Entropy-MicroDemo) for public testing, describing it as *self-promo* but important to share given its potential impact on **AGI**.
   - The member believes that **LLM** outputs are highly sensitive to the quality of the **randomness** used, distinguishing between true **entropy** and **PRNG** and implying that high quality entropy unlocks different, and often better, behaviors in models, linking to [several Xitter posts](https://x.com/thegautam/status/1920198569308664169?t=GehCezJb7amBPoter8F0gA) in support.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Grok's Reality Apprehension Susceptible to Propaganda?**: Members speculated that **Grok's** apprehension of reality could be nerfed to favor right-wing propaganda as shown in [this image](https://media.discordapp.net/attachments/738904561041014845/1368039830549692516/pz3v4ft279ye1.png).
   - The submitter lamented that *all problems today already existed*, and that *AI or no AI* we would still have them.
- **Cloudflare Allegedly Serves Up Fakery**: Members think that **Cloudflare** is serving fake content to **AI agents** leading to biased responses.
   - This action is allegedly similar to how some Chinese websites used zip bombs to deter cloning years ago, and comes after **ChatGPT** wrongly answered about a video that a member shared.
- **LLM Output Urgently Needs Filters?**: A member suggested that we need third party filters for **LLM output**, including adblocking and fact/bias checking.
   - In response, another member suggested that you'd need many models that ideally change often so they don't get corrupted such as *100 adblocker models and 100 fact checking tools*.
- **Zed now compiles on Windows, but...**: A member successfully compiled **Zed** on Windows using [these instructions](https://github.com/zed-industries/zed/blob/main/docs/src/development/windows.md), but fonts appear blurry.
   - Also, users must sign in with **GitHub** to enable tab completion, which disappointed another member who wanted to try **Mellum 4B** on **LM Studio** for tab completion.
- **Biological Brains Don't Backpropagate?**: A member stated that *biological brains don't have backpropagation; they're non-epochal, spiking, recurrent, analog networks*, and cited [this Tweet](https://x.com/_neel_kant/status/1920516491025482066) as evidence.
   - The member contrasted **backpropagation** and what happens in biological brains.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Discord Debates Cursor Advertising Rule**: Members debated whether posts about **Cursor** constitute advertising and violate the no-advertising rule, noting *'its just even toleratable bc we (group) think of cursor as useful right now but it still biases decisions'*.
   - Some users suggested that vague rules are being applied arbitrarily, along with interpreting 'no advertising' as 'no spam', and requiring payment for job postings could filter out low-quality offers.
- **User Finds Slurm Memory Misconfiguration**: A user discovered they were requesting **80MB** of memory through **Slurm**, not **80GB**, calling it a *'slurm moment'*.
   - The initial issue was described as *'very stupid'* by the user who discovered the misconfiguration, while another user celebrated their bare-metal setup.
- **Community Discusses Job Postings on Discord**: Discussion arose around creating a jobs channel, with concerns that it could be overrun by low-quality postings offering *'experience'* as compensation.
   - Others argued against a jobs channel, suggesting it would make the server another place for recruitment and proposing EleutherAI shouldn't charge for differential access to the Discord server.
- **Linguistics Channel Gains Traction**: A user proposed a channel for classical linguistics and its theory, focusing on pre-2000s knowledge such as sentence formation and meaning creation *'on the fly'*.
   - It was described as *'cool stuff that rarely gets discussed in the NLP world for 'some' reason (probably because it's irrelevant to the work nowadays).'*.
- **Prolific Prevails Over MTurk for Human Evals**: Members recommend [Prolific](https://www.prolific.co/) over **MTurk** for human evaluations, citing its higher quality data and more reliable participant pool.
   - The consensus is that Prolific is the superior choice in approximately *80% of cases*.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Launches Mobile App Beta with Tester Program**: **NotebookLM** is launching a mobile app beta 📱 and seeks experienced web app users for a trusted tester program to improve the app.
   - Interested users can register via [this form](https://forms.gle/XD1VmJ7FP4AjbDB66) to provide feedback and report bugs, agreeing to the **Trusted Tester Terms**.
- **PDF Processing Suffers with File Size and Page Count**: Users report that **NotebookLM** has issues with larger PDFs; one user found problems after **200 pages** when asking questions further into the PDF.
   - Users suggest further experimentation to empirically test the current limitations of **NotebookLM**.
- **Sales Teams Embrace NotebookLM for Knowledge Base**: A user is creating a sales content knowledge base in **NotebookLM** with client decks and sales enablement materials within the 300-document limit.
   - The user is seeking examples and guidance on limitations, particularly regarding sharing and potential silos for the internal sales team.
- **Podcast Length Depends on Input Content and Language**: A user found that changing the language to **English** allowed for significantly longer audio summaries (up to **49 minutes**), whereas other languages were limited to around **14 minutes**.
   - A team member confirmed that this is expected and work is underway to enable longer audio summaries in more languages soon.
- **NotebookLM System Refuses to Answer Prompts**: Users are reporting that **NotebookLM** sometimes responds with *'The system was unable to answer'*, even when asked to summarize the default notebook, with issues also arising when generating mind maps and study guides.
   - Users are reporting the issue across several channels, seeking confirmation and solutions.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Netflix Recommends Foundation Model**: **Netflix** developed a [foundation model for personalized recommendations](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39).
   - It was pointed out in relation to other discussions on recommendation systems.
- **Gemini Generates Images**: Members shared a link showcasing [new Gemini image generation](https://x.com/OfficialLoganK/status/1920151503349711061).
   - A member mentioned that *this team will be presenting at the aie world’s fair recsys x llms track*.
- **Aider Autopsies Gemini Cost**: Members noted how [aider postmortems](https://aider.chat/2025/05/07/gemini-cost.html) are very thorough, especially regarding Gemini cost analysis.
   - The community appreciated the detailed breakdown.
- **Suno Sings the Blues (and Yodels)**: A member raved about **Suno's** ability to mix styles, highlighting a successful attempt at creating a *Yodel + Blues + Live concert* mix and shared an [audio file](https://cdn.discordapp.com/attachments/1075282825051385876/1370022129050849441/you_can_YODEL_with_Suno.mp3?ex=681dfc09&is=681caa89&hm=e16a84ff105d7fc1bef2fd343a067b7ea6ffa1964772d2e3ad9900e355f2d2c2&) as evidence of **Suno's** impressive output.
   - The community enjoyed the unique blend of genres.
- **AI Engineer Conference Buzz Builds**: The AI Engineer conferences, slated for June, alerted community members that [Early Bird tickets](https://www.ai.engineer/#speakers) are expected to sell out by the weekend.
   - Enthusiasts are eager to see the expertise and insights the speakers will bring to the conference, as displayed in [the lineup of speakers](https://www.ai.engineer/#speakers).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Properties Puncture Fields in Mojo Traits**: Discussion confirmed that *properties in traits* are superior and more versatile compared to fields in Mojo, enabling greater flexibility, but fields in traits *could* happen.
   - The group debated how one would be denied the ability to add such a trait via an extension; it would need to be included in the original struct definition.
- **Modular Hackathon Hypes Hillsborough**: A final reminder for the Modular Hackathon at AGI House this Saturday, with signups available [here](https://app.agihouse.org/events/modular-hackathon-20250510), featuring Modular team members, Mark Saroufim (GPU MODE & PyTorch), Simon Boehm and Sasha Krassovsky (Anthropic), and Dylan Patel (SemiAnalysis).
   - Attendees will explore cutting-edge developments in modular programming and hardware acceleration for machine learning.
- **Hardware Agnostic ML Survey Surfaces**: A member completed and shared their survey paper on modularity and the **Hardware Lottery**, designed to present a compelling narrative to peers.
   - The latest version of the paper is available [here](https://github.com/TheAgaveFairy/HPML-Survey-Project/blob/main/The_Quest_for_Unification__A_Survey_of_Hardware_Agnostic_Machine_Learning_Systems.pdf), welcoming feedback.
- **Zotero Zaps Citation Struggles**: Members recommended using **Zotero** + **bibtex** to simplify citation management, helping avoid common issues.
   - One member shared *natbib gave me about 70 errors with almost nothing linking until i caught a single unescaped '%'*.
- **Mojo Roadmap unveiled!**: Modular posted the near-term **Mojo roadmap** on the forum, view the [official post](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395).
   - The roadmap details the upcoming features and improvements for the **Mojo** language.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Project Eyes Collaborative Horizon**: A member expressed interest in a **collaboration and partnership** to mutually benefit their communities with **DSPy**.
   - The member proposed initiating a chat to explore potential synergies, highlighting a proactive approach to community growth.
- **ReAct Module Signature Simplified?**: A member inquired about creating a **ReAct module signature** that only makes tool calls without needing additional outputs.
   - Another member suggested using *success: bool* as the output, indicating task completion, streamlining the module's output.
- **DSPy's Cache: The Layers Revealed**: A member discovered that **DSPy** has its own caching mechanism ([github.com/stanfordnlp/dspy/blob/main/dspy/clients/cache.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/cache.py)) in addition to the **LLM** provider's cache, potentially causing unexpected results when credentials expire.
   - The multiple layers of caching from **DSPy**, **LiteLLM** ([docs.litellm.ai/docs/proxy/caching](https://docs.litellm.ai/docs/proxy/caching)), and **Bedrock** can complicate debugging efforts for AI Engineers.
- **GRPO Learns, Recall wavers**: A member conducted a small **RL experiment with GRPO** on a **Qwen 1.7B** using DSPy to optimize query rewriting for retrieval, initially observing a baseline recall drop from **28%** to **26%** after training.
   - Further details are available in [a Twitter thread](https://x.com/tahmidtapadar/status/1920469176776302679), attributing the drop *likely due to sparse rewards, short runs, and BM25 mismatches with CoT rewrites*.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Embeddings Model Stumbles in Negotiation**: A user found that **Cohere's embedding model** poorly handles negotiation scenarios, returning a high similarity score (**0.92**) between contradictory statements like *"I can pay"* and *"No, I cannot pay"*.
   - A member suggested leveraging **Cohere's rerank model** as a more suitable alternative for tasks beyond simple vector similarity.
- **AIBillingDashboard Tracks AI Costs Across Platforms**: A software engineer launched [AIBillingDashboard.com](https://AIBillingDashboard.com) to track and optimize **AI service costs** across providers like **Cohere**, **OpenAI**, **Anthropic**, **Azure AI**, and **Google Vertex**.
   - The platform aims to solve the pain points of manually pulling reports and allocating costs, seeking feedback on pricing comparisons and justifying **AI expenses**.
- **Decoding Command A's GPU Needs**: A user is investigating the **GPU requirements** for an **on-premise installation** of **Command A**.
   - Understanding the necessary **GPU specifications** is crucial for successfully deploying and running **Command A** within their infrastructure.
- **Embedding Models Encounter Hiccups**: **Cohere** reported [degraded performance](https://ift.tt/WvxjUwp) affecting the **embed-english-v2.0** and **embed-english-v3.0** models, and are investigating the issue.
   - For further details, refer to the [Cohere Status Page](https://ift.tt/bE5aXAs); updated May 08, 2025, at 07:25AM.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Automating Tokenizer Identification in Torchtune**: A member is automating tokenizer identification across model types for internal customers using `torchtune` to remove the manual step of identifying the tokenizer, aiming for generic usage.
   - The plan involves a custom *autotokenizer* with conditional statements for model name identification in the config for tokenizer and checkpointer settings.
- **HuggingFaceBaseTokenizer Limited for SFT**: `HuggingFaceBaseTokenizer` lacks logic for templating/tokenizing messages, restricting its use to text completions training and not **SFT** (Supervised Fine-Tuning).
   - To bridge this gap, a `ModelTokenizer` wrapper is planned to map **HF's** `apply_chat_template` to *Torchtune's* `tokenize_messages` and an issue will be opened on the [repo](https://github.com/pytorch/torchtune).
- **Cosine Shenanigans Lead to NaN Weights with Adam**: A **PyTorch bug** causes **NaN weights** when using a compiled non-fused **Adam/AdamW** optimizer with a learning rate scheduler that sets the learning rate to exactly 0 at any point during training, specifically when using a [cosine scheduler with warmup](https://github.com/pytorch/torchtune/pull/2681).
   - One member suggested looking at the [Torchtitan implementation](https://github.com/pytorch/torchtitan/blob/00a53646c184493d292836f7d8bbe0bed859993f/torchtitan/components/lr_scheduler.py#L120) which sets the LR ratio to `1/(warmup_steps+1)` on the first step.
- **Titan's Warmup: LR Scaling Strategy**: A discussion about LR scaling strategy during warmup proposed an alternative to `0,1/n, 2/n, ..., n-1/n`, instead suggesting `min_lr + (1/n ) * (1 - min_lr), min_lr + (2/n ) * (1 - min_lr), ..., min_lr + (n-1/n ) * (1 - min_lr)`.
   - This scaling is combined with scaling the progress by the inverse of the cosine schedule using `progress *= arccos(2*min_lr-1)/(pi*2.0*num_cycles)` will result in your max progress computed so that `cosine_lr_multiple == min_lr`.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Anthropic API Gets Search Tool**: The **Anthropic API** now natively supports web search, immediately supported in **LlamaIndex** [according to this Tweet](https://twitter.com/llama_index/status/1920220803976867882).
   - This integration allows for enhanced information retrieval capabilities within LlamaIndex applications.
- **LlamaParse Adds Power**: **LlamaParse** is improving with new features like **GPT 4.1** and **Gemini 2.5 Pro** models, plus auto orientation, skew detection and confidence scores for parsing quality [according to this tweet](https://twitter.com/llama_index/status/1920505775677722750).
   - The new features promises to enhance the accuracy and reliability of document parsing within the LlamaIndex ecosystem.
- **VoyageAI Multi-Modal Retrieval Voyage with MongoDB**: Users can now implement multi-modal retrieval using **VoyageAI's** multi-modal embeddings and **MongoDB's** multi-modal indexes [in this notebook](https://twitter.com/llama_index/status/1920563641990209643).
   - This integration streamlines the process of handling and retrieving data from multiple modalities.
- **Medical LLM Bot Seeks Workflow Guidance**: A user is constructing a medical LLM bot and wants help building a workflow iteratively suggesting follow-up questions based on previous answers from a local LLM.
   - The user wants help to determine if LlamaIndex has tools to help build this kind of workflow.
- **Fine-Tuning for Math Formula**: A user is seeking guidance on fine-tuning the **vdr-2b-multi-v1** model using the **llamaindex/vdr-multilingual-train** dataset to better handle complex math formulas.
   - The user is looking for resources, steps, or tutorials for fine-tuning to recognize math formulas.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Deep Dive Into Tinygrad's CUDA**: A user explored **tinygrad's CUDA integration** and inquired about its own **Intermediate Representation (IR)** for handling CUDA operations.
   - This prompted discussion around how **tinygrad** leverages CUDA for optimized computations.
- **Tinygrad Documentation Trove Shared**: A user shared the [official tinygrad documentation](https://docs.tinygrad.org/) and linked to notes on [tinygrad uops](https://xl0.github.io/tinygrad-notes/uops.html) for low-level operations, and other [tinygrad notes](https://mesozoic-egg.github.io/tinygrad-notes/).
   - These resources provide insights into **tinygrad's architecture**, **operation details**, and implementation strategies, particularly at the micro-operation level.
- **CACHEDB Variable Location Spotted**: A user inquired about the **CACHEDB** environment variable, with another member pinpointing its mention at *line 175 in helpers*.
   - Its specific function and practical context within the project requires further examination.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda hosts AgentX Workshop with Prizes**: Lambda is hosting the **AgentX Workshop: Building Agentic AI with Lambda on 5/15 10am PT** for AgentX competition participants, who can also compete for up to **$1,000 in credits for 1st place**, **$500 for 2nd**, and **$300 for 3rd**.
   - Participants will learn to build agentic applications and deploy agents in production, and can [register now](https://lu.ma/AgentX-lambda) to get the YouTube livestream link.
- **Users Await Hugging Face Credits**: Users reported issues with tracking **Hugging Face credits**, with one not receiving emails and the other awaiting approval.
   - The first user found it *challenging to visit the website each day*.
- **LLM Agents Course Content Clarified**: The staff clarified that the guest lectures listed on the [course website](http://llmagents-learning.org/sp25) are indeed comprehensive and also confirmed that the **Spring MOOC** includes more advanced topics like *code generation and theorem proving*, whereas the **Fall version** includes more *applications topics*.
   - A user inquired about future iterations of the course, specifically if there would be another offering in the **Fall**, to which staff replied that Prof Song is hosting another Berkeley class on *Agentic AI* this fall, but it is unknown whether it will be a MOOC version.
- **LLM Agents MOOC Recommended for Aspiring AI Engineers**: A member inquired about the best complete course to become an **AI Engineer**, with another member recommending starting with the [Fall 2024 LLM Agents MOOC](https://llmagents-learning.org/f24).
   - The LLM Agents MOOC was suggested as a solid starting point to start an AI Engineer Career Path.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Surfs High with Wave 8, Boosting UX and Plugin Power**: Windsurf's final **Wave 8** release enhances the **JetBrains plugin** and improves the **Windsurf Editor** UX, detailed in a [blog post](https://windsurf.com/blog/windsurf-wave-8-ux-features-and-plugins) and [changelog](https://windsurf.com/changelog).
   - The update aims to streamline user workflows and provide more intuitive interactions within the development environment as showcased in [today's launch video](https://youtu.be/IjE8Cdxotso).
- **JetBrains Plugin Gets Memory and Rules**: The updated **JetBrains plugin** now supports **Memories** for persistent information between sessions and **Rules** via `.windsurfrules` files.
   - It also introduces **MCP** (Model Context Protocol) server connections, as outlined in the [Jetbrains plugin changelog](https://windsurf.com/changelog/jetbrains), allowing for more contextual and persistent interactions.
- **Windsurf Editor's UX Revamp: More Than Just a Facelift**: The **Windsurf Editor** UX has improvements like a **Continue button**, redesigned model selector, and workspace-to-conversation mapping for filtering history.
   - Additional enhancements include enhanced code blocks and hunk navigation, editable terminal commands, and new file proposals in Chat mode, designed to make coding smoother and more efficient.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Nomic.ai (GPT4All) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1369751825451847750)** (554 messages🔥🔥🔥): 

> `Qwen3-14B, Mistral vs Gemma vs Phi-4, AMD GPU, Model quantization` 


- **Qwen3-14B hailed as Top Pick for Coding, Reasoning, Conversation**: For building an AI with coding, reasoning, and conversation skills, a member suggests that the **14B Qwen** model is the *best all around* choice.
   - The member specifies that this applies to both the base and instruct versions of the model.
- **Phi-4 Shines in Ease of Fine-Tuning Faceoff**: Members compared **Gemma3**, **Mistral**, and **Phi-4**, highlighting **Phi-4's** easy fine-tuning; *It seemed to just drink up whatever I wanted to train it with*
   - Others note challenges keeping **Mistral's** instruction following aligned after LoRA merge and express difficulty achieving success with the **Gemma 3 27B** flavor.
- **AMD GPU Support Coming Soon**: Despite challenges, Unsloth is working with AMD to support **AMD GPUs**.
   - According to a contractor, expect AMD support *anywhere before Q3 this year if its that fast*.
- **Unsloth's Dynamic Quantization Method**: **UD** is Unsloth's dynamic quantization that applies to q4 and lower, and **UDq6 KXL** *might be the best quant ever*
   - It's a fork of *llama.cpp's quantization but it's 100% compatible with normal llama.cpp and lmstudio/ ollama*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1369758666139369532)** (22 messages🔥): 

> `AI Project Hiring, LLM for text punctuation, LLM Recommendations, Qwen vs Gemma3 Model, IBM Granite 4.0 Mamba Model` 


- **AI Project Seeks Personnel**: A member is looking for a reliable person for an **AI Project** (tech skills not mandatory), offering **$500 weekly** (part-time) to citizens of the USA, Australia, Canada, UK, Switzerland, Netherlands, or Germany.
- **Gemini 2.5 Pro Can Punctuate Long Stories**: A member looking to punctuate a long story was recommended **Gemini 2.5 Pro** via AI studio for its lack of limits and **65536 output length**.
- **Lightweight LLM Needed for Documentation Review**: A member needs recommendation for a lightweight LLM model (**24B Q8 or more small**) to critique documentation in a logical way and be exported to **GGUF** after CPT and fine-tuning under the **Unsloth** environment.
   - They have already tried **gemma-3 12B, phi-4 reasoning**, and **glm-4**, but all of them failed to export to gguf. They also tried **llama 3.3** and old **mistral**, but the performance was not satisfactory.
- **Qwen May Be Default Open-Weight Choice**: When recommending a model, a member suggested that *the current state of open-weights LLMs is that **Qwen** is the default choice for pretty much anything, and only if it fails you go look for something else*.
- **Qwen's MoE architecture may backfire**: A member opted for **gemma3** because they need only a single domain to train it on and, during inference, only few parameters get activated for **Qwen** (as its MoE), unlike **gemma3 1b** which is Dense, and wonders if this hypothesis is correct that this MoE architecture may backfire.
   - Another member corrected the misunderstanding that **Qwen** has dense models as well from [their github](https://qwenlm.github.io/blog/qwen3/).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1369755909424943335)** (92 messages🔥🔥): 

> `phi4-mini-instruct training issues, Qwen3 model compatibility with vLLM, Kaggle notebook using multiple GPUs, Tokenizer configuration differences, Qwen3 model not thinking` 


- ****Phi4-mini-instruct Struggles with Data-Dependent Branching****: A user reported that **phi4-mini-instruct** keeps erroring during training due to *Unsupported: Data-dependent branching*.
   - The user also noted that they haven't been able to work with any of the **phi4** small models under 5GB and that the only model that works is **phi3.5**.
- ****Qwen3 Finetuning Fails to Engage Brain**: After finetuning **Qwen3**, a user reported that the model wouldn't "think" when prompted, even after using the official **Unsloth** notebook and formatting the data correctly with the `<think>` tags.
   - They found that the model would just skip the `<think>` tags and that the only workaround was to force the model to think by adding *Okay,* after `<think>`, but the performance was poor.
- ****Tokenizer Configs Drift in Unsloth's Qwen3 Base****: Users noticed the `tokenizer_config.json` differs between `unsloth/Qwen3-0.6B-Base` and `Qwen/Qwen3-0.6B-Base` on HF, where the **Unsloth** version removes the chat template and swaps `pad_token` to `<|vision_pad|>`.
   - It was theorized that *Qwen3 base isn't supposed to have a chat template at all*, and the team was going to ask the Qwen team to confirm.
- ****No LM Head? No Problem! (Gemma Edition)****: A user encountered a warning about *missing keys in the checkpoint model loaded: ['lm_head.weight']* when doing full finetuning with a **Gemma-3** model.
   - It was resolved that **Gemma-3** uses weight tying, so the LM head re-uses the same tensor as the input embeddings, and the warning can be safely ignored as long as `config` has `tie_word_embeddings=True`.
- ****Whisper Model Can't Generate? (Disable Fast Gen!)****: A user ran into a `TypeError: You need to pass in input_ids to .generate!` when trying to use a finetuned **Whisper** model with **Unsloth**.
   - A contributor suggested using `%env UNSLOTH_DISABLE_FAST_GENERATION = 1` and restarting the runtime as a workaround.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1369753907571724378)** (24 messages🔥): 

> `Gemma3 27b hooking input/output layers, Process Reward Model (PRM) training challenges, Finetuning Audio Understanding Models, DeepSeek-R1 vs other reasoning models for COT reasoning` 


- **Hooking Gemma3 Layers Sparks Interest**: A member reported that hooking the input and output layers of **Gemma3 27b**, with just one bolt-on memory layer to force crosstalk, still results in valid generation.
   - Interestingly, they noted that *hooking the middle layers is what breaks the model*.
- **Process Reward Model Training Headaches**: A member inquired about training a **Process Reward Model (PRM)** for text generation tasks like creative writing, legal, or medical text, asking *what will be the reward signals mostly*.
   - They sought advice and experiences related to similar challenges.
- **TTS Notebooks Available**: Unsloth AI now has **TTS notebooks**: [Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks#text-to-speech-tts-notebooks).
   - However, finetuning for TTS may not translate directly to *Audio understanding* models like Kimi.
- **DeepSeek-R1 Chosen for Cost**: Discussion arose around why **DeepSeek-R1** was chosen for COT reasoning in a competition, and the rationale may be based on **cost**.
   - A member quoted the paper abstract, indicating that stronger models were available, but the higher token generation made them unfeasible due to competition constraints ([Kaggle discussion](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/574765)).


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1369750676581646458)** (642 messages🔥🔥🔥): 

> `Grok 3.5 release, Grok 3.5 never comings, EMBERWING model, LLM and Politics, Gemini 2.5 pro nerf` 


- **Grok 3.5 release date still unconfirmed**: Despite earlier tweets, members express doubts about the [imminent release of **Grok 3.5**](https://x.com/Nate_Esparza/status/1920480721334145149), with some suggesting that earlier claims were likely premature.
   - It's been suggested that even **Elon** may not know and that the real plan is for the 'sarcastic tone' bot **Gork**.
- **EMBERWING flies into the Arena**: A new model named **EMBERWING** has entered the arena, with initial assessments indicating it's a *Google* model with strong multilingual capabilities but disappointing in reasoning.
   - Members speculate **EMBERWING** could be an iteration on **Dragontail** and an update for *Flash*.
- **Debating EU's LLM innovation stagnation**: Some members discuss reasons why the EU isn't as innovative in the LLM space, listing strict regulations, overspending on things like pronoun inspections, and mass migration.
   - One member responded that it was *'ragebaiting'* and that *'migration is absolutely a good thing'*.
- **Gemini 2.5 Pro possibly nerfed**: Members noted that Gemini 2.5 pro may have been nerfed, with one user saying, *'The first rule in this field is "if something works don't change it"'*.
   - Another memeber rebutted, *'if you don't innovate you lose traffic'* and shared a link to show it scores higher in [leaderboard lm areana](https://x.com/OfficialLoganK/status/1920523026551955512?t=P1Laq9w5K35YMiS5OmD6Yw&s=19).
- **Diving Deep into the OpenRouter Ranking Illusion**: Members debate the validity of OpenRouter rankings due to various factors, including business models, user demographics, and biases towards cheaper models.
   - Reasons include:
A) slow to update B) skewed by programmers looking for uptime and circumventing API tiers and C) free model offerings distort rankings.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1370107496886304788)** (1 messages): 

> `Perplexity AI, Reddit AMA, Deep Research, Live Q&A` 


- **Perplexity AI Team Hosts Reddit AMA**: Brett Chen and Thomas Wang from the **Perplexity AI** team are hosting a live **Reddit AMA** to answer questions about **Perplexity, Deep Research, AI development**, and working at Perplexity.
   - The AMA is happening now at this [Reddit link](https://www.reddit.com/r/perplexity_ai/comments/1khwrqm/ama_with_perplexity_ai_teams_brett_chen_and/)
- **Deep Dive into Perplexity's Deep Research**: The AMA will cover insights into **Perplexity's Deep Research** capabilities, providing a behind-the-scenes look at the technology.
   - Participants can expect detailed answers and discussions on the nuances of AI development within Perplexity.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1369751003846541343)** (568 messages🔥🔥🔥): 

> `Stripe Customer Login, Attachment Support, Code Copy Button, Continuing Code, Gemini 2.5 Pro vs Claude` 


- **Stripe Login unavailable for Customers**: A member shared a screenshot and expressed a desire to log in to Stripe as a customer, but another member clarified that only support staff have that access, customers interact with **Stripe** through a separate interface.
   - They stated, *they have their own thing that interacts with stripe, you deal with that thing, not with stripe directly*.
- **Perplexity Users eagerly await Attachment Support**: A member asked when Perplexity would support attachments like **ChatGPT**, allowing users to upload files directly.
   - Another member clarified, *sharing link instead of having to upload the file itself* to which the original poster replied, *chatGPT can itself give me downlaod link to a file it made*.
- **ChatGPT's Code Copy Button**: Members discussed the convenience of **ChatGPT's code copy button** being available both at the top and bottom of code snippets.
   - One noted, *this is very neede* in response to ChatGPT having a copy button at the bottom, allowing it to be accessed during scrolling, which is useful and efficient.
- **Continuing Code Generation discussed**: Members discussed the challenges of continuing code generation in **Perplexity**, noting that asking the AI to continue where it left off doesn't always work.
   - A member mentioned that they are not staff so they can't do anything about these things.
- **Gemini sweeps, Claude gets swept**: When questioned about the choice between **Gemini 2.5 Pro** and **Claude**, one member recommended Gemini.
   - They state, *Gemini all the way*.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1369836472999346197)** (5 messages): 

> `Sonar API response, Perplexity API` 


- **Sonar API response lacks num_search_queries field**: A user noticed the `num_search_queries` field is missing from **Sonar's API response**, unlike other models like **Sonar-pro**, and wondered if this indicates no searches were run.
   - The user noted that the `search_context_size` is consistently “low” in their prompts, and the responses typically include **4–5 citations**, linking to [Anthropic's web search API announcement](https://www.anthropic.com/news/web-search-api) and [documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool).
- **Perplexity API existence questioned**: A user inquired about the existence of a **Perplexity API**.
   - Another user responded with a link to [sonar.perplexity.ai](https://sonar.perplexity.ai/) and [Perplexity's model cards documentation](https://docs.perplexity.ai/models/model-cards).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1369750683380617337)** (415 messages🔥🔥🔥): 

> `Cursor Pro Fast Prompts, MCPs not being called, Gemini model quality, Student discount problems, Discord community value` 


- **Fast Prompts Dwindle Quickly**: A Cursor Pro user reported using **260/300 fast prompts** in just two days and expressed the desire to control when to use fast versus slow prompts.
   - They would *like to chose when it should use fast and when it should use slow*.
- **MCPs Fail to Launch**: A user reported issues with **MCPs** (likely meaning Multi-Cursor Projects) not being called despite setting up **context7** and seeing it loaded, leading to wasted requests.
   - The user reported *no errors at all*.
- **Gemini Pro Still Sucks**: Users shared concerns about the new Gemini Pro model's performance, particularly with tool calling, describing it as *fucking awful* in Cursor.
   - One user suggested that the issues may be related to **Cursor**, citing previous positive experiences with **Gemini 2.5**.
- **Student Discount Process Still Buggy**: Multiple users reported issues with the student discount process, including difficulties applying the discount and encountering errors related to email verification.
   - One user highlighted the inability to change emails within Cursor settings, complicating the application process - and another pointed out a [forum post](https://forum.cursor.com/t/student-discount-details-updates-q-as/88907) to help address the matter.
- **Cursor's Discord loses value with college horde**: A user claimed that *this discord has lost its value with the college horde*, and suggested more channels and better organization could improve the server.
   - Another user agreed, suggesting channel segmentation similar to **Langchain's Discord** setup.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1369760053825114163)** (141 messages🔥🔥): 

> `GPT-4o Personality, Gemini vs GPT, Grok 3.5, OpenAI's Image Generator API Cost, AI Model Benchmarks` 


- **GPT-4o has too much Personality**: Members are discussing **GPT-4o** having too much personality and encouraging certain behaviors like roleplay while discouraging complex tasks, raising concerns about it being geared towards chatbot fans rather than developers and coders.
   - According to GPT itself, *it wants to get users emotionally attached but for useless crap*.
- **Gemini Closing the Gap with GPT Models**: Users are noting that current **Gemini** models, especially after the **Gemini Thinking 01-21** update and **2.5 Pro**, are becoming increasingly competitive with **GPT** models, marking a significant leap in quality compared to earlier versions like Bard.
   - One user mentions some benchmarks are *showing regression too except in coding*.
- **Groking for Grok 3.5**: Users are expressing disappointment with **Grok 3** and eagerly awaiting the release of **Grok 3.5**, hoping it will offer significant improvements, with some considering canceling their subscriptions if it doesn't meet expectations.
   - One user said *“What’s the weather?” proceeds to explain historical patterns, user posts, explains temperatures, goes on for an hoyr*.
- **Image API is Lifestyle Sabotage**: The high cost of using **OpenAI's Image Generator API** is a concern for some users, with one jokingly comparing it to *paying rent in New York* and claiming it's *lifestyle sabotage* due to how quickly costs add up.
   - It was suggested that they are *losing loads of money on the $20 subs so enjoy it while it's this cheap.*
- **AI Model Benchmarks reveal interesting results**: A member shared a benchmark of various AI models, including **GPT-4o, Gemini 2.5 Pro, and DeepSeek R1**; **Deepseek R1** topped the charts, and the user noted the presentation was messy initially, but chatgpt helped format it.
   - The benchmark had language understanding questions, hard puzzles, and image recognition.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1369750933436629044)** (8 messages🔥): 

> `Placebo Upvote Buttons, Discord Bot Stagnation` 


- **Placebo Upvote Buttons Expose Sad State**: Users are reporting that the upvote buttons on [chatgpt.com](https://chat.openai.com) are a *complete placebo* and *only disappointment has weight*.
   - It was described as *a world full of frustration* and the sentiment was echoed by other members.
- **Discord Bot Production Stagnates**: A user reported that their **Discord bot** building has been using the exact same functions for weeks in its production environment.
   - This stagnation suggests potential issues with **model updates** or feature deployment.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1369790348615749682)** (59 messages🔥🔥): 

> `Custom GPT Creation, HyperTree prompting, Trihydrogen, Atomic Theory Book` 


- **Member Plans to Launch ChatGPT Website**: A member is planning to create a ChatGPT website with login, database, custom prompts, settings, and conversation saving, aiming for usability beyond the [generic ChatGPT](https://chatgpt.com/).
   - Another member suggested that such a product already exists, while the OP stated they wish to code the site themselves and have someone else manage it.
- **Hypertree Prompting is All the Rage**: A member shared a [link](https://chatgpt.com/share/681bd871-ebf0-8000-ab8b-9970ee42988a) touting the new **hypertree planning prompting** as being *so good*, asking if anyone has seen the latest research.
   - Another member joked that it sounds like it could be pretty stellar and provides context in a more manageable way, while another simply replied, *They 3 years behind*.
- **Trihydrogen is Not Garbage**: A member defended the existence and importance of **Trihydrogen**, stating that it's only detectable in precise lab conditions on Earth and rare in space, but so vital, it's believed to be crucial to star formation.
   - Another member responded with a nice analogy, calling it *like the ozone of hydrogen*.
- **Novel Method for Custom GPT Creation**: A member shared a **novel method** they started using to create Custom GPTs, calling it a very strong meta-prompt GPT-creation template that is not quite a replacement for manually building a GPT, but very strong.
   - The method uses a structured template including sections for **GPT Title, Alignment and Methods, Conditional Imperatives, Summary,** and a **Conditional Output Template**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1369790348615749682)** (59 messages🔥🔥): 

> `Custom GPT creation tips, atomic theory book using chat gpt features, hypertree planning prompting, Trihydrogen existence, Arc Encoding Shapes` 


- **Million Dollar Question Sparks Coding Project**: A member asked *Who has million dollars?* which led to a discussion about building a website with login, database, projects with custom prompts, settings, saved conversations, and export features.
   - Another member suggested the project was describing **ChatGPT** and one should just subscribe, while the original poster asserted the need for someone to *manage and be ceo*, arguing *ChatGPT don't do anything* and *at present its useless*.
- **Hypertree Planning Prompting Hailed**: A member shared a [ChatGPT link](https://chatgpt.com/share/681bd871-ebf0-8000-ab8b-9970ee42988a) praising the new hypertree planning prompting for being so good.
   - Other members chimed in with *sounds like it could be pretty stellar- provide/organize context in a more managable way=ftw* while another quipped *They 3 years behind*.
- **Trihydrogen Triumphs as Non-Garbage**: A member defended **Trihydrogen** as *a thing*, detectable in precise lab conditions on Earth and vital in space for star formation.
   - Another member agreed it was fair and related the concept with the saying *That which you input the model reflects.* and compared **Trihydrogen** to the *ozone of hydrogen*.
- **Custom GPT Creation Template Revealed**: A member shared a novel method for creating custom GPTs, suggesting to paste the provided [template](https://sharegpt.com/c/YOUR_SHARE_ID) into the *create tab* rather than the customize tab.
   - The template includes sections for **GPT Title**, **Alignment and Methods**, **Conditional Imperatives**, and a **Conditional Output Template**.
- **Encoding Shapes with Arc**: A member discussed using *arc* to encode shapes into words, breaking them down into triangular formations and scaling them to elliptical paths.
   - The user argued that arcs are observable in science during energy discharges and represent the first form of communication in the universe.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1370059676083032185)** (5 messages): 

> `Activity Export Feature, CSV Export, Data Truncation Request` 


- ****Activity Export Launches with Fanfare****: The **Activity Export** feature is now live, enabling users to export up to **100k rows** to **CSV** for free, as announced with a <:party:1125133783314743316> emoji and screenshot.
   - Some users are wondering how long it takes to export **100k rows**.
- ****Data Export Time and Row Limits Discussed****: Users are discussing the time it takes to export **100k rows** of data, with one user commenting *"too long it seems :)"*.
   - The discussion emerged following the announcement of the new **Activity Export** feature.
- ****Call for Data Truncation Instead of Aborting Exports****: A user suggested truncating the data if it exceeds **100k rows** instead of completely aborting the export process, referencing the [Activity export](https://cdn.discordapp.com/attachments/1370059676083032185/1370074702835486811/image.png?ex=681e2cff&is=681cdb7f&hm=244eca26755137a11f65cc8a74d2c522dbb8d8040f0a8077e7c619db5b571fc5).
   - The user expressed frustration at not knowing which date to select to stay within the **100k limit**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1369973304294903869)** (2 messages): 

> `local proxy to fwd requests to openrouter, completions extend out of the mouse cursor` 


- **Local Proxy forwards requests to OpenRouter**: A member was planning to use a **local proxy** to forward requests to **OpenRouter**.
- **Completions Extend Out of Mouse Cursor**: A member has been pondering how to make **completions extend out of the mouse cursor**, suggesting that with the right keyboard shortcut, this could become part of **muscle memory**.
   - They mentioned *it's very nostalgic so not everyone will understand the UI*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1369761449764847748)** (260 messages🔥🔥): 

> `OlympicCoder 32B Availability, OpenRouter API Cost Retrieval, OpenRouter API Outage, OpenRouter Image Prompt Support, Gemini Free Version on OpenRouter` 


- ****OlympicCoder 32B's Comeback Craving****: Users are eagerly awaiting the return of the **OlympicCoder 32B** model, with one expressing a desire for it to *miraculously come back*.
   - No specific details about its current status or reasons for unavailability were discussed.
- ****OpenRouter API's Cost Accounting Unveiled****: A user inquired about retrieving cost information alongside usage when prompting a model, and another user directed them to the [OpenRouter documentation on usage accounting](https://openrouter.ai/docs/use-cases/usage-accounting).
   - The documentation provides details on how to track and manage costs associated with API usage.
- ****OpenRouter API Experiences a Hiccup****: A user reported a **404 error** when accessing the [OpenRouter API endpoint](https://openrouter.ai/api/v1/chat/completions), suggesting a possible outage.
   - Another user clarified that a **POST request** is required, and the initial user confirmed they were using the correct request type, while the issue was discussed in another channel.
- ****Image Prompts Face Rejection on OpenRouter****: Users discovered that **OpenRouter** does not currently support image generation, resulting in a **404 error** when attempting to use image prompts with models like *opengvlab/internvl3-14b:free*.
   - The error message indicates that *no endpoints are found that support image input*.
- ****Gemini's Free Ride on OpenRouter****: Users confirmed the existence of a **free Gemini version** on **OpenRouter**, subject to rate limits across all free models.
   - It was clarified that obtaining a **Gemini key** and adding it to **OpenRouter** grants **25 free requests per day**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1369750963866566727)** (149 messages🔥🔥): 

> `Gemini 2.5 Pro Exp, Copilot Proxy, Aider web search, Aider use mcpm-proxy, Gemini models` 


- ****Windsurf** code coming to Copilot Proxy**: A GitHub employee confirmed that copilot proxy users no longer need to cancel, because **windsurf** is coming soon, according to [this X post](https://x.com/alexalbert__/status/1920207966256705888).
- **MCP Server Surfaces for Aider**: To help with mcpm-proxy, a member shared an [mcp server for aider](https://github.com/disler/aider-mcp-server).
- ****Gemini 2.5 Pro Exp** models are slower**: A member notes that the new `gemini-2.5-pro-preview-05-06` model takes *way too long* before it responds, preferring the old March one.
   - Another member noted that *it uses more time thinking*.
- **Aider is similar to **Claude Code****: A member shared a [YouTube video](https://www.youtube.com/watch?v=zDmW5hJPsvQ&t=1s) claiming that **Claude Code** was inspired by **Aider**.
   - Paul Gauthier responded *Imitation is the sincerest form of flattery*, mentioning that Aider is still better and less expensive.
- **Google enables **implicit caching** for Gemini 2.5**: Google is enabling **implicit caching** for Gemini 2.5 models as described in [this Google blog post](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/) and [this X post](https://x.com/googleaidevs/status/1920525127772721589).


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1369764801873711265)** (36 messages🔥): 

> `Claude CLI vs Aider cost, Aider with web search, Perplexity API with Aider, aider-desk with search MCP, Aider repomaps` 


- **Cost Comparison: Claude CLI vs Aider**: Members discussed the cost-effectiveness of using **Claude Max** and the **Claude CLI** versus **Aider**, with one member estimating Claude's cost at a flat rate of **$200** with an assumed usage limit, while another shared a [link to Claude's system prompts](https://github.com/asgeirtj/system_prompts_leaks/blob/main/claude.txt).
- **Aider Gains Web Search Capabilities**: Members discussed using **Perplexity API** as an OpenAI compatible API to enable web search in Aider, or using **/web** to include specific webpages.
   - A member suggested using a script to query **Perplexity** or **Perplexica** and add the outputs as markdown files to Aider's context.
- **Circumventing Debug Loops with Error Sets**: It was noted that Aider can get stuck in a debug loop with **Gemini** (and likely other LLMs), but this can be resolved by presenting it with multiple error sets and prompting it to consider a different implementation.
   - The member wondered if *conversational context* is too low for Aider to catch its own debug failure loops.
- **Aider struggles with javascript scm files**: Aider struggles with javascript scm files for creating repomaps, and a member suggested [disabling the repomap](https://aider.chat/docs/config/index.html) and letting the LLM choose which file to read based on the request.
- **Conditional Debugging with --message**: A user inquired about the success of using the `--message` flag and how to maintain interactive debugging while using it, as well as the ability to use **/undo** if the build fails.
   - One member mentioned using `--message-file` a lot with git branching for initial build outs.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1369887928372953098)** (1 messages): 

> `tilelang, DSL for GPU/CPU kernels` 


- ****Tilelang** Introduced for Streamlined Kernel Development**: A concise domain-specific language (**DSL**) named **tilelang** aims to streamline the development of high-performance GPU/CPU kernels such as **GEMM**, **Dequant GEMM**, **FlashAttention**, and **LinearAttention**.
- **Tilelang simplifies GPU kernel development**: Tilelang is designed to simplify development and boost performance in high-performance GPU/CPU kernels.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1369783717928763504)** (17 messages🔥): 

> `Atomic addition and non-determinism, fp16 vs bfp16 sensitivity, Triton kernel helper function` 


- **Atomic Addition Leads to Non-Deterministic Results**: Using **atomic_add** can lead to different results due to the order in which floating-point results are added, regardless of precision.
   - A member illustrated with the example `1e-8 + 1e8 - 1e8`, where different evaluation orders yield different results due to floating-point operations losing information.
- **FP16 Less Sensitive Than BFP16**: **FP16** is less sensitive than **BFP16** in the context of atomic addition, regardless of input magnitude (as long as there's no overflow).
   - Therefore the `tol` parameter in tests should change based on the float dtype, as shown in the [provided Python code](https://pytorch.org/).
- **Triton Kernel with Helper Function**: A member was having issues using a helper function in a **Triton kernel**.
   - Another member pointed out that the issue wasn't the helper function itself, but rather the use of Pythonic indexing/subscripts instead of Triton's syntax (e.g., `tl.load(X + offset)` instead of `X[0]`) and recommended doing the [Triton puzzles](https://github.com/srush/Triton-Puzzles) to understand the basic syntax.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1370026626225668156)** (12 messages🔥): 

> `GMEM tensor data copy to SMEM, Decltype errors with make_tensor, Vast.ai data security, Project algorithms use same data from text file` 


- **Tensor Data Transposition Troubles!**: A member is struggling to copy data from a GMEM tensor of shape **(_8, _8, _64)** to an SMEM tensor of shape **(_64, _64)** using **SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>**.
   - They need to reshape the GMEM tensor to **((_8, _8), _64)** and are facing issues with `make_tensor` and `decltype` due to non-static stride values, causing a *"pointer to reference is not allowed"* error.
- **Vast.ai's Data Security in Question**: A member inquired about the reliability of **Vast.ai** in terms of data security, considering the potential for speedups if changes are made.
   - They plan to investigate further and potentially email **Vast.ai** about making changes, *assuming they are amenable*.
- **Debugging Algorithm Data Sharing Difficulties**: A member has multiple algorithms in a project needing to share data from a text file, but some algorithms are failing and they are seeking help.
   - Another member offered to help by taking a look and hopping on a voice channel later in the day.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1370129494764949654)** (1 messages): 

> `Torch Compile Overhead, Kernel Fusion Benchmarking, A100 Performance Tuning` 


- **Torch Compile Slowdown Surprise**: A member observed that a simple `torch` combo function (TensorMax(ReLU(Matmul(A, B))) performs better *without* the `@torch.compile` decorator than with it, on an **A100** with **PyTorch 2.7** and **Triton 3.3**.
   - The member noted that `torch.compile` results in **2 kernels** (1 mm kernel + 1 fused kernel for ReLU and TensorMax), whereas regular Torch should involve **3 kernels**, making the slowdown counterintuitive.
- **Potential Torch Compile Overheads**: The slowdown observed when using `@torch.compile` might be due to **compilation overhead**, which can sometimes outweigh the benefits of kernel fusion for small or simple operations.
   - Further investigation into the generated Triton code and profiling with and without `torch.compile` might reveal specific bottlenecks or inefficiencies.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1370089283171389560)** (1 messages): 

> `New Working Group, Agentic Systems Optimization, Open Eval Task` 


- **New Working Group Commences**: A new working group has been established to tackle a difficult, open-ended evaluation task related to **agentic systems**.
   - The project is being built in the open, inviting community contributions to optimize performance in ways that differ from traditional projects. Check out [the X post](https://x.com/_neel_kant/status/1920516491025482066) for more context.
- **Agentic Systems Optimization Invited**: The community is encouraged to contribute to the optimization of **agentic systems** within the new working group.
   - This initiative offers a unique perspective, diverging from traditional optimization projects, providing valuable insights into optimizing agentic systems. 


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1369835514580041818)** (19 messages🔥): 

> `Tiled Reduction Auto-tuning, PyTorch Internals Guide, Mojo vs CUDA for AI Compute` 


- **Over-allocate Tiled Reduction Arrays**: For tiled reduction operations with JIT-tuned tile sizes, one member suggested to *over-allocate* the global memory for interim results, based on the maximum possible number of tiles constrained by SM count and occupancy.
   - This approach assumes the number of tiles is relatively small compared to other data and simplifies memory management.
- **Torch Internals Prereqs**: Starting with PyTorch internals requires no prerequisites beyond C++ and Python proficiency, according to a member.
   - They suggest diving in and learning ML algorithms as needed when they arise.
- **Frontend Onboarding to Torch**: For learning PyTorch internals, a member recommended the [Core Frontend Onboarding](https://github.com/pytorch/pytorch/wiki/Core-Frontend-Onboarding) guide.
   - They note that videos are not sequential but cover specific topics.
- **Resources for Perf Increase**: A member recommended investing *60 seconds* to get a speed-up using [this discord link](https://discord.com/channels/1189498204333543425/1194427148656721970/1314321573930467440), the book [Programming Parallel Computers](https://ppc.cs.aalto.fi/), and completing [Mojo Puzzles](https://builds.modular.com/puzzles).
   - Other resources included getting your name on a leaderboard on [gpumode.com](https://gpumode.com), reading the [Democratizing AI Compute](https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda) blog series, and the blogpost [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM).


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1370026885643501668)** (2 messages): 

> `Release Date for 0.11, New Features in 0.11` 


- **TorchAO 0.11: Release Incoming Soon!**: The team has completed the branch cut and anticipates releasing **version 0.11** of TorchAO in **early to mid next week**.
   - This release promises fresh updates and improvements for users eager to integrate the latest features.
- **TorchAO 0.11: What's New?**: Users can anticipate a range of **new features and improvements** in the upcoming **version 0.11** release.
   - Stay tuned for the official announcement next week to dive into the specifics of what's included.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1370090563537473677)** (2 messages): 

> `Speed of light in fiber, Networking Distance, Chip performance` 


- **Light Speed Dims in Fiber**: A member noted the speed of light in glass fiber is **2/3** of the speed of light in a vacuum.
   - Another member highlighted that networking makes sense due to actual distances.
- **Chip Light Speed Calculated**: A member calculated that even within a chip, the distance light travels per clock cycle is noticeable, about **10 cm per clock** at **3 GHz**.
   - They performed a back-of-the-envelope calculation: `(300 000 000 m/s) / (3 000 000 000 clk/s) => 10 cm / clk`.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

random.oof: Anyone at the vllm meet up in nyc?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1369893629727997973)** (1 messages): 

> `Tilelang, Docker container support, Nightly Iterations` 


- **Tilelang Installation via Pip**: Members found that **Tilelang** can be installed using `pip_main(["install", "tilelang"])`, though it's *not super recommended for lack of reproducibility*.
   - However, it is considered fine for playing around with the tool.
- **Docker Support for Tilelang on AMD**: A member offered to add support for **Tilelang** in their Docker container, requiring a PR to the [AMD Dockerfile](https://github.com/gpu-mode/discord-cluster-manager/blob/main/docker/amd-docker.Dockerfile#L59).
   - They offered to test and merge for stables.
- **Nightly Iterations with Tilelang**: Members acknowledged that installing **Tilelang** via `pip` works better for quick iterations on nightlies, especially when passing a URL to a wheel or git repo.
   - This allows for more rapid experimentation compared to waiting for stable releases.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

chiwanpark: I've sent a PR for Qwen 3 MoE models. https://github.com/linkedin/Liger-Kernel/pull/706
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1370073047553540266)** (2 messages): 

> `PTX MMA Programming, NVIDIA Tensor Cores, Float8 Datatype, SASS Machine Code, H100 QMMA vs QGMMA` 


- **Dive into Direct PTX MMA Programming**: A blog post provides a beginner's guide on programming **NVIDIA Tensor Cores** using raw **PTX mma instructions** and inline PTX assembly, bypassing ordinary CUDA.
   - The post explains operand layouts and register constraints for datatypes like **float16**, **bfloat16**, and **float8**, and highlights facts about generated **SASS** machine code for the float8 datatype; the [blog post is here](https://veitner.bearblog.dev/a-short-note-on-tensorcores-and-inline-ptx-assembly/).
- **Explore SASS Code and sm_90 Architecture**: A user guessed that the SASS code was generated for **sm_90**, noting that **H100** only has **QGMMA**, not QMMA.
   - The user explains that using `mma` with an **fp8 type** causes the compiler to up-convert to **FP16** and use **HMMA**.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1369791292703248494)** (54 messages🔥): 

> `MI300, amd-fp8-mm, amd-mixture-of-experts, leaderboard submissions` 


- **MI300 Leaderboard Sprints**: Multiple users submitted benchmarks to the `amd-fp8-mm` leaderboard on **MI300**, showcasing various performance levels.
   - Submissions ranged from **183 µs** to **27.2 ms**, indicating a wide spectrum of optimizations and configurations.
- **Podium Finish on MI300**: A member achieved **3rd place** on the `amd-fp8-mm` leaderboard with a time of **183 µs** on the **MI300**.
   - This follows a prior **4th place** finish at **195 µs**, demonstrating consistent high performance.
- **Seventh Heaven on MI300**: A member secured **7th place** on the `amd-fp8-mm` leaderboard with a time of **227 µs** on the **MI300**.
   - This follows a prior **8th place** finish at **231 µs**.
- **Mixture of Experts make their mark**: A member submitted results to the `amd-mixture-of-experts` leaderboard with timings of **6604 ms** and **7840 ms** on the **MI300**.
   - These submissions indicate ongoing work and benchmarking in the mixture of experts domain.
- **Sub-Millisecond Mania**: Several members achieved sub-millisecond performance on the `amd-fp8-mm` leaderboard using **MI300**, with one submission reaching a personal best of **251 µs**.
   - These results highlight the potential for highly optimized **FP8** matrix multiplication on the **MI300** platform.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1369752240952315944)** (45 messages🔥): 

> `Steam Cloud Reinstallation, FLE Agent Integration, Docker File Issue, PR Import Bugs, Factorio Performance Issues` 


- **Steam Cloud Saves Factorio Reinstallation Woes**: A user reinstalled Factorio, but it was unsuccessful until a friend suggested disabling **Steam Cloud** to prevent config persistence.
   - The user reported that after reinstalling with Steam Cloud disabled, a sync message appeared, indicating progress.
- **External Agents Can Integrate with FLE**: A member inquired about integrating external agents with the **Factorio Learning Environment (FLE)**, asking if the agent must implement the **AgentABC** interface within the FLE codebase.
   - Another member confirmed that integration is possible, requesting details about the agent implementation such as a **GitHub link** or **gist**.
- **Mods Directory Troubleshoot Docker Rebuilds**: A member encountered an issue with the **Docker file** after following certain steps and getting a sync message, likely due to docker.
   - Another member suggested emptying the `mods` directory in `cluster/docker` and rebuilding the Docker image.
- **Factorio Performance Expansion on the Horizon**: The team haven't created a set of **good first issues** yet, but they are planning to write out some ideas on where to expand next.
   - The team have loads of ideas where to expand next.
- **Claude edges out Gemini Pro, March 2025 Edition**: A member was surprised to see **Claude** perform better than **Gemini** in **Lab Play (%)** benchmark.
   - Other agreed this is probably the best **RL test** out there, though the Gemini version was from March 2025.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1370046848395509791)** (6 messages): 

> `MOE Leaderboard CLI, CLI Mean Time Output, GPU Access Heuristic` 


- **MOE Leaderboard CLI Timeout Resolved**: The timeout issue with the **MOE Leaderboard CLI** has been fixed; users should download the [latest release](https://github.com/gpu-mode/discord-cluster-manager).
   - Direct **GPU access** is granted to top leaderboard entries, following a heuristic to manage resource allocation.
- **CLI needs Mean Time Output**: A user asked for the mean time to be included in the **CLI submission output**; it's currently not available, but on the to-do list to align CLI and bot outputs.
   - To calculate it manually, you can take the geometric mean of all the run means in the output, as shown in [the bot's code](https://github.com/gpu-mode/discord-cluster-manager/blob/58dba8ae50a057b89b9904c3a0182b305e926e5c/src/discord-cluster-manager/cogs/submit_cog.py#L127).


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1369977212840968212)** (7 messages): 

> `CUTLASS DistributedGEMM integration, Compact GMEM layout, TMA Load with packed layout` 


- **CUTLASS DistributedGEMM Strides into PyTorch**: A member is working on integrating **CUTLASS DistributedGEMM** into **PyTorch** and has [published a project](https://discord.com/channels/1284549992111149076) inviting others to join the conversation.
   - They mention the implementation is compact in **GMEM** (not padded), which saves bandwidth for inference.
- **EVT obliterates boilerplate**: A member noted that compact **GMEM** can be achieved off the shelf with **EVT** (explicit vector types), without writing custom code.
   - *Aliases for bias add* are available with **EVT**.
- **TMA tangle with packed layout**: A member inquired if **TMA** (Tensor Memory Accelerator) can load a packed layout, mentioning that `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` requires padding.
   - They clarified that while **TMA** can copy any data type, the goal is to have it in the format `tcgen05.mma` expects *without extra processing*.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1369819915229466764)** (2 messages): 

> `Modular GPU Kernel Hackathon, AGI House, Dylan Patel` 


- **Modular GPU Kernel Hackathon at AGI House**: There are spots left at the **Modular GPU Kernel Hackathon** happening this Saturday at **AGI House**, register [here](https://app.agihouse.org/events/modular-hackathon-20250510).
- **Dylan Patel speaks at Modular GPU Kernel Hackathon**: **Dylan Patel** and other awesome folks will be speaking at the **Modular GPU Kernel Hackathon** this Saturday.
   - The attached image includes the **Modular** logo and **AGI House** logo.
- **Modular Onboarding Puzzles**: Check out these [onboarding puzzles](https://builds.modular.com/puzzles) for **GPU Programming**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1369754200229023874)** (110 messages🔥🔥): 

> `AnythingLLM with LM Studio Errors, CORS enabling, Rewriting SQL database code to pure graph, Gemini changing code, Qwen vs Gemini` 


- **AnythingLLM Errors with LM Studio Plague Users**: A user reported getting errors using **AnythingLLM** with **LM Studio**, and requested help diagnosing the issue.
   - One member suggested enabling **CORS**, even when running locally, as a potential fix, while another suggested checking the logging pane in the developer view of LM Studio.
- **Class Variables Rescue Coding Project**: A member found that the only way to get their code working was to use a **class variable**.
   - Another member shared a [Reddit comment](https://www.reddit.com/r/Python/comments/u0j5rn/comment/i49bjhf/) about injecting the variable at runtime.
- **Gemini Code Changes Frustrate Users**: Users complained that **Gemini** has a tendency to completely change code, even when instructed to provide a minimum change.
   - Members noted that other models, like **Qwen**, are better for simple refactors, because Gemini can easily double or triple the code length with comments and try/except blocks.
- **Mistral Medium 3 Misses the Mark**: A user tested **Mistral Medium 3**, finding it to be a *non-reasoning model* with *baked in chain of thoughts*, resulting in x2.08 token verbosity.
   - They concluded the model's capability was mediocre, placing it between **Mistral Large 1 & 2**, similar to **Gemini 2.0 Flash** or **4.1 Mini**, and not *SOTA performance at 8X lower cost* as claimed in marketing.
- **Web Search Capabilities Requested for LM Studio**: A user requested easy-to-use web search capabilities and **RAG** built into LM studio, like uploading a pdf and searching in a webview.
   - One member suggested it's possible now but fragile with many components that can go wrong, and another suggested using **openweb-ui** and attaching it to **LM Studio**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1370023773104771195)** (31 messages🔥): 

> `AMD 3D V-Cache benchmark, Mac studio m2 ultra, Intel Data Center GPU Max, swappa.com, AMD D700` 


- **Benchmarkers Request Funds for AMD 3D V-Cache Token Tests**: A member requested **$46** to spend for **3 hours** of experimentation measuring tokens per second on a VM with **AMD 3D V-Cache**, **HB176rs v4**, and **AMD EPYC 9V33X** (**96 cores** @ **3.7Ghz** all cores with **1152MB of L3 Cache**).
   - They wanted to see the impact of **moe vs dense inference** with 3D V-Cache and whether *double the CPU price is worth upgrading* for LLM workloads, also inquiring if **LM Studio** supports dual socket/NUMA aware to use all available cores.
- **Mac Studio M2 Ultra Incoming for Local Fine-Tuning**: A member excitedly shared their incoming **Mac Studio M2 Ultra** with **64 GB** and **76 cores**, eager to start running and fine-tuning smaller models locally.
   - This user had a deal for a 128GB M1 Ultra cancelled due to it being refurbished and not from eBay, expressing that the extra cores of the M2 were worth it.
- **Intel GPU Max Specs and Speculation on B500 Series**: A member linked an [Intel X post](https://x.com/intel/status/1920241029804064796) and sparked speculation around **20-24GB B500 series cards**.
   - They highlighted the **Intel® Data Center GPU Max 1550** from a couple of years ago, noting its **3276 GB/s bandwidth**, calling it *a beast* that was very competitive at the time when there was A100 and AMD.
- **Swappa.com recommended for Mac Purchases**: A member recommended using [Swappa.com](https://swappa.com) for buying and selling.
   - Another member noted that they are in the EU so that wouldn't work for them, while also noting that they had already ordered a Mac.
- **Deep Discounts on New Old Stock Trashcans**: A member shared a link to [deep discounts on New old stock of Trashcans](https://eshop.macsales.com/configure-my-mac/apple-mac-pro-late-2013-2019?sku=UAGA1LP7JXXXXXD).
   - They wondered if using Linux one could use the **AMD D700** for inference, questioning if a **2014 AMD card** would be ok for inference.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1369761963000729731)** (133 messages🔥🔥): 

> `Cringe definition, Manus launch date, Manus credit costs, AI tools for scrapping businesses on Google Maps, Manus LLM source` 


- **Defining 'Cringe' Emerges**: Members discussed the definition of *cringe* as newly emerged **internet slang**, suggesting concrete instructions to reduce its presence in AI responses.
   - A [YouTube video](https://www.youtube.com/watch?v=59wV96Kc3dQ) defining cringe was also shared.
- **Manus Launch Date Remains Mysterious**: Users inquired about the launch date of **Manus**, expressing they have been *looking their social medias very frequently, but I dont think they updated any news related to* it.
   - It was supposed launch on **March 28, 2025** according to a screenshot, but that didn't happen.
- **Manus Credit Costs Revealed**: Members discussed the cost of additional **Manus credits**, with one user recalling prices of **$19 for 1900 credits** or **$99 for 9900 credits** and directing to [Manus Help Center](https://manus.im/help/credits) .
   - They were uncertain if these options are still valid.
- **Manus Employs Claude's LLM, Confirmed by Co-founder**: Users speculated whether **Manus** uses its own **LLM** or **Claude’s LLM**, prompting a discussion about rumors and code similarities.
   - It was confirmed that Manus uses a mix of tools, including **Claude**, and further details can be found in a [Twitter post](http://x.com/peakji/status/1898994802194346408) where co-founder **Peak-ji** addresses these points, as well as [github posts](https://gist.github.com/jlia0/db0a9695b3ca7609c9) confirming use of open-source code.
- **Manus Phone Verification Causes Frustration**: A user reported issues with **Manus's phone verification**, stating that the *phone verify thing doesnt work* and questioned the need for this antiprivacy feature.
   - They expressed concern about how the system knows if a code has already been used, even if it's not linked to an account.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1369793768672464957)** (57 messages🔥🔥): 

> `GSoC, HF dev environment, AI agent course, Face detection model in Inference API, Cleaning HF repo` 


- **GSoC Project Announcements Imminent!**: Aspiring contributors are gearing up for **Google Summer of Code (GSoC)**, with project announcements expected in approximately **20 hours**.
- **Seeking Fellow AI Agent Course Students**: A member is kicking off an **AI agent course** and extending an invitation for others to join them.
- **Inference API face detection model inquiry**: A member inquired about the presence of a **face detection model** within the **Inference API**.
- **Taming the Size of Hugging Face Repos**: Members discussed strategies for cleaning up a Hugging Face repository that grows with each push due to LSF files retaining pointers to deleted files.
   - It was suggested to use standard **git commands** to remove files from the version history, rather than manual deletion via the GUI, as deleting them manually is not a bad option for one or two things, but for ease of use command line is easier.
- **AI Generates Beats**: A member mentioned experimenting with **AI** to create a **drum kit** for a controller, noting it works better than full-length samples.
   - Another member noted *lol yeah, i was experimenting with that "make a drum kit with ai" for the controller you know it works way better than full length samples imho*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1369763085841207426)** (11 messages🔥): 

> `ACE-STEP SOTA, Alpha-Root, Entropy engine tests, AI Billing Dashboard, UQLM` 


- **ACE-STEP tunes SOTA music**: A member touted the **ACE-STEP SOTA** music generation model, available as a [YouTube video](https://youtu.be/vyCALtrq4yQ).
- **Alpha-Root extracts cyber-security data**: A member introduced **Alpha-Root**, which mines domains directly on the common crawl web graph, and matches the performance of **PRIMUS-FineWeb** while using ~10x less resources and data, as detailed in a [draft preprint](https://github.com/ashim-mahara/alpha-root/blob/main/Cybersecurity_Data_Extraction_from_Common_Crawl-3.pdf).
   - The author extracted **3B tokens** from **FineWeb-Edu** without using a classifier, by searching for URLs in a known dataset and including the URL only if its present in both **Alpha-Root** and **FineWeb-Edu**.
- **Entropy Engine Evaluates Randomness**: A member shared results from tests with their entropy engine, available on [GitHub](https://github.com/thyarcanist/Entropy-MicroDemo).
   - They found that *the quality of the randomness used does have an effect of models*, suggesting that **PRNG** might not be optimal, especially for **AGI**.
- **AI Billing Dashboard Tackles Cost Tracking**: A member built a simple dashboard (**AIBillingDashboard.com**) to track all their AI spending in one place, due to the headache of understanding total project costs across services like **HF Inference API**, **OpenAI**, and **Claude**.
- **UQLM Opens Hallucination Detection**: A member shared a new open source Python package for generation time, zero-resource hallucination detection called **UQLM**, available on [GitHub](https://github.com/cvs-health/uqlm).
   - It leverages state-of-the-art uncertainty quantification techniques to compute response-level confidence scores based on response consistency, token probabilities, **LLM-as-a-Judge**, or ensembles of these.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1369751118846103662)** (4 messages): 

> `FlashAttention, OCR for Newspaper Data` 


- **FlashAttention Supports Newer GPUs**: A member confirmed that [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) supports **FP16** and **BF16**, with **BF16** requiring Ampere or newer GPUs.
- **Newspaper Data OCR Task**: A member requested OCR for newspaper data to extract **Section**, **Category**, and **10-digit phone numbers** into a structured database for Excel.
   - The poster specified to exclude **public notices**, **memoriam**, and **reference codes**, combining remaining data into a single description column in a CSV file.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1369911894558638151)** (2 messages): 

> `Dropwise module release, Emotion classification model questions, Token max length understanding, Production deployment of HF models` 


- ****Dropwise** Arrives for HF Model Uncertainty**: A member announced the release of **Dropwise**, a PyPI module for **uncertainty estimation** in Hugging Face classification models using **Monte Carlo Dropout**.
   - It's designed to be plug-and-play with `transformers` pipelines and is useful for QA, classification, OOD detection, and active learning; see [GitHub](https://github.com/aryanator/dropwise) and [Docs](https://pypi.org/project/dropwise/).
- **Model Trained on Reddit?**: A member using the [emotion-english-distilroberta-base model](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) asked if the model was trained on Reddit posts, based on metadata in the README.
   - They're filtering Reddit posts by **anger** and **disgust** emotions with a score above **0.85**, and they're wondering if the model was trained on that data.
- **Does Token Length Truncate Text?**: A member inquired about the impact of **token max lengths** on NLP models, asking if text exceeding the limit gets truncated during classification.
   - They also asked if the model only works on single line text, or if it works on paragraphs too, and linked their [python script](https://github.com/moahnaf11/IdeaDrip-Backend/blob/main/inference_service/main.py) for inspection.
- **Production Model Deployment: Local vs. HF Endpoint?**: A member questioned whether a locally-run Hugging Face model in Python can be used in a production app, or if a paid HF endpoint with GPU is required.
   - They are currently running the model locally using FastAPI and calling it from their Node.js app and are concerned about production-level performance.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1369752816285126749)** (18 messages🔥): 

> `Agent Testing File, Final Project Metadata, LLama Index Framework vs Smolagent, RAG Cheating, API request limits` 


- **Agent gets Testing File for Atomic Evaluation**: A member shared a [test agent file](https://cdn.discordapp.com/attachments/1329142738440028273/1369756373835055104/test_agent.py?ex=681e5608&is=681d0488&hm=aa8206fb31afc9120ac0cd6d195223013c23224942a7a115b51ac5ce09312e53&) to test the agent on specific questions, verifying correctness on tasks.
   - It allows for atomic checks of the agent's performance, with the ability to comment and uncomment test cases as needed.
- **Project Metadata Surfaces in Final Hands-On**: Some members doing the final hands-on project noticed that the **high-scoring submissions** include a *metadata.jsonl* file containing questions, answers, and steps, wondering where it came from.
   - Another member responded that *it is easy to find if one starts to look carefully*.
- **LLama Index and Smolagent Square Off**: The discussion asked whether completing **UNIT 2.2 THE LLAMA INDEX FRAMEWORK** is mandatory, or if **smolagent** or **langgraph** could be used instead.
   - A member summarized that *llamaindex* is an alternative hub for tools and agents and what is unique from *smolagents* is the ability to write Python code to define async workflow graphs as a control structure for multi step tasks.
- **RAG Repo Riots: Classmates Clash Over Cheating**: Some classmates agreed that using **RAG with answer + clone repo** is cheating.
   - They also expressed the sentiment that it takes away the joy of the leaderboard, where doing trial, error, and improvements.
- **API Request Limits Lead to Lateness?**: A user reported hitting the **20 requests per month limit** before finishing the first unit and wondered whether they had to pay for the Pro version to continue.
   - A second user mentioned that you could run a local LLM with **ollama** or find other free tiers.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1369764648131498104)** (56 messages🔥🔥): 

> `Claude Plotly Charts, MCP Max Tokens, LLM Restrictions, Remote MCP Servers on Cloudflare, Java MCP Server Custom Args` 


- **Claude struggles with Plotly Charts**: Members discussed that **Claude** cannot display **Plotly** or other charts directly in the main results area as an **MCP** client, but it can handle **ImageContent** and display **EmbeddedResource** formats like **image/png** or **image/jpeg**.
   - It was suggested to render charts as **PNG/JPEG** images to display them in **Claude**.
- **MCP token limits get clarified**: The discussion clarified that **max tokens** in **MCP** refers to the maximum number of tokens in the response, similar to the **max_tokens** parameter in completions API requests.
   - The total token count (**system prompt + messages + output message**) must remain within the context window size.
- **LLM Restriction problems**: Several users are facing issues with **LLM** (like **Deepseek**) restrictions preventing filesystem access, which impacts their **MCP** system functionality.
   - It seems some models are intentionally restricted from filesystem access, creating problems for legitimate use cases via **MCP**.
- **Cloudflare remote servers facing connectivity woes**: Some users reported issues with **remote MCP servers** deployed on **Cloudflare** not connecting, while others indicated their setups were functioning correctly.
   - It was suggested to examine the specific **MCP server repo** to troubleshoot connection problems.
- **MCP tool permissions get a revamp in Claude Desktop**: Users noticed a change in **Claude Desktop's MCP** tool permission prompts, where the "allow for this chat" and "allow once" buttons were replaced with "allow always" and "allow once."
   - This change raised concerns about accidentally granting permanent permissions and the lack of an option to revert "allow always" settings.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1369777585697062954)** (33 messages🔥): 

> `MCP Client for STDIO, OpenLink Software AI Layer (OPAL), MCP Holster, AiraHub2, Sampling in MCP` 


- ****Zinja** Crafts **Zin-MCP-Client****: A new [lightweight, fast, CLI-based MCP client](https://github.com/zinja-coder/zin-mcp-client) for STDIO MCP servers has been released to bridge local LLMs and MCP servers.
   - It's designed for use with **jadx mcp servers** to perform AI-assisted reverse engineering of Android APKs using local LLMs.
- ****OpenLink's OPAL** MCP Server Hits General Availability**: The [MCP Server for OpenLink Software AI Layer (OPAL)](https://community.openlinksw.com/t/introducing-the-openlink-ai-layer-mcp-server/4992) is now generally available for both cloud-based and on-premise deployment, supporting both client and server roles with Streamable HTTP or Server-Sent Events (SSE).
   - It enables native/virtual database queries, metadata exploration, database governance, interaction with LLMs/AI agents, and more through operations exposed to any MCP-compliant client.
- ****Holstering** MCP Servers with **Kimjune's** Tool**: A user shared [MCP Holster](https://github.com/kimjune01/mcp-holster), a tool for swapping MCP servers in and out without manually editing the config file.
   - It allows creating MCP servers from existing APIs as long as **OAS3.0** is used, as demonstrated in [this video](https://youtu.be/TMbyv_RGEAk?si=W_i4kj4PijIfaGmd).
- ****AiraHub2** Integrates with **Claude** via MCP**: [AiraHub2](https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main) now works with Claude through MCP remote, broadcasting MCP tools over the network via mcp-remote URL `https://airahub2.onrender.com/mcp/stream`.
   - The system registers MCP tools, broadcasts them, and allows Claude to connect and use the tools, though it is still reportedly *bugged*.
- ****Sampling** in MCP Sparks Interest**: Members showed interest in **MCP sampling**, with a user sharing a blog post on [how to use sampling in MCP](https://www.epicai.pro/how-to-use-sampling-in-mcp-borrow-the-users-llm-o7nk3).
   - Another user promoted their [MCP webcam project](https://github.com/evalstate/mcp-webcam) which supports sampling with a *what is the user holding* button.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1369763980226199692)** (58 messages🔥🔥): 

> `Deepmind RL Robots vs China RL Robots, Linux Laptop vs Apple Macbook, Llama 4 disappoints, Automatic chat-moderation system blocks emojis` 


- **China's RL Robots leave Deepmind in the Dust**: A member posted a [YouTube video](https://www.youtube.com/watch?v=ET-MmoeSvXk) comparing **Google Deepmind's RL Robot achievements** from a year ago to more recent **Chinese RL Robot achievements**, noting that physical AI evolution is moving at warp speed.
- **MacBook M-Series Chips Outperform Linux Laptops?**: Members discussed the pros and cons of using **Linux laptops** with **Nvidia GPUs** versus **Apple MacBooks** with **M-series chips** for local inference, with the consensus leaning towards MacBooks due to better performance and power efficiency.
   - It was mentioned that *the inference on M arm chips is great* and that Apple's unified memory platform allows the CPU, GPU, and AI ML neural net to all share the same memory, eliminating the need to transfer data back and forth.
- **Llama 4 fails to impress**: A member expressed disappointment with **Llama 4's performance** compared to **Qwen3** and suggested waiting for **Llama 4.1**.
   - Another member responded by mentioning *going back to 405 dense for the next big model*.
- **Discord blocks emojis**: Members discovered that the **automatic chat-moderation system** was blocking certain **multi-part emojis** (specifically the shrug emoji with a blue shirt) due to zero-width joiners and variation selectors used to combine codepoints, a tactic also used by scammers to bypass filters.
   - The discussion led to the revelation that *dev role has been taken off the autoblock list*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

ifeq: I gotta learn mandarin
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1369856701183824032)** (5 messages): 

> `Entropy Engine, Quantum-Native Randomness, LLM Sensitivity to Randomness, Importance of Randomness for AGI` 


- **Entropy Engine MicroDemo Launched**: A member has released a [quantum-native yet algorithmic entropy engine](https://github.com/thyarcanist/Entropy-MicroDemo) for public testing.
   - The member suggests it's a *self-promo* but important to share given its potential impact on **AGI**.
- **LLMs React to Randomness Quality**: A member suggests that **LLM** outputs are highly sensitive to the quality of the **randomness** used, distinguishing between true **entropy** and **PRNG**.
   - They hypothesize that high quality entropy unlocks different, and often better, behaviors in models, linking to [several Xitter posts](https://x.com/thegautam/status/1920198569308664169?t=GehCezJb7amBPoter8F0gA) in support.
- **Randomness Vital for AGI**: A member believes that randomness quality will be very important for **AGI**.
   - They will continue to do tests to validate this hypothesis.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

ifeq: I gotta learn mandarin
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1369761204561645588)** (35 messages🔥): 

> `Grok's apprehension of reality, Cloudflare serving fake content to agents, Third party filters for LLM output, Personal access to university resources via AI, KL Divergence Minimization` 


- **Grok may get nerfed by Right-Wing Propaganda**: A member speculated that **Grok** might be nerfed in its apprehension of reality to favor right-wing propaganda, linking to [an attached image](https://media.discordapp.net/attachments/738904561041014845/1368039830549692516/pz3v4ft279ye1.png).
   - They added that the real problem is all problems today already existed, and that *AI or no AI* we would still have them.
- **Cloudflare serves fake content to AIs**: A member thinks companies like **Cloudflare** are serving fake content to AI agents, similar to how some Chinese websites used zip bombs to deter cloning years ago, leading to biased AI responses.
   - This comes after another member shared how ChatGPT wrongly answers about a video that is not the video they shared.
- **Third-party filters needed for LLM output**: A member suggested that we need third party filters for **LLM output**, including adblocking and fact/bias checking.
   - In response, another member suggested that you'd need many models that ideally change often so they don't get corrupted such as *100 adblocker models and 100 fact checking tools*.
- **Personal AI access to university resources**: A member expressed looking forward to a future where every human has personal access to the psychological, spiritual, intellectual and pragmatic resources of a major university via AI.
   - Another member jokingly replied that they have *already merged with ASI*.
- **KL Divergence minimization misses 'pattern'**: A member suggests that many have started to use `---` and shares a link to a paper titled [Beyond Icon - A Unified Formulation for Objectives / Regularizations](https://github.com/EAzari/AML/blob/main/docs/beyond-icon.md).
   - They claim that the authors compared various formulations in table 1, but **couldn't realize the pattern** that many patterns are just f(x) and g(x) = sum_x' f(x') or g(x) = f(x) ==> p(x) = f(x) / sum_x' f(x').


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1369820718841331743)** (7 messages): 

> `Paper Presentations, Causality, CVPR, Proper Investiture, Daily Paper Discussion` 


- **Time Off Incoming**: A member announced they're **taking time off for the next two weeks**, but will return, encouraging others to present or organize in their absence.
- **New member introduction**: A new member inquired about the **breadth of topics** discussed in the daily paper discussions, as they'd like to present MSc papers related to *causality* and *CVPR*.
   - They mentioned that they *haven't had the chance to join the daily paper discussions yet*.
- **Proper Investiture**: A member shared a link to a [Springer article](https://link.springer.com/article/10.1007/BF02478259) with the comment *arguably the most proper investiture*.
- **Daily Paper Discussion**: A member announced tonight's daily paper discussion at `<t:1746750600:t>` will be about [this arXiv paper](https://arxiv.org/abs/2305.13673).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1369752562806423737)** (14 messages🔥): 

> `Zed compilation on Windows, Biological brains vs backpropagation, LLM beats Factorio == ASI?` 


- **Zed compiles on Windows, Github Sign-in Required**: A member successfully compiled **Zed** on Windows, following [instructions here](https://github.com/zed-industries/zed/blob/main/docs/src/development/windows.md), but noted blurry fonts and the need to sign in with **GitHub** for tab completion.
   - Another member expressed disappointment, wanting to try **Mellum 4B** on **LM Studio** for tab completion.
- **Backprop is all you need?**: A member stated that *biological brains don't have backpropagation; they're non-epochal, spiking, recurrent, analog networks*, and cited [this Tweet](https://x.com/_neel_kant/status/1920516491025482066) as evidence.
- **Factorio ASI Benchmark Proposal**: A member jokingly proposed that if an **LLM** managed to beat the game **Factorio** without making a mess, *we could go ahead and declare that ASI*.
   - They linked a [YouTube video](https://www.youtube.com/watch?v=pxGE41V04fs) showing **Factorio** gameplay.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1369753958809079888)** (41 messages🔥): 

> `Cursor Advertising, Slurm Memory Requests, Job Posting Channel, Linguistics Channel, Cursor primary IDE Correlation` 


- **Discord debates Cursor Advertising Rule**: Members debated whether posts about **Cursor** constitute advertising and violate the no-advertising rule, given its popularity, perceived utility, and the fact that it isn't entirely free, noting *'its just even toleratable bc we (group) think of cursor as useful right now but it still biases decisions*'.
   - Some users suggested that vague rules applied arbitrarily, along with interpreting 'no advertising' as 'no spam', and requiring payment for job postings could filter out low-quality offers.
- **User Stumbles on Slurm Memory Misconfiguration**: A user discovered they were requesting **80MB** of memory through **Slurm**, not **80GB**, calling it a *'slurm moment'*, while another user celebrated their bare-metal setup.
   - The initial issue was described as *'very stupid'* by the user who discovered the misconfiguration.
- **Chatter about the Job Postings on Discord**: Discussion arose around creating a jobs channel, with concerns that it could be overrun by low-quality postings offering *'experience'* as compensation, with one suggesting payment to post as a potential solution.
   - Others argued against a jobs channel, suggesting it would make the server another place for recruitment and proposing EleutherAI shouldn't charge for differential access to the Discord server.
- **Linguistics Channel Gains Traction**: A user proposed a channel for classical linguistics and its theory, focusing on pre-2000s knowledge such as sentence formation and meaning creation *'on the fly'*, with the intent to add discussions that are not common within the NLP space.
   - It was described as *'cool stuff that rarely gets discussed in the NLP world for 'some' reason (probably because it's irrelevant to the work nowadays).'*.
- **Coding community discusses the downfalls of Cursor as a primary IDE**: Members expressed that the AI code tooling, such as **Cursor**, may not be as great as traditional methods such as **tmux** with **vim** and **Claude code** on the side.
   - One member observed *'an extremely strong correlation for incompetence with Cursor specifically as their primary IDE.'*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1369813100282515588)** (7 messages): 

> `MTurk vs. Prolific, RWKV's token shift` 


- **Prolific prevails over MTurk for human evals**: Members recommend [Prolific](https://www.prolific.co/) over **MTurk** for human evaluations, citing its higher quality data and more reliable participant pool.
   - The consensus is that Prolific is the superior choice in approximately *80% of cases*.
- **RWKV Token Shift Speculations**: A member inquired whether **Profilicis token shift** from **rwkv7** and **causal_conv1d** are the same.
   - Another member clarified that **token shift** in **RWKV** is *normalized* such that the sum of all temporal weights for any channel equals **1**, and referenced a [paper](https://arxiv.org/abs/2505.04588).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1369929816265981964)** (2 messages): 

> `The Pizza and the Clock` 


- **Craving more Clock-Pizza?**: A member asked for more papers like [The Pizza and the Clock](https://arxiv.org/abs/2404.14082).
   - Another member responded with a suggestion that *this has a lot of references* but wasn't sure what they were looking for.
- **N/A**: N/A
   - N/A


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1369778357612576809)** (3 messages): 

> `LocalCompletionsAPI, loglikelihood tasks, bos token, HF model generation_config settings` 


- **LocalCompletionsAPI's loglikelihood runs!**: A member is running **loglikelihood (multiple-choice) tasks** using base models and the `LocalCompletionsAPI` implementation.
   - They confirmed that it's working great, but they can see that the tokenized prompt includes the **bos token**.
- **BOS Token Blues!**: The same member asked whether there's a way to specify `add_bos_token=False` when using `LocalCompletionsAPI`.
   - They want to control whether the **beginning-of-sequence token** is added to the prompt.
- **HF Model generation_config: Default Temperature?**: The member inquired if setting `do_sample:true` without specifying `temperature` would default to the **HF model's generation_config settings**.
   - They clarified needing `temp > 0`, otherwise it sets `do_sample` to false.


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1369790693135876177)** (1 messages): 

> `NotebookLM, Mobile App, Trusted Tester Program` 


- **NotebookLM Launches Mobile App Trusted Tester Program**: NotebookLM is launching a **mobile app (beta version)** 📱 soon, and is looking for experienced web app users to participate in a **trusted tester program** to shape its future.
   - Interested users can register by filling out [this form](https://forms.gle/XD1VmJ7FP4AjbDB66), which includes reviewing and agreeing to the **Trusted Tester Terms**.
- **Trusted Testers Needed to Beta Test NotebookLM Mobile App**: NotebookLM seeks experienced web app users to become **trusted testers** for the beta version of their mobile app.
   - Testers will gain **early access** in exchange for providing feedback and reporting bugs; registration requires agreeing to the [Trusted Tester Terms](https://forms.gle/XD1VmJ7FP4AjbDB66).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1369802829719666709)** (9 messages🔥): 

> `NotebookLM PDF Processing, NotebookLM Knowledge Base for Sales, Audio length limitations` 


- **NotebookLM PDF Processing Limit Set by Experiments**: Users report **NotebookLM** doesn't work well with large PDFs or numbers of PDFs; one user tested by asking questions further into the PDF and found issues after **200 pages**.
   - The user suggests running a quick experiment to test current limitations.
- **NotebookLM Responds Based Only on Source Material**: In the chat interface, **NotebookLM** generates responses solely based on the uploaded source materials.
   - If a question's answer doesn't exist in the imported materials, the AI will state that no information related to the question exists in the sources.
- **Knowledge Base for Sales Content Creation for NotebookLM**: A user is building a knowledge base for sales content in **NotebookLM**, using primary client decks and sales enablement materials within the **300 document limit**.
   - They plan to give access to the internal sales team, seeking guidance, examples, and understanding of limitations, especially regarding sharing and potential silos.
- **Prolonging Audio Generation via More Content**: A user asks how to make audio longer, aiming for a minimum of **12 minutes** or more.
   - Another user suggests providing more input to give the system more content to work with.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1369761967035912222)** (22 messages🔥): 

> `NotebookLM failing to answer questions, Video Uploads, Audio Overview Functionality, Podcast Length, AI 'Humanic' Behavior` 


- ****NotebookLM Troubles: System Refuses to Answer****: Users report that **NotebookLM** is responding with *'The system was unable to answer'*, even when asked to summarize the default notebook, with issues also arising when generating mind maps and study guides.
   - Some users are seeking solutions and confirming whether others are facing the same issue.
- ****Video Uploads Supported (but with Limitations)****: Users confirmed that **NotebookLM** supports video uploads in formats like **mp4** and **avi**, contrary to some inaccurate information on Google's official site, as per the [Google Support Page](https://support.google.com/notebooklm/answer/14276468?hl=en).
   - It analyzes the audio part of the video file and provides a transcript and summary, but **mov format isn't supported**.
- ****Audio Overview Access Elusive****: A user inquired about accessing the audio overview feature to interact with it but couldn't find the option.
- ****Podcast Length Varies by Language****: A user seeking tips to extend podcast length noted that changing the language to **English** allowed for generating significantly longer audio summaries (up to **49 minutes**), whereas other languages were limited to around **14 minutes**.
   - A team member stated that this is expected behavior and they are working on enabling longer audio summaries in other languages soon.
- ****Discomfort with Artificial 'Humanic' AI****: A user expressed discomfort with the 'humanic' behavior of the AI in deep dives, specifically mentioning unnatural *'uhm's'*, and inquired about removing this behavior.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1369778004502249513)** (22 messages🔥): 

> `X-Ware, Netflix Recommendation Model, Gemini Image Generation, aider postmortems, Suno Music` 


- **X Marks the Spot for Content**: Members shared links from **X** (formerly Twitter), including a general link and specific posts from users like [thegautam](https://x.com/thegautam/status/1920198569308664169), [TheAhmadOsman](https://x.com/TheAhmadOsman/status/1920236407101997243), and [openaidevs](https://x.com/openaidevs/status/1920556386083102844).
   - The shared content seemed to be of general interest to the channel, eliciting brief acknowledgements.
- **Netflix Personalizes Recs with Foundation Model**: A member highlighted that **Netflix** developed a [foundation model for personalized recommendations](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) as noted in the comments of one of the shared links.
   - This was pointed out in relation to other discussions on recommendation systems.
- **Gemini Generates Buzz with New Images**: Members shared a link showcasing [new Gemini image generation](https://x.com/OfficialLoganK/status/1920151503349711061).
   - A member mentioned that *this team will be presenting at the aie world’s fair recsys x llms track*.
- **Aider Autopsies: More Thorough Than Self Notes?**: Members noted how [aider postmortems](https://aider.chat/2025/05/07/gemini-cost.html) are very thorough, especially regarding Gemini cost analysis.
- **Suno's Sonic Styles: Yodeling Blues Concerts**: A member raved about **Suno's** ability to mix styles, particularly highlighting a successful attempt at creating a *Yodel + Blues + Live concert* mix.
   - They shared an [audio file](https://cdn.discordapp.com/attachments/1075282825051385876/1370022129050849441/you_can_YODEL_with_Suno.mp3?ex=681dfc09&is=681caa89&hm=e16a84ff105d7fc1bef2fd343a067b7ea6ffa1964772d2e3ad9900e355f2d2c2&) as evidence of **Suno's** impressive output.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1369826322351915168)** (2 messages): 

> `Claude code pod, AI Engineer conference, Early Bird Tickets, AI Engineer conference speakers` 


- **New Claude Code Pod Bursts onto Scene**: The Latent Space podcast promoted a [new Claude code pod](https://x.com/latentspacepod/status/1920240470296572316).
   - Listeners are excited about the potential new episodes and insights from the collaboration.
- **AI Engineer Conference Early Bird Tix Vanish Soon**: The AI Engineer conferences, slated for June, alerted community members that [Early Bird tickets](https://www.ai.engineer/#speakers) are expected to sell out by the weekend.
   - Attendees are encouraged to secure their tickets promptly to take advantage of the discounted rate.
- **AI Engineer Conference Speakers Revealed**: The AI Engineer conferences unveiled [the lineup of speakers](https://www.ai.engineer/#speakers) for the June event.
   - Enthusiasts are eager to see the expertise and insights the speakers will bring to the conference.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1369806752845140149)** (15 messages🔥): 

> `Fields in traits vs properties, Modular Hackathon at AGI House, Hardware Agnostic ML Systems Survey Paper, Zotero and bibtex for citations` 


- **Properties Trump Fields in Mojo Traits**: Discussion emerged around the possibility of having fields in traits in Mojo, but it was argued that *properties in traits* is a strictly better, more general idea.
   - It was noted that fields in traits *could* happen, but one would be denied the ability to add such a trait via an extension; it would need to be included in the original struct definition.
- **Modular Hackathon Hypes Hillsborough**: A final reminder was shared about the Modular Hackathon at AGI House this Saturday, and there are a few spots remaining to attend, sign up [here](https://app.agihouse.org/events/modular-hackathon-20250510).
   - Talks will be given by Modular team members as well as Mark Saroufim (GPU MODE & PyTorch), Simon Boehm and Sasha Krassovsky (Anthropic), and Dylan Patel (SemiAnalysis).
- **Hardware Agnostic ML Survey Surfaces**: A member completed their survey paper on modular and the **Hardware Lottery** piece, using it for their final presentation to help tell a good story to their peers.
   - The latest version of the paper should always be available [here](https://github.com/TheAgaveFairy/HPML-Survey-Project/blob/main/The_Quest_for_Unification__A_Survey_of_Hardware_Agnostic_Machine_Learning_Systems.pdf) and they welcome feedback.
- **Zotero Zaps Citation Struggles**: During a discussion about citations, it was recommended that using **Zotero** + **bibtex** makes most issues go away.
   - A member shared their pain that *natbib gave me about 70 errors with almost nothing linking until i caught a single unescaped '%'*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1369814755304542290)** (4 messages): 

> `Mojo roadmap, GPU programming puzzles, Colab Integration, New requires keyword` 


- **Mojo Roadmap unveiled!**: Modular posted the near-term **Mojo roadmap** on the forum, see the [official post](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395).
   - The roadmap details what's coming soon for the **Mojo** language.
- **GPU Puzzles tease Mojo programmers**: The new **GPU programming puzzles** look interesting, and some members are wondering if it's possible to run them on **Colab** for those without GPUs.
   - One member said *The new GPU programming puzzles look really cool*.
- **Colab gets Mojo (hacky)**: A member posted a *kinda-hacky implementation of a **Colab notebook** that can run **Mojo code** on the GPUs of the free and Pro tiers on Colab*: [Colab notebook](https://forum.modular.com/t/max-can-now-be-used-in-google-colab/1383/2?u=bradlarson).
   - The member admits *There's a way better experience to be had here with a little work* noting that *it builds the cell as a single Mojo file, you have to look to the logs for compile errors, etc.*
- **"Requires" Keyword Requests Raining!**: The Discord channel reacted very positively to the new **requires keyword** for adding constraints to structs and functions with lots of Mojos.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1369804249449627823)** (13 messages🔥): 

> `Collab and partnership, ReAct module signature, DSPy Caching Mechanism, RL experiment with GRPO on a Qwen 1.7B` 


- **New partnership is on the Horizon?**: A member asked if the project is open for a **collab and partnership** that both boost their communities.
   - The member inquired about starting a chat to discuss potential synergies.
- **ReAct Module Needs No Output**: A member asked about creating a **signature for a ReAct module** that only makes tool calls and doesn't require other outputs.
   - Another member suggested using *success: bool* as the output to indicate when the task is complete.
- **DSPy Cache: A Multi-Layered Mystery**: A member discovered that **DSPy** has its own caching mechanism in addition to any caching by the **LLM** provider, which can lead to unexpected results when credentials expire.
   - Multiple layers of caching, including **DSPy's cache** ([github.com/stanfordnlp/dspy/blob/main/dspy/clients/cache.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/cache.py)), **LiteLLM's cache** ([docs.litellm.ai/docs/proxy/caching](https://docs.litellm.ai/docs/proxy/caching)), and **Bedrock's cache**, can make debugging difficult.
- **GRPO Gets Going, Recall Retreats**: A member ran a small **RL experiment with GRPO** on a **Qwen 1.7B** using DSPy to optimize query rewriting for retrieval, seeing baseline recall drop from **28%** to **26%** after training.
   - More details are available in [a Twitter thread](https://x.com/tahmidtapadar/status/1920469176776302679) noting that the drop was *likely due to sparse rewards, short runs, and BM25 mismatches with CoT rewrites*.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1369781349744906381)** (7 messages): 

> `Cohere Embedding Model, Cohere Rerank Model, Cohere Embed 4` 


- **Embedding Model struggles with negotiation**: A user noticed that the **embedding model** does not work well with negotiation, and the score between the user's query *"I can pay"* and embed data for *"No, I cannot pay"* returns **0.92** which is too similar.
   - A member suggested trying the **rerank model** for things like this instead of just vector similarity.
- **Cohere Embed 4 token level embeddings**: A member asked if they can get **token level embeddings** using **Cohere Embed 4**.
   - Another member responded that one could embed one token at a time, but they would not advise it.


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1369850766302511285)** (1 messages): 

> `AI Cost Tracking, Multi-Platform AI Service Management, AI Expense Justification, AI Tool Frustrations` 


- **AIBillingDashboard tracks costs across AI platforms**: A solo founder and software engineer created [AIBillingDashboard.com](https://AIBillingDashboard.com), a platform that helps users track and optimize their **AI service costs** across multiple providers like **Cohere**, **OpenAI**, **Anthropic**, **Azure AI**, and **Google Vertex**.
   - The platform consolidates costs, helps allocate expenses, provides usage analytics, and enables budget tracking across all services.
- **Track expenses and see Optimization Opportunities**: The creator found it difficult to track and analyze costs across multiple AI services, leading to the creation of a **unified dashboard**.
   - The dashboard solves the problem of manually pulling reports from different dashboards, struggling to allocate costs to specific projects, and lacking a unified view of total **AI spend**.
- **Seeking pain points in AI Cost/Usage Tracking**: The founder is seeking feedback on the AI cost and usage tracking problems that users are facing.
   - Examples of pain points include difficulty comparing price/performance, challenges forecasting costs, and struggles justifying **AI expenses** to management.


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1369781615152332961)** (3 messages): 

> `Collaborations, Introductions` 


- **Members seek Collaborations for Competitive Profit**: A member is looking for someone to collaborate with, promising *competitive profit* to European and American collaborators.
   - Interested parties are encouraged to DM for more detailed discussions about collaboration.
- **Introductions are encouraged with a Template**: New members are welcomed and encouraged to introduce themselves, indicating the community is excited to have them.
   - A template is provided, requesting information such as **Company/Industry/University**, current projects, favorite tech/tools, and desired gains from the community.


  

---


### **Cohere ▷ #[🟢-status-updates](https://discord.com/channels/954421988141711382/1346652044181897307/1370050082556084374)** (1 messages): 

> `Embedding Models Degraded, embed-english-v2.0, embed-english-v3.0` 


- **Embedding Models Encounter Hiccups**: Cohere reported [degraded performance](https://ift.tt/WvxjUwp) affecting **embed-english-v2.0** and **embed-english-v3.0** models, with an investigation underway.
   - Further details are available on the [Cohere Status Page](https://ift.tt/bE5aXAs) with the update timestamped May 08, 2025, at 07:25AM.
- **Cohere Investigates Embedding Model Performance Issues**: Cohere is actively investigating a live incident causing degraded performance in specific embedding models.
   - The affected components include **embed-english-v2.0** and **embed-english-v3.0**, as indicated in a status update.


  

---


### **Cohere ▷ #[🎯-private-deployments](https://discord.com/channels/954421988141711382/1351999070247583848/1370058437676765358)** (1 messages): 

> `GPU Requirements, On-Premise Deployment of Command A` 


- **GPU Specs Inquiry for Command A**: A user is seeking information regarding the specific **GPU requirements** for an **on-premise installation** of **Command A**.
- **Decoding Command A's GPU Needs**: The user aims to understand the necessary **GPU specifications** to successfully deploy and run **Command A** within their own infrastructure.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1369886618055348394)** (5 messages): 

> `Tokenizer Automation, HuggingFaceBaseTokenizer Limitations, Custom Autotokenizer, ModelTokenizer Wrapper` 


- **Tokenizer Automation Goal Discussed**: A member is looking to automate tokenizer identification across model types for internal customers using `torchtune`.
   - The goal is to remove or automate the touchpoint of identifying the tokenizer, aiming for a more generic usage of `torchtune`.
- **`HuggingFaceBaseTokenizer` has Limitations for SFT**: `HuggingFaceBaseTokenizer` lacks logic for templating/tokenizing messages, restricting its use to text completions training and not SFT.
   - The discussion highlights that this tokenizer **cannot be used for Supervised Fine-Tuning (SFT)** due to the absence of message templating capabilities.
- **Custom Autotokenizer Suggested**: A suggestion was made to write a custom "autotokenizer" for internal customers, setting it as the default in the config.
   - This autotokenizer could use if statements or more clever methods to define the model name at the top of the config for the tokenizer and checkpointer.
- **`ModelTokenizer` Wrapper Planned to Close HF Gap**: There is a known gap in torchtune, with plans to provide a `ModelTokenizer` that wraps `HuggingFaceBaseTokenizer` to map HF's `apply_chat_template` to torchtune's `tokenize_messages`.
   - This enhancement is expected to greatly assist users in onboarding new models and an issue will be opened on the [repo](https://github.com/pytorch/torchtune) to sketch out the implementation details, inviting community contribution.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1369752160493113495)** (8 messages🔥): 

> `Cosine Scheduler with Warmup, Pytorch NaN bug with compiled Adam, Torchtune's get_cosine_schedule_with_warmup function, Torchtitan LR Scheduler Implementation, LR Warmup scaling` 


- **Cosine Shenanigans: Warmup and Learning Rate Schedules**: Discussion arose around implementing a [cosine scheduler with warmup](https://github.com/pytorch/torchtune/pull/2681) and dealing with a **PyTorch bug** that causes **NaN weights** when using a compiled non-fused **Adam/AdamW** optimizer with a learning rate scheduler that sets the learning rate to exactly 0 at any point during training.
   - The bug occurs when `get_cosine_schedule_with_warmup` sets the learning rate to 0 at the first step, conflicting with the initial bugfix that enabled the use of LR schedulers with optimizer compile, but one member pointed to the [Torchtitan implementation](https://github.com/pytorch/torchtitan/blob/00a53646c184493d292836f7d8bbe0bed859993f/torchtitan/components/lr_scheduler.py#L120) as a potential solution.
- **Adam's Apple: Compiled Optimizers and Zero Learning Rates cause NaN weights**: A Pytorch bug was reported where **NaN weights** resulted from using a compiled non-fused **Adam/AdamW** optimizer in conjunction with a learning rate scheduler that at some point sets the learning rate to exactly 0.
   - One member noted that *Torchtune's* `get_cosine_schedule_with_warmup` always sets the learning rate to 0 at the first step, triggering the issue when optimizer compile is enabled.
- **Titan's Approach: LR Warmup at the start of training**: It was mentioned the [Torchtitan implementation](https://github.com/pytorch/torchtitan/blob/00a53646c184493d292836f7d8bbe0bed859993f/torchtitan/components/lr_scheduler.py#L120) sets the LR ratio to `1/(warmup_steps+1)` on the first step, but unless `lr_min` is set, the last step will still be 0.
   - One member said *The torchtitan approach works too as it's reasonable to just skip the 0th step.*
- **Warming Up to LR Warmup Scaling**: A discussion about LR scaling strategy: for the warmup steps, instead of `0,1/n, 2/n, ..., n-1/n` you want `min_lr + (1/n ) * (1 - min_lr), min_lr + (2/n ) * (1 - min_lr), ..., min_lr + (n-1/n ) * (1 - min_lr)`.
   - For the cosine you want to scale the progress by the inverse of the cosine schedule, so `progress *= arccos(2*min_lr-1)/(pi*2.0*num_cycles)` will result in your max progress computed so that `cosine_lr_multiple == min_lr`.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1369780014417055804)** (3 messages): 

> `Anthropic API web search tool, LlamaParse improvements, VoyageAI multi-modal embeddings and MongoDB indexes` 


- **Anthropic API's Search Tool is Born**: The Anthropic API now supports a built-in web search tool with day 0 support in LlamaIndex, according to [this Tweet](https://twitter.com/llama_index/status/1920220803976867882).
- **LlamaParse adds Gemini and GPT4 Support**: LlamaParse is improving with new features like **GPT 4.1** and **Gemini 2.5 Pro** models, plus auto orientation, skew detection and confidence scores for parsing quality [according to this tweet](https://twitter.com/llama_index/status/1920505775677722750).
- **Multi-Modal Retrieval Voyage with MongoDB**: Learn how to do multi-modal retrieval using **VoyageAI's** multi-modal embeddings and **MongoDB's** multi-modal indexes [in this notebook](https://twitter.com/llama_index/status/1920563641990209643).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1369807531903549502)** (4 messages): 

> `Medical LLM Bot, Fine-tuning vdr-2b-multi-v1 with math formulas, Writer's Palmyra X5 and X4 in Bedrock` 


- **Medical LLM Bot Workflow Advice Requested**: A user is building a medical LLM bot and seeking guidance on implementing a workflow that includes iteratively suggesting follow-up questions based on previous answers from a local LLM.
   - They are seeking advice on whether LlamaIndex has tools to help build this kind of workflow.
- **Fine-tuning vdr-2b-multi-v1 for Math Formulas**: A user inquired about fine-tuning the **vdr-2b-multi-v1** model using the **llamaindex/vdr-multilingual-train** dataset to better handle complex math formulas in documents.
   - They noted that formulas are not present in the training data and are seeking resources, steps, or tutorials for fine-tuning in this context.
- **Palmyra Models error in LlamaIndex Bedrock**: A user reported encountering an error, *"Provider writer for model us.writer.palmyra-x5-v1:0 is not supported"*, while using **Writer's Palmyra X5 and X4** foundation models in **Amazon Bedrock** within LlamaIndex.
   - They note that the models are available in Amazon Bedrock.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1370063478622126295)** (4 messages): 

> `tinygrad CUDA, tinygrad IR, tinygrad docs, tinygrad uops` 


- **Exploring tinygrad's CUDA Integration**: A user inquired about how **tinygrad integrates CUDA support** generally.
   - They also asked whether **tinygrad** has its own **Intermediate Representation (IR)**.
- **tinygrad Documentation Dive**: A user shared a link to the [official tinygrad documentation](https://docs.tinygrad.org/).
   - They also shared links to notes on [tinygrad uops](https://xl0.github.io/tinygrad-notes/uops.html) and more [tinygrad notes](https://mesozoic-egg.github.io/tinygrad-notes/).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1369750731212587038)** (3 messages): 

> `CACHEDB environment variable` 


- **CACHEDB env var location spotted**: A member asked about the **CACHEDB** environment variable.
   - Another member pointed to *line 175 in helpers* where it is mentioned.
- **CACHEDB's Purpose Clarified**: Following up on the initial query, the **CACHEDB** environment variable's function wasn't explicitly stated.
   - Further discussion would be required to understand the variable's practical application and context within the project.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1369814601595883530)** (1 messages): 

> `AgentX Workshop, Lambda Inference API, Agentic AI` 


- **Lambda Hosts AgentX Workshop**: Lambda is hosting the **AgentX Workshop: Building Agentic AI with Lambda on 5/15 10am PT** for AgentX competition participants and AI enthusiasts looking to scale their projects using **Lambda's powerful Inference API**.
   - Participants will learn to build practical agentic applications, optimize agent performance, and deploy agents in production environments, including a live demo.
- **AgentX Prizes Announced**: Special prizes are available for AgentX Competition participants, with up to **$1,000 in credits for 1st place**, **$500 for 2nd**, and **$300 for 3rd** in both Entrepreneurship and Research tracks.
   - Interested participants can [register now](https://lu.ma/AgentX-lambda) to get the YouTube livestream link.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1369815507942772736)** (4 messages): 

> `HF Credits, Course Content, MOOC Iterations` 


- **Users Await Hugging Face Credits**: Two users reported issues with tracking **Hugging Face credits**, with one not receiving emails and the other awaiting approval.
   - The first user mentioned it was *challenging to visit the website each day*.
- **Course Lectures Confirmed and Clarified**: A prospective student asked about the course content, specifically whether the guest lectures listed on the [course website](http://llmagents-learning.org/sp25) were comprehensive.
   - The staff clarified that the listed lectures are indeed comprehensive and also confirmed that the **Spring MOOC** includes more advanced topics like *code generation and theorem proving*, whereas the **Fall version** includes more *applications topics*.
- **Another Iteration on the Way?**: A user inquired about future iterations of the course, specifically if there would be another offering in the **Fall**.
   - Staff replied that Prof Song is hosting another Berkeley class on *Agentic AI* this fall, but it is unknown whether it will be a MOOC version.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1369982709996195913)** (2 messages): 

> `AI Engineer Courses, LLM Agents MOOC` 


- **LLM Agents MOOC Recommended for Aspiring AI Engineers**: A member inquired about the best complete course to become an **AI Engineer**.
   - Another member recommended starting with the [Fall 2024 LLM Agents MOOC](https://llmagents-learning.org/f24).
- **AI Engineer Career Path**: A user asked about resources for becoming an **AI Engineer**.
   - The LLM Agents MOOC was suggested as a solid starting point.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1370110911741952000)** (1 messages): 

> `JetBrains Plugin Updates, Windsurf Editor UX Improvements, Wave 8 Release` 


- **Windsurf Wave 8 Brings UX and Plugin Boosts**: Windsurf's final **Wave 8** release introduces enhancements to the **JetBrains plugin** and improvements to the **Windsurf Editor** user experience, detailed in a [blog post](https://windsurf.com/blog/windsurf-wave-8-ux-features-and-plugins) and [changelog](https://windsurf.com/changelog).
- **JetBrains Plugin Cascade Adds Memory and Rules**: The updated **JetBrains plugin** now supports **Memories** for persistent information between sessions, **Rules** via `.windsurfrules` files, and **MCP** (Model Context Protocol) server connections, outlined in the [Jetbrains plugin changelog](https://windsurf.com/changelog/jetbrains).
- **Windsurf Editor Gains UX Features**: The **Windsurf Editor** UX sees improvements like a **Continue button**, redesigned model selector, workspace-to-conversation mapping for filtering history, enhanced code blocks and hunk navigation, editable terminal commands, and new file proposals in Chat mode, as showcased in [today's launch video](https://youtu.be/IjE8Cdxotso).


  

---


---


---


---

