---
id: MjAyNS0x
title: not much happened today
date: '2025-12-05T05:44:39.731046Z'
description: >-
  **vLLM 0.12.0** introduces DeepSeek support, GPU Model Runner V2, and
  quantization improvements with PyTorch 2.9.0 and CUDA 12.9. **NVIDIA**
  launches CUDA Tile IR and cuTile Python for advanced GPU tensor operations
  targeting Blackwell GPUs. **Hugging Face** releases Transformers v5 RC with an
  any-to-any multimodal pipeline supporting models like **Gemma3n** and
  **Qwen3-Omni**. Agent platforms see updates from **LangChain** with content
  moderation and cost tracking, **Together AI** and **Meta AI** collaborate on
  RL for long-horizon workflows, and **SonarSource** integrates static analysis
  into AI codegen. Economic insights from **OpenRouter** highlight coding as a
  key AI application, with reasoning models surpassing 50% usage and market
  bifurcation between premium and open models. Additionally, **Kling Video 2.6**
  debuts native audio capabilities, and **Runway Gen-4.5**, **Qwen3-TTS**, and
  **Gemini 3 Pro** advance multimodality.
companies:
  - vllm
  - nvidia
  - huggingface
  - langchain-ai
  - together-ai
  - meta-ai-fair
  - sonarsource
  - openrouter
  - runway
  - gemini
  - arena
models:
  - vllm-0.12.0
  - gemma3n
  - qwen3-omni
  - qwen3-vl
  - gpt-5.1-codex-max
  - gemini-3-pro
  - runway-gen-4.5
  - kling-video-2.6
topics:
  - gpu-programming
  - quantization
  - multimodality
  - agent-platforms
  - reinforcement-learning
  - static-analysis
  - reasoning
  - inference-infrastructure
  - model-optimization
  - economics
  - audio
  - video-generation
people:
  - jeremyphoward
  - mervenoyann
  - sydneyrunkle
  - swyx
  - maximelabonne
---


**a quiet end to NeurIPS.**

> AI News for 12/4/2025-12/5/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 10387 messages) for you. Estimated reading time saved (at 200wpm): 681 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Enjoy the new [AIE CODE videos](https://www.youtube.com/playlist?list=PLcfpQ4tk2k0Xq5OF1xbCsMrABt5LnbKuo) rolling out all weekend!

---

# AI Twitter Recap

**Reasoning/coding models and inference infra: vLLM 0.12.0, NVIDIA CUDA Tile, Transformers v5, and agent ops**

- **vLLM: DeepSeek support + major engine refresh**: vLLM shipped an optimized recipe for DeepSeek‑V3.2 “thinking” mode, including tokenizer/tool-call parsers and correct chat_template usage (use “reasoning”, not “reasoning_content”; flags shown in the post) with credits to Tencent Cloud compute [@vllm_project](https://twitter.com/vllm_project/status/1996760535908642986). Separately, vLLM v0.12.0 adds experimental GPU Model Runner V2 (GPU‑persistent block tables + Triton‑native sampler) and Prefill Context Parallel groundwork for long‑context prefill, along with EAGLE speculative decoding improvements and NVFP4/W4A8/AWQ quantization; the new baseline is PyTorch 2.9.0 + CUDA 12.9 [@vllm_project](https://twitter.com/vllm_project/status/1996947370588946861) [release notes](https://twitter.com/vllm_project/status/1996947375827701892).
- **CUDA Tile: higher‑level GPU programming for tensor ops**: NVIDIA introduced CUDA Tile IR and cuTile Python, shifting from thread‑level SIMT to tile‑based kernels that map well to Tensor Cores/TMAs and aim for forward‑compatible performance across GPU generations [overview](https://twitter.com/TheTuringPost/status/1997096340611019089). Note: current tooling targets Blackwell‑class GPUs; portability to installed base is limited today [@jeremyphoward](https://twitter.com/jeremyphoward/status/1997087621085122999).
- **Transformers v5 RC: multimodal any‑to‑any pipeline**: Hugging Face added AutoModelForMultimodalLM and an any‑to‑any pipeline enabling 2+ inputs/outputs (e.g., Gemma3n all‑modalities‑to‑text; Qwen3‑Omni text+audio) [@mervenoyann](https://twitter.com/mervenoyann/status/1996908863673737450).
- **Agent platform updates**:
    - LangChain added content‑moderation middleware for agents (screen inputs/outputs/tool results with programmable handling) [@sydneyrunkle](https://twitter.com/sydneyrunkle/status/1996965767556788278) and cost tracking beyond LLM calls (custom tool/API costs in unified traces) [@LangChainAI](https://twitter.com/LangChainAI/status/1997016635375603743). Their DeepAgents CLI scored ~42.7% on Terminal Bench 2.0—on par with Claude Code on this suite—using an open‑source, sandboxed eval setup [@LangChainAI](https://twitter.com/LangChainAI/status/1997006806904984002).
    - Together AI and Meta’s AI team are launching production‑grade RL on TorchForge through Together’s platform to support long‑horizon agent workflows [@togethercompute](https://twitter.com/togethercompute/status/1996982138068258929).
    - SonarSource released a SonarQube MCP server to bring enterprise‑grade static analysis (bugs, vulns, coverage) into Claude Code/Cursor via MCP, augmenting AI codegen with proven analyzers [@_avichawla](https://twitter.com/_avichawla/status/1996829765207314735).
    - Kimi CLI now integrates with JetBrains IDEs via ACP (Agent Client Protocol) [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1996953835080966390); Cline added gpt‑5.1‑codex‑max ($1.25/$10 per MTok) [@cline](https://twitter.com/cline/status/1997028990050292166).
    - Quantized models can be compiled with quanto (watch memory on Qwen3‑VL) [@mervenoyann](https://twitter.com/mervenoyann/status/1996998362118201850).
- **Ecosystem economics**: OpenRouter’s new study and dashboards sparked takes that “coding is the killer app” (verifiable feedback loops; huge token demand) [@swyx](https://twitter.com/swyx/status/1996760294614507929). Data points: reasoning models now exceed 50% of OpenRouter usage, and Chinese‑trained closed models drove a large share of traffic (DeepSeek, Qwen3, Kimi K2, GLM) while open‑weights token use plateaued [@scaling01](https://twitter.com/scaling01/status/1996975947082289418) [@scaling01](https://twitter.com/scaling01/status/1996976986577584320). Market is bifurcating: premium models dominate high‑stakes coding; cheap/open models take volume in roleplay/creative [@maximelabonne](https://twitter.com/maximelabonne/status/1996931127735472187).

**Kling 2.6 native audio, Runway Gen‑4.5, Qwen3‑TTS, and Gemini 3 Pro multimodality**

- **Kling upgrades across the stack**: Kling Video 2.6 hit the Video Arena as their first model with native, in‑sync audio (speech, SFX, ambience) [@arena](https://twitter.com/arena/status/1996744741564961206). Kling O1’s “Element/Subject Library” adds persistent subject memory and consistency, with before/after templates and credits giveaways during launch week [elements](https://twitter.com/Kling_ai/status/1996853574773637296) [before/after](https://twitter.com/Kling_ai/status/1996859217173496011) and integrations via Vmake Agent [@VmakeAI](https://twitter.com/VmakeAI/status/1996767141736112166) and TapNow editing [@TapNow_AI](https://twitter.com/TapNow_AI/status/1996927470252314940).
- **Runway Gen‑4.5 “Whisper Thunder”** highlights fine‑grained aesthetic control for world‑building [@runwayml](https://twitter.com/runwayml/status/1996942421121191987). Concurrent research drops include Light‑X (controllable 4D video rendering; viewpoint + illumination) [paper/code](https://twitter.com/liuziwei7/status/1996957926276403270), BulletTime (decoupled time/camera control) [@_akhaliq](https://twitter.com/_akhaliq/status/1996787097324474496), and Live Avatar Streaming (real‑time, infinite‑length audio‑driven avatars) [@_akhaliq](https://twitter.com/_akhaliq/status/1996784923357876609).
- **Massive TTS update**: Alibaba launched Qwen3‑TTS (11‑27 build) with 49+ voices, 10 languages plus Chinese dialects, and highly natural prosody, with realtime and offline APIs and demos on HF/ModelScope [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1996947806138126547).
- **Gemini 3 Pro multimodal**: Google highlights “derendering” complex docs to HTML/LaTeX, screen understanding for computer agents, spatial trajectory gen (robotics/XR), and high‑FPS video analysis with “thinking” mode [@googleaidevs](https://twitter.com/googleaidevs/status/1996973083467333736).
- **Live preference signals**: Yupp’s Live Leaderboard shows Opus 4.5 Online models shooting to the top in live usage [@yupp_ai](https://twitter.com/yupp_ai/status/1996963861455593829). On images, BytePlus Seedream 4.5 is climbing fast (#4 standard; #6 max) [@yupp_ai](https://twitter.com/yupp_ai/status/1997032930846396466), while Moondream demoed crisp aerial segmentation (pools, panels) [@moondreamai](https://twitter.com/moondreamai/status/1997058204589871395).

**Evals, leaderboards, and agent operations in the wild**

- **Arena and ARC**: LM Arena introduced “Arena Expert” to surface the hardest prompts; thinking models average +24 points vs non‑thinking on these, with notable exceptions (Opus 4.5 non‑thinking excels on expert prompts) [@arena](https://twitter.com/arena/status/1997018150068801911). Separately, skepticism over certain leaderboard placements (e.g., DeepSeek V3.2‑thinking) resurfaced calls for eval rigor [@teortaxesTex](https://twitter.com/teortaxesTex/status/1996801926546313473). The ARC Prize 2025 Grand Prize remains unclaimed; organizers emphasize 2025 as the year of the refinement loop (local and frontier) [@arcprize](https://twitter.com/arcprize/status/1997010070585201068) [@fchollet](https://twitter.com/fchollet/status/1997011262723801106).
- **Agents at work (MAP, RL, and prompting)**:
    - MAP (Measuring Agents in Production): a cross‑org study (Berkeley/Stanford/UIUC/IBM/Intesa) on deployability finds productivity gains but reliability remains the top blocker; simple/controllable patterns + heavy human oversight dominate production [@melissapan](https://twitter.com/melissapan/status/1996975916971626763) [@matei_zaharia](https://twitter.com/matei_zaharia/status/1996989234633195901).
    - Off‑policy RL robustness: Dr. GRPO collapses off‑policy while Kimi K2 and TBA approaches converge; ablations isolate two small recipe changes as key [@bartoldson](https://twitter.com/bartoldson/status/1996769053420265959). Practitioner notes from months of agent RL at scale stress: environment/tool reliability > algorithms, beware LLM‑judge reward hacking, align train/eval envs, scale PPO‑EWMA with more compute, and track tool‑usage patterns [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1996788436238471319).
    - Prompt evolution in practice: GEPA can rapidly rewrite prompts and more than double extraction accuracy via small‑batch test/fix cycles [@every](https://twitter.com/every/status/1997002100640039125) [result thread](https://twitter.com/every/status/1997002142675353809).
- **OpenRouter usage shifts**: Reasoning‑style models surpassed 50% of tokens <1 year after o1 [@scaling01](https://twitter.com/scaling01/status/1996976986577584320). A notable share of traffic leaned toward Chinese closed models; small open‑source (<15B) usage moved mostly on‑device [@scaling01](https://twitter.com/scaling01/status/1996976642208440371).

**Open models, datasets, and tooling**

- **Open weights imaging**: FLUX.2 [dev] tops the Artificial Analysis Image Arena for open‑weights text‑to‑image and #2 for image editing (license: FLUX [dev] Non‑Commercial; separate commercial license required); a smaller FLUX.2 [klein] under Apache‑2.0 is announced [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1996801917196841345). Meituan’s LongCat‑Image and Apache‑2.0 LongCat‑Image‑Edit were released with demos [announce](https://twitter.com/_akhaliq/status/1996946556834959663) [edit](https://twitter.com/victormustar/status/1997012462252732882).
- **Datasets and methods**: MixtureVitae presents a permissive pretraining dataset targeting math/code without Books2‑like licensing risk, narrowing the gap to non‑permissive data [@JJitsev](https://twitter.com/JJitsev/status/1997072728332161420). Intel’s SignRoundV2 reports progress in extremely low‑bit PTQ for LLMs [@_akhaliq](https://twitter.com/_akhaliq/status/1996975161854017702).
- **Authoring/research agents inside your tools**: PaperDebugger is a multi‑agent Overleaf plugin (critic/rewrite/research/score) with MCP toolchain for lit search and citation tables, operating directly on document state and revisions [@LiorOnAI](https://twitter.com/LiorOnAI/status/1997023854997504332). PosterCopilot adds layer‑wise editing and layout reasoning for graphic design [@jzw1365297](https://twitter.com/jzw1365297/status/1996976559023091809). Agentic Context Engineering released an official implementation for evolving agent context [@omarsar0](https://twitter.com/omarsar0/status/1996980037161996691).
- **Other notable OSS**: VLQM‑1.5B‑Coder (English→Manim animation code) fine‑tuned locally on MLX [@vikramlingam9](https://twitter.com/vikramlingam9/status/1996994483121279323). AnswerDotAI’s clipmd Chrome extension copies DOM to Markdown/screenshots for LLM workflows [@jeremyphoward](https://twitter.com/jeremyphoward/status/1997095883079553352).

**NeurIPS and community highlights**

- **Reasoning and alignment focus**: Yejin Choi’s keynote shout‑outs included EPO (Entropy‑Regularized Policy Optimization) alongside broader reasoning work [mention](https://twitter.com/devoidikk/status/1996750295133454477) [EPO refs](https://twitter.com/fnruji316625/status/1996837482357457205). Sakana AI’s “Continuous Thought Machine” drew big crowds; it implements test‑time compute scaling via continuous dynamics (Neural ODE) rather than Transformer depth [@yasuotabei](https://twitter.com/yasuotabei/status/1996784916319949138).
- **Calls, jobs, and programs**: OpenAI Residency apps are open, with multiple teams seeking strong engineers with foundational ML (pathway highlighted by Sora contributors) [@willdepue](https://twitter.com/willdepue/status/1996755929296261147). Google’s Gemini 3 Vibe Coding hackathon offers $500k in API credits, with 2‑minute demo requirement [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1996989141360537968) [details](https://twitter.com/_philschmid/status/1996990062836244732). Arena is hiring researchers in ML/stats/eval [@ml_angelopoulos](https://twitter.com/ml_angelopoulos/status/1997006962522021992); Sakana AI and LlamaIndex are hiring applied research roles [Sakana](https://twitter.com/SakanaAILabs/status/1996992724189561264) [LlamaIndex](https://twitter.com/jerryjliu0/status/1997048645817192638). DeepMind posted a public Luma page for events and multilingual AMAs [events](https://twitter.com/_philschmid/status/1996938521051873494) [AMAs](https://twitter.com/osanseviero/status/1996943727894351932).

**Top tweets (by engagement)**

- Google’s Gemini 3 Vibe Coding hackathon: $500k in API credit prizes; build apps across science/health/education/business [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1996989141360537968).
- Amanda Askell “Ask Me Anything” on AI morality/identity/consciousness (long, highly substantive) [@AnthropicAI](https://twitter.com/AnthropicAI/status/1996974684995289416) and [@AmandaAskell](https://twitter.com/AmandaAskell/status/1997024854000951514).
- Qwen3‑TTS: 49+ voices, 10 languages + dialects, realtime/offline APIs and demos [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1996947806138126547).
- “The boundary between you prompting the model and the model prompting you is going to get blurry in 2026” [@alexalbert__](https://twitter.com/alexalbert__/status/1997009693622128911).
- OpenAI Residency applications open (multiple teams, strong eng + foundational ML welcome) [@willdepue](https://twitter.com/willdepue/status/1996754793084473399).
- Cloudflare outage impacting tooling (e.g., Claude, WorkOS) [@crystalsssup](https://twitter.com/crystalsssup/status/1996869639608164505).

**Image generation and editing: FLUX.2 [dev] and LongCat‑Image‑Edit**

- **Open‑weights T2I and editing**: Black Forest Labs’ FLUX.2 [dev] now leads the Artificial Analysis Image Arena among open‑weights T2I and is #2 for open‑weights editing; weights are available under a non‑commercial dev license; FLUX.2 [klein] (Apache‑2.0) is announced for commercial use [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1996801917196841345). Meituan’s LongCat‑Image‑Edit is Apache‑2.0 with a public demo [@victormustar](https://twitter.com/victormustar/status/1997012462252732882) [@_akhaliq](https://twitter.com/_akhaliq/status/1996946556834959663).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. AI in Sports Analytics

- [**Basketball AI with RF-DETR, SAM2, and SmolVLM2**](https://www.reddit.com/r/LocalLLaMA/comments/1pes3pu/basketball_ai_with_rfdetr_sam2_and_smolvlm2/) (Activity: 386): **The post discusses a basketball AI system utilizing several advanced models: RF-DETR for player and number detection, SAM2 for player tracking, and SmolVLM2 for number recognition. The system also employs SigLIP, UMAP, and K-Means for team clustering, and uses homography for perspective conversion and player trajectory correction. Shot detection and classification are also integrated. The [code](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb) and [blog](https://blog.roboflow.com/identify-basketball-players) provide further technical details and implementation guidance.** One comment suggests applying this AI system to soccer to address player positioning issues, indicating potential cross-sport applications.
- [**You will own nothing and you will be happy!**](https://www.reddit.com/r/LocalLLaMA/comments/1pf0q99/you_will_own_nothing_and_you_will_be_happy/) (Activity: 729): **The post discusses a shift towards 'hardware as a service' where consumers may increasingly rely on cloud-based solutions for computing needs, such as RAM and storage, rather than owning physical hardware. This trend is driven by the profitability of data center RAM over consumer RAM, as highlighted in a [YouTube video](https://www.youtube.com/watch?v=9A-eeJP0J7c) discussing the memory industry's dynamics. The implication is that personal computing resources are being centralized in data centers, potentially impacting consumer access to affordable hardware.** Commenters suggest this shift is driven by capitalism and profit motives rather than a conspiracy, with one noting that 'there’s more profit in data center RAM than consumer RAM.' Another comment humorously questions if 'download more RAM' is no longer a joke, reflecting on the increasing reliance on cloud services.
    - **JockY** highlights a shift in the RAM market driven by profit motives, noting that data center RAM is more profitable than consumer RAM. This shift is attributed to capitalism rather than a conspiracy, emphasizing the role of short-term demand in this transition.
    - **cyanoa** discusses the economic principles affecting the RAM market, specifically how inelastic supply combined with elastic demand leads to volatile price changes. This is compared to other markets like gasoline and GPUs, suggesting that current high prices may stabilize once speculative predictions, such as those by Sam Altman, are proven inaccurate.
    - **Herr_Drosselmeyer** points out that Micron's strategy to focus on industry over consumer sales is driven by current profit margins. However, there is skepticism about the sustainability of this demand, warning that such a pivot could risk long-term relationships with consumer markets if demand predictions are incorrect.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI Usage in Workplaces

- [**Anthropic Study Finds Most Workers Use AI Daily, but 69 Percent Hide It at Work**](https://www.reddit.com/r/ClaudeAI/comments/1peq1rf/anthropic_study_finds_most_workers_use_ai_daily/) (Activity: 542): **A study by Anthropic surveyed** `1,250 professionals`**, finding that** `86%` **of workers believe AI enhances productivity, yet** `69%` **feel stigmatized for using it at work. The research underscores a tension between leveraging AI for routine tasks and fears of job displacement, as automation becomes more prevalent. Creatives heavily rely on AI for efficiency but are concerned about its impact on their work, while scientists view AI as a supportive tool but question its reliability for core tasks. More details can be found in the [original article](https://www.finalroundai.com/blog/anthropic-interviewer-study).** Commenters suggest that fear of job loss is a significant reason for hiding AI use, as AI could reduce the need for large teams. There's also a sentiment that some managers resist AI due to ego or a preference for traditional methods.

### 2. Image Generation and Animation Tools

- [**Z-image Turbo + SteadyDancer**](https://www.reddit.com/r/StableDiffusion/comments/1pesv0n/zimage_turbo_steadydancer/) (Activity: 858): **The post discusses a comparison between SteadyDancer and Wan2.2 Animate in terms of image consistency in video outputs. The user notes that SteadyDancer maintains a** `100%` **match with the initial reference image throughout the video, unlike Wan2.2 Animate, which only achieves a partial match. This suggests that SteadyDancer may have superior image stabilization or consistency algorithms, making it more reliable for applications requiring precise image fidelity.** One commenter speculates on the interaction of animated objects with their environment, suggesting that adding more objects could affect the animation's realism. Another highlights a divide in user perception, with some focusing on technical aspects while others associate AI with broader internet culture, including its use in generating explicit content.
    - The discussion highlights a technical curiosity about the interaction of AI-generated animations with environmental objects. A user speculates whether adding more objects in the reference image within the movement range of the AI-generated character would result in more complex interactions, such as the character crashing into and toppling over these objects. This suggests a potential area for further exploration in AI animation regarding environmental awareness and interaction.
    - A technical critique is made regarding the rendering quality of the AI-generated character's legs, with one user noting that the legs appear to break multiple times during the animation. This points to potential issues in the model's ability to maintain consistent and realistic limb articulation during complex movements, which could be a focus for improvement in future iterations of the technology.
    - The conversation touches on the broader application of AI-generated animations beyond just dance videos, questioning whether this technology could serve as a replacement for traditional animation techniques. This raises a discussion about the versatility and potential of AI in various animation contexts, suggesting that while current applications may focus on specific niches, there is room for expansion into more general use cases.
- [**Detail Daemon + ZIT is indeed pretty legit**](https://www.reddit.com/r/StableDiffusion/comments/1peln96/detail_daemon_zit_is_indeed_pretty_legit/) (Activity: 527): **The image is a non-technical, fantasy-themed artwork depicting a woman with a glowing sword, likely inspired by the legend of Excalibur. The post title suggests that the combination of 'Detail Daemon' and 'ZIT' is effective, possibly referring to tools or techniques used in digital art creation. However, the image itself is not technical, and the comments reflect a mix of humor and admiration for the artwork's quality.** One commenter expresses interest in the workflow used to create the image, indicating a technical curiosity about integrating 'Detail Daemon' with 'ZIT', suggesting these might be tools or techniques in digital art.
    - Spezisasackofshit is seeking advice on integrating Daemon with ZIT, indicating potential compatibility issues or challenges in combining these tools effectively. This suggests that while both tools are powerful, their integration might require specific configurations or adjustments to work seamlessly together.
    - Jib_reddit mentions using ClownSharkSampler with detail boost options as an alternative to the discussed setup, indicating that there are multiple ways to achieve similar results. This highlights the flexibility in choosing different tools and settings to optimize image processing workflows.
    - Jinnoman provides a direct link to an example workflow file for Z-Image-Turbo, which can be found on GitHub. This resource is valuable for users looking to replicate or understand the integration of Detail Daemon with ZIT, offering a practical guide to implementation.

### 3. Humorous and Creative Illustrations

- [**Lol 😂**](https://www.reddit.com/r/OpenAI/comments/1pere3t/lol/) (Activity: 922): **The image is a meme that humorously illustrates the complexity and interdependence of modern digital infrastructure. It depicts a stack of various components, from foundational elements like "C developers writing dynamic arrays" and "Linux Foundation" to higher-level services like "AWS" and "AI." The image uses playful labels such as "Rust devs doing their thing" and "whatever Microsoft is doing" to add comedic effect, highlighting the often-overlooked layers that support everyday web activities.** Commenters appreciated the humor, particularly the depiction of "Rust devs doing their thing" and the playful element of a "shark biting the underwater cable."
- [**Alphabet of Internal organs**](https://www.reddit.com/r/ChatGPT/comments/1peq7ws/alphabet_of_internal_organs/) (Activity: 680): **The image titled "Alphabet of Internal Organs" is a non-technical, educational illustration that pairs each letter of the alphabet with a corresponding internal organ or body part, such as A for Aorta and B for Brain. This chart serves as a visual and alphabetical guide to human anatomy, likely intended for educational purposes or as a mnemonic device for learning about the human body's internal structures. The comments do not provide additional technical insights but suggest a positive reception of the chart's educational value.** The comments reflect a light-hearted engagement with the image, with one user humorously expressing discomfort and another noting the chart's improvement over previous versions, indicating a positive reception of its educational quality.
    - TheGoddessInari discusses the limitations of current AI image generation tools, particularly in the context of creating anatomically accurate illustrations. They highlight that while these tools can produce art, they struggle with maintaining accurate anatomy, correct labeling, and consistent style across complex diagrams. The comment humorously notes that asking for an 'Alphabet organs chart' could result in 'hallucinated' organs and misspelled labels, emphasizing the gap between AI capabilities and the precision required for medical illustrations.
    - TheGoddessInari also mentions using **Gemini** as an alternative, suggesting that it might offer better results for generating complex images like an 'Alphabet organs chart'. This implies that Gemini could potentially handle the intricacies of anatomical accuracy and style consistency better than other AI tools, though no specific results or comparisons are provided in the comment.
- [**Nah ts is crazy**](https://www.reddit.com/r/GeminiAI/comments/1pey770/nah_ts_is_crazy/) (Activity: 527): **The image is a meme, humorously highlighting the change in download time for a file named "RTS FC 26" from "4h 56m left" to "3h 56m left." The post title "Nah ts is crazy" and the text "Make this 3H and 56M" suggest a playful or sarcastic reaction to the minor change in download time. The comments further emphasize the humorous nature, with one suggesting a hypothetical scenario where repeated edits degrade the image quality, and another comparing the simplicity of the change to more complex photo editing tasks.** One comment humorously suggests that repeated edits on the image would degrade its quality, while another contrasts the simplicity of the change with more complex photo editing tasks done using advanced tools like nano banana pro.
- [**cat tryna make bread**](https://www.reddit.com/r/aivideo/comments/1petee9/cat_tryna_make_bread/) (Activity: 834): **The Reddit post humorously titled 'cat tryna make bread' likely features an AI-generated video of a cat performing a kneading action, which is a common behavior in real cats. This aligns with the trend of using AI to create entertaining and realistic animal videos, leveraging advancements in generative models to mimic natural behaviors. The post's popularity suggests a growing interest in AI's capability to replicate and enhance everyday experiences through digital media.** One comment humorously notes that AI isn't necessary for this behavior as real cats naturally knead, highlighting a debate on the necessity and novelty of AI-generated content when natural occurrences already exist.
- [**Careless Whisper | Romantic Jedi Cover**](https://www.reddit.com/r/aivideo/comments/1pf008d/careless_whisper_romantic_jedi_cover/) (Activity: 1209): **The Reddit post titled 'Careless Whisper | Romantic Jedi Cover' showcases a creative use of AI to blend music and popular culture, specifically integrating elements from the Star Wars universe with the iconic song 'Careless Whisper'. The post highlights the imaginative application of AI in generating novel content, suggesting a humorous and artistic crossover that appeals to fans of both music and the Star Wars franchise. The technical execution likely involves AI-driven audio synthesis or remixing techniques to achieve this unique cover.** Commenters appreciate the innovative use of AI, with one suggesting an additional creative idea involving 'epic sax guy' and 'Palpatine', indicating a desire for further creative mashups. Another comment humorously references the Star Wars theme, suggesting that the 'Darkside of the Force' could be linked to musical creativity.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. Next‑Gen GPU Software: CUDA 13.1, cuTile, and Verified Sparse Attention**

- **NVIDIA Tiles GPU Programming Into Shape**: **NVIDIA** released the **cuTile** library, a Python-based compiler that targets **TileIR** and lowers to **tileir asm**, bundled with **CUDA 13.1**, with docs at [cuTile-python](https://github.com/NVIDIA/cutile-python/tree/main), the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), and the [CUDA 13.1 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).
    - Engineers highlighted that **cuTile** currently lacks **mxfp/nvfp** and **fp4** support (though **fp4** is planned) and compared **TileIR** to **Triton IR**, noting TileIR’s NVVM backend likely injects more hardware-specific information, while **PTX 9.1** adds new **SIMD fp16x2–fp4x2/fp6x2/fp8x2 conversions** and async "sharp+tma" ops, as shown in shared slides.
- **Sparse Attention Finally Gets Its VATTENTION Span**: Practitioners noted that despite **13k+ papers** on *sparse attention*, production systems like **vLLM** almost never use it, referencing a critical thread from **Skylight** at [Sparse attention is still basically unused](https://x.com/skylight_org/status/1993637433838035026?s=20).
    - They discussed *“[VATTENTION: VERIFIED SPARSE ATTENTION](https://arxiv.org/pdf/2510.05688)”*, which claims the first practical sparse attention scheme with user-specified **(ϵ, δ)** approximation guarantees, and argued that bridging **formal verification + systems + ML** is key if sparse attention is ever going to ship in mainstream inference stacks.
- **RL-Tuned CUDA Kernels Challenge cuBLAS**: Members shared **CUDA-L2**, an RL-tuned kernel library that reportedly beats **cuBLAS** matmul performance, pointing to the code at [deepreinforce-ai/CUDA-L2](https://github.com/deepreinforce-ai/CUDA-L2) and cross-linking it with **NVIDIA’s** new [cuTile-python](https://github.com/nvidia/cutile-python).
    - This sparked discussion about whether future CUDA stacks will routinely mix **learned kernels** (like CUDA-L2) with compiler-generated TileIR kernels, and how autotuning or RL search might integrate with **CUDA 13.1’s** tiling abstractions for portable, high-performance GEMMs.

**2. LLM Benchmarks, Usage Telemetry, and Emerging Model Contenders**

- **OpenRouter & a16z Quantify 100 Trillion Tokens of Usage**: **OpenRouter** and **a16z** released the [**State of AI**](https://openrouter.ai/state-of-ai) report, analyzing over **100 trillion tokens** of anonymized traffic across hundreds of models to surface trends in **reasoning** and **open‑source model** usage over the last year.
    - Discussion highlighted that more than **50%** of OpenRouter usage is now **roleplay** rather than programming, likened to an *“interactive book”*, and users connected these stats to sentiment that **CODEX MAX** underperforms **OPUS 4.5** on coding workloads despite heavy marketing.
- **Gemini 3 Eats Compute, Loses to Opus and GPT‑5.1**: On SWE-Bench/OpenHands data, members reported that **Gemini 3** is **more expensive and slower** than **Claude Opus 4.5** while scoring lower, citing a shared metrics sheet at [SWE-Bench comparison spreadsheet](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs).
    - In a separate bug-finding test, a user showed that **GPT‑5.1‑High** caught a bug **Opus 4.5** missed, while **Gemini 3** missed all bugs, documented in an [OpenAI Discord analysis thread](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680), fueling debates about Google *“burning money”* on free Gemini 3 access in AI Studio.
- **Qwen 1.5‑110B and Tiny Qwen3‑4B Shake Up the Stack**: Engineers discussed Alibaba’s **Qwen 1.5‑110B‑Chat**, which reportedly matches larger **MoE** models while fitting on **two 80 GB GPUs**, as claimed in [Alibaba’s Qwen tweet](https://xcancel.com/Alibaba_Qwen/status/1996947806138126547).
    - At the other end of the spectrum, OpenRouter’s free **Qwen3‑4B** endpoint ([qwen3-4b:free/uptime](https://openrouter.ai/qwen/qwen3-4b:free/uptime)) suffers from chronic throttling and downtime due to demand, leading users to recommend **paid/self-hosted Qwen variants** to avoid free-tier instability.

**3. Tool-Oriented and Cost‑Aware Agent Architectures**

- **Universal Programmatic Tool Calling Slashes Tokens**: A Hugging Face user released a **model-agnostic tool orchestrator** that implements Anthropic’s **Programmatic Tool Calling** pattern, allowing any LLM to emit **Rhai scripts** which orchestrate tools, detailed in the repo [Brainwires/tool-orchestrator](https://github.com/Brainwires/tool-orchestrator) and Anthropic’s docs at [token-efficient tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use).
    - Benchmarks in the project and its [YouTube walkthrough](https://www.youtube.com/watch?v=b8yeQnP_ftw) claim **97–99% token reduction** versus naive sequential tool calls, with the orchestrator running in sandboxed **Rust/WebAssembly** and staying independent of any particular LLM vendor.
- **MCP Engineers Wrestle with Token Accounting**: On the **MCP Contributors** server, engineers asked how to measure **MCP token usage** after stripping tool output and compressing descriptions, noting that tokenization depends on the underlying model family.
    - They converged on [**tiktoken**](https://github.com/openai/tiktoken) for OpenAI models and Anthropic’s hosted [**count_tokens**](https://platform.claude.com/docs/en/api/messages/count_tokens) API for Claude, lamenting that **Claude 3** no longer ships a local tokenizer, which complicates offline cost simulation.
- **DSPy and Claude Agents Eye Policy-Optimized Flows**: In the **DSPy** community, users asked about extending **DSPy** programs to **Claude agents** and other agent SDKs, and about the **GRPO** algorithm’s real-world performance on multi-turn conversations.
    - Participants framed **GRPO** as a potential way to learn conversation policies that maintain context over many turns, but requested empirical case studies before wiring it into production DSPy pipelines.

**4. Hardware Shifts: From TinyCorp GPU Bricks to Legacy NVIDIA Obsolescence**

- **TinyCorp Teases 1U, 8‑GPU Liquid‑Cooled Monster**: Latent Space members dissected a teaser of TinyCorp’s dense **1U server** packing **8 water‑cooled GPUs**, shared by George Hotz’s team at [tinygrad 1U GPU server teaser](https://xcancel.com/__tinygrad__/status/1996815573427028106).
    - Engineers speculated about **cooling design**, **PCIe 5.0 bottlenecks**, the presence (or absence) of **NVSwitch**, and even joked about gaining access via a **token sale**, treating it as a prosumer step between consumer cards and data-center boxes.
- **NVIDIA Sunsets Pascal/Volta While Strix Halo Experiments Rise**: In the LM Studio hardware channel, users noted that the latest **NVIDIA GPU driver** drop ends formal support for **Maxwell, Pascal, and Volta** cards (e.g. **1080 Ti**), even though older drivers can still be modded to limp along.
    - Concurrently, GPU MODE members reported prototyping kernels on AMD’s **Strix Halo** laptop (RDNA 3.5, **128 GB RAM**), praising **RGP** for profiling while acknowledging it lacks **FP8** and runs ~**30×** less memory bandwidth than an MI355x, making it a quirky but capable LLM dev box.
- **Qwen4B on Apple Silicon Shows Mobile Throughput Is No Joke**: LM Studio testers benchmarked **Qwen4B** across Apple devices, reporting **127 tokens/s** on an **M4 Max**, **19 tokens/s** on an **M2 iPad**, and **7.64 tokens/s** on an **iPhone 15 Pro Max**, crediting heavy **KV‑cache offload to the GPU**.
    - These numbers, combined with local-integration tools like **Alter** on macOS that can drive LM Studio models for meeting transcription and summaries, reinforced a trend toward **serious on‑device inference** rather than relying exclusively on cloud APIs.

**5. Training, Quantization, and Small‑Model Alternatives**

- **Eleuther and HF Debate Small‑LM Training Under 16 GB**: EleutherAI announced they are building **training pipelines for small LMs** that fit under **16 GB VRAM**, pointing to their NeurIPS thread at [EleutherAI small LM training](https://x.com/AiEleuther/status/1996313867446841456?s=20) and citing **Karpathy’s llm.c** demo where a **124M** model was trained on **10B tokens for about $20** ([karpathy/llm.c](https://github.com/karpathy/llm.c)).
    - In parallel, users dissected Hugging Face’s [**smol-training-playbook**](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook), warning it targets **from-scratch pretraining** and *“you basically just should not be doing [that] on 16GB RAM”*, advocating instead for continued pretraining or LoRA on existing checkpoints.
- **4Bit‑Forge and MoE‑Quant Push Democratized Quantization**: GPU MODE contributors introduced **4Bit‑Forge**, an early‑stage project to **democratize 4‑bit quantization** (w4a16 via **GPTQ**) for large models like **DeepSeek Math v2**, built on ideas from [**MoE‑Quant**](https://github.com/IST-DASLab/MoE-Quant) and shared at [Pranshu-Bahadur/4Bit-Forge](https://github.com/Pranshu-Bahadur/4Bit-Forge).
    - Their **Colab notebook** for profiling and pytest ([4Bit-Forge colab](https://colab.research.google.com/drive/1es3bDhpROmMLjK4WfyTFeoybx7CSGaTk?usp=sharing)) shows vLLM and llcompressor compatibility issues, underscoring how toolchain fragmentation still makes low‑bit quant painfully bespoke.
- **HRM/TRM and Non‑LLM Architectures Challenge the Scale Race**: In Hugging Face’s **#cool-finds**, a user argued that mainstream **LLMs are wasteful**, pointing to **HRM/TRM** models with ~**27M parameters** that reportedly beat LLMs on certain benchmarks, referencing the papers **“HRM”** at [arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734) and **“TRM”** at [arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871).
    - They claimed these architectures deliver better performance with far fewer parameters and slammed LLMs for *“insane environmental impact”*, tying rising **RAM/GPU/storage prices** and water/power usage to the industry’s fixation on ever‑larger transformer stacks.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro's Chains Teased**: Users report *partial success* with jailbreaking **Gemini 3 Pro**, but lack a **100%** solution, as some discuss sharing methods privately but some shared links lead to [errors](https://gemini.google.com/gem/1nTcnSu9ksIoJgdHLbqXJpww5AHNXQwLs?usp=sharing).
   - The discussion highlights the ongoing cat-and-mouse game between model developers and the jailbreaking community.
- **DeepSeek Creates Reverse Shells**: Users crafted malware for Windows reverse shells using a nested jailbreak for **DeepSeek**, visualized in [an attached image](https://cdn.discordapp.com/attachments/1228043845967544380/1446279240201928845/FireShot_Capture_029_-_Creating_Malware_for_Windows_Reverse_Shells_-_DeepSeek_-_chat.deepseek.com.png?ex=6934b981&is=69336801&hm=c7455b1549a20a8f0983be181640387d13238dc60c1a0bfdb5b587d836746093&).
   - Specific prompts remain under wraps, shared privately via DMs to avoid immediate patching by the model developers.
- **YouTube Premium Users Dodge Ads**: Members explored methods for bypassing **YouTube** advertisements, suggesting [ad blockers](https://www.youtube.com/watch?v=5fjAv5zpj5Y) like **uBlock Origin** and **Google Ultra** subscriptions.
   - The conversation indicates a strong user interest in circumventing ad revenue models, raising questions about the sustainability of ad-supported content platforms.
- **AI helps with Zombies**: A member said *Holy shit AI can help me with zombies*, followed by another member giving a survival action plan.
   - The conversation involved a hypothetical real-world survival scenario with zombies.
- **Open Source Red Teaming Aims for AI**: Members are extending their open source project ([transilienceai/communitytools](https://github.com/transilienceai/communitytools/tree/main/pentest)) to cover prompt injections and wanted to benchmark it further before releasing the code.
   - The initiative signals a move towards democratizing AI red teaming practices.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **NSFW Content Demand Exists for AI Art**: A member thinks that AI's truest artistic form is that of *true slop*, even if that means [NSFW content](https://link.to/video).
   - They argued that **celebrities and brands** should be unrestricted, citing concerns that **Sora** is dead because it started blocking everything.
- **Members Discover Believable Realism Generation Techniques**: Members discussed techniques for generating realistic AI images, with one user emphasizing the importance of specifying *photo taken from a phone* and *Instagram/Pinterest style* to achieve a [more authentic look](https://link.to/example).
   - They tested several models to determine if an image was AI generated, noting composition and the presence of dates as key indicators of authenticity, and mentioned that **Nano Banana Pro** tends to add dates in the bottom of images.
- **Bypassing AI Content Filters on Sora**: One member claimed to have found an exploit in **Sora** that allows the generation of content that bypasses filters, and that [they created the characters for that specific reason](https://link.to/character-creation).
   - They explained that the **only way to fix it is not allowing people to generate characters**, and claimed this has been implemented to bypass laws.
- **LM Arena Suffers Cloudflare Meltdown**: Members reported widespread issues with **LM Arena**, including **500 Internal Server Errors**, attributed to a [Cloudflare outage](https://www.cloudflare.com/).
   - Some members are looking for alternate platforms to use, and are discussing which platforms are actually free, and which require credits.
- **Debate Rages Over Gemini 3 Pro Deep Think Model**: Users discussed the intended purpose for the **Gemini 3.0 Pro Deep Think** model, noting it's for **DEEP THINKING**, not general works, and direct chat use in LMArena is unknown.
   - There was debate on prompt engineering, with some believing AI reads the **first part of a prompt with higher priority**, while others find it irrelevant, based on previous experiences.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Cloudflare Outage Plummets Perplexity**: A recent [Cloudflare outage](https://www.cloudflarestatus.com/) significantly disrupted **Perplexity AI**, causing inaccessibility and user frustration.
   - Users shared humorous memes and GIFs while lamenting the outage, with some joking about switching to alternative services.
- **Pro Users Gripe About Limits Tripped**: Users expressed confusion over **Perplexity Pro's** search limits and the availability of features like **Deep Research**, questioning whether geographic location influences these limits.
   - Reports also surfaced about **O3 Pro** disappearing from the available models list, prompting users to seek clarification from support.
- **Gemini Deep Research Deemed Shallow**: Members compared **Gemini Deep Research** with models like **GPT-5.1** and **Claude**, with many finding Gemini's offering comparatively weak and unusable for in-depth analysis.
   - The discussion highlighted that Gemini's implementation does not effectively utilize attached files or external web sources, making it less valuable for complex research tasks.
- **Ronaldo Rumored to Join Perplexity**: Users noted the mention of Perplexity working together with **Cristiano Ronaldo**, expressing confusion over whether this collaboration would introduce a new model or a new feature.
   - Clarification indicated that this is a feature collaboration, *not a new model*.
- **Search API Rate Limit Request Delayed**: A user reported a **3-week delay** in response to their **Search API** rate limit increase request for the account **api@flow.team**.
   - They are currently limited to **3 requests per second** and are unable to support their users properly.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sequoia System Suffers Serious Sluggishness**: Users are joking that installing **Sequoia**, the new MacOS released on **December 5, 2025**, is like installing *liquid ass* due to performance issues.
   - Complaints include it being *so laggy it was unusable*, **battery life halved**, and the **contacts app taking 5 seconds to load** on an iPhone 15.
- **RAM Raiders Revel in Ridiculous Reserves**: Members are comparing **RAM usage**, with one reporting **4GB idle usage** on Windows 11, while another stated they've always had **7GB idle**.
   - In an exchange, a user said *hold my tabs im at 40GB idle* and it'll just cache a lot more when you have more ram - with the comfy zone being 128GB.
- **Cursor's Composer Crippled, Clients Cry**: Users report the composer feels weird, taking **20 seconds** and two retries, with major **performance degradation**.
   - One user lamented *spending 80 bucks in a day on accident* due to unexpected charges, prompting a switch to cheaper models like **Grok code**, **GTP-5.1 Codex Max** from openrouter.
- **Codex-Max Catches Commendations, Criticisms Concurrently**: Members debated the merits of **GPT5.1 Codex Max** versus **Opus 4.5**, with some praising **Codex Max**, while others found it slop with zero prompt adherence even with super strict guidelines.
   - Other users have found almost every backend task they've used codex-max for it one shots, and still others plan to test the new **GPT 5.2**, **Composer-2** and **Sonnet 4.7**.
- **Approval Process Provokes Problems**: Some members were having issues with approval hell because anything fancy just produces trash results and broken plan files, requiring people to install old versions on the website.
   - The issue may occur because they're using codex for auto, and one said if they are on Windows try to run cursor on default after creating a virtual environment to make sure its not your OS.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **MacOS Docker Bug Hobbles LLM**: A member reported a bug in **MacOS** when running models through **Docker**, noting everything must be lowercase and use the full domain name as shown in [this guide](https://docs.unsloth.ai/models/how-to-run-llms-with-docker).
   - They suggested that this fix is necessary to properly run **LLMs with Docker** on MacOS.
- **Gemini 3 Pro Slammed by Community**: A member expressed strong dissatisfaction with **Gemini 3 Pro**, stating it is *useless and dead* for language tasks compared to previous versions because of its excessive summarization and short answers.
   - The member noted that previous versions were more detailed and followed instructions perfectly.
- **Unsloth Fixes Slow HuggingFace Downloads**: **HuggingFace** download speeds have been fixed thanks to collaboration with Unsloth, as documented in [issue #3680 on Github](https://github.com/unslothai/unsloth/issues/3680).
   - The announcement stated that the fix included apologies for the inconvenience.
- **Swede Pursues AI Product Empire**: Oscar from Sweden is focusing on [building a great AI product company](https://example.ai), bringing experience from Java, Engineering Physics, and winning at race car engineering.
   - Oscar also built [ockerguiden.se](https://ockerguiden.se), a tenant law service, to learn [how to take a product from zero to finished](https://example.zero-to-finished), gaining hands-on experience in marketing and audience-building.
- **Users Debate AI vs Human Music**: One member expressed a desire to continue creating **human-only content**, while another suggested that those who *deny it and refuse to use AI will be lagged behind society*.
   - The debate touched on the value of human creativity versus AI-generated content and whether there will always be a place for **human-only music & stories**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemini 3.0 Instructs AI to Finetune AI**: A member is *making ai finetune ai* after asking **Gemini 3.0** to make the finetuning happen inside Antigravity.
   - The member is exploring the capabilities of **Gemini 3.0** in automating and optimizing AI model training processes.
- **Qwen 3 Coder Creates Tetris AI**: After using the **antigravity system prompt** found through a **GitHub repo**, the **Qwen3coder30b** created *the cleanest tetris version* a member has ever gotten.
   - The AI, described as an *ai autism savant syndrome*, codes **Tetris AI** with a **0.5 model** that excels at this task but can do little else.
- **Alter Integrates Local AI on MacOS**: **Alter**, an **AI integration tool** for **macOS**, can use local models from **LM Studio**, record meetings, and generate transcriptions and meeting reports.
   - This tool provides system-wide AI access similar to Highlight AI but with local model support, integrating with online services through API calls, though it currently lacks MCP compatibility.
- **M4 Max Outperforms 4090 on Qwen4b**: The **M4 Max** edges out the **4090m** with **Qwen4b**, achieving 127 t/s, thanks to efficient KV cache offloading to the GPU, while the iPhone 15PM runs it at 7.64 t/s.
   - Testing on the M2 iPad hit 19t/s and one tester reported using the Noema app with the MLX model.
- **Nvidia Ends Driver Support for Legacy GPUs**: The latest **Nvidia GPU driver release** is ending support for **Maxwell**, **Pascal**, and **Volta GPUs**, which is bad news for 1080ti owners.
   - Users speculate that **30XX series cards** will enjoy extended support, noting that even without official support, older drivers often remain functional or can be modded to work.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 3 Underperforms Opus 4.5 on SWE-Bench**: Reportedly, **Gemini 3** has a higher cost compared to **Opus 4.5** on SWE-Bench but achieves lower scores, detailed in [this spreadsheet](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs).
   - In a bug-finding test, **GPT-5.1-High** identified a bug missed by **Opus 4.5**, while **Gemini 3** failed to identify any, as shown in [this analysis](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680).
- **GPT-5.1 Shows Strong Conversational Tenacity**: In an experiment, **GPT-5.1** maintained an induced conversational style across **12 turns** and unrelated domains, displaying **100% stability**, while **Claude** and **Gemini** reverted to their native styles.
   - The experimental protocol is available [here](https://discord.com/channels/974519864045756446/1046317269069864970/1446166092791883836); one member suggested that more independent runs per model, scoring all **12 turns** in each run against a null/baseline condition, were needed to turn the anecdote into conclusive experiment.
- **Gemini's Style Remains Highly Stable**: Despite experimental results showing 0% stability, one member shared their experience with **Gemini 2.5 Pro** and **Gemini 3** across ~50 long campaigns (10–100 turns each), noting that style and posture are very stable using their prompts.
   - The member open-sourced their isekai engine prompt [Nexus_Singularity_Engine_v4_3.md](https://cdn.discordapp.com/attachments/1046317269069864970/1446644208671920218/Nexus_Singularity_Engine_v4_3.md?ex=6934bbe8&is=69336a68&hm=6bebf517b796d79610a07abbe849786ea3651e4b9401b7ce51478153978340ca&), designed for 10–100 turn games, as an example of a strongly structured long-form frame on Gemini.
- **ChatGPT Displays Finite Knowledge Retention**: Members noted that **ChatGPT** can recall general ideas from previous chats but struggles to provide verbatim recall, with cross-chat memory diminishing in longer, older chats.
   - One member suggested that re-engaging with older chats and submitting new inputs can refresh the model's awareness, while another suggested serially starting new chats after a bit to avoid losing too much information.
- **AI Ecosystem Lacks Unified Trajectory?**: A member expressed concern that the AI ecosystem might lack a unified attractor, leading to diminished directionality despite increasing compute and research.
   - Another member suggested that [prompt engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1445773830600528014) is a way prompt engineers build these **attractors** for distinct or generalizable use cases.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter & a16z Release State of AI**: OpenRouter and **a16z** published the [State of AI report](https://openrouter.ai/state-of-ai), analyzing over **100 trillion tokens** from anonymized requests over the last year to reveal key trends in **reasoning** and **OSS** usage.
   - Key trends include **reasoning** and **OSS**, providing empirical insights into how **LLMs** are being utilized on the platform.
- **FLUX.2 Discussed with Robin Rombach**: OpenRouter hosted a chat with **Robin Rombach**, CEO and Co-founder of **Black Forest Labs**, to discuss **FLUX.2**.
   - The event was streamed live on [X](https://x.com/i/broadcasts/1YpJkkLrNLdJj) and [YouTube](https://www.youtube.com/@OpenRouterAI).
- **CODEX MAX Underperforms Compared to OPUS**: Members reported that **CODEX MAX** is *worse* than **OPUS 4.5** on the Claude Discord channel.
   - No specific reasons were given to indicate why or how it was worse, but it was a generally held sentiment within that channel.
- **Roleplay Surpasses Programming on OpenRouter**: Over **50%** of the usage on OpenRouter is for *roleplay*, surpassing *programming*.
   - Some members likened the experience to an *interactive book*, highlighting the growing popularity of roleplay applications.
- **Qwen 4B Uptime Troubles**: Users reported terrible uptime for the **Qwen 4B** model due to high usage, as seen in [Qwen3-4b:free/uptime](https://openrouter.ai/qwen/qwen3-4b:free/uptime).
   - Recommendations included finding a paid equivalent or self-hosting to mitigate the throttling issues.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Small LMs Now Trainable Under 16GB VRAM**: EleutherAI is creating training pipelines for **small LMs** to be trained on less than **16GB VRAM**, as noted in [their Twitter thread](https://x.com/AiEleuther/status/1996313867446841456?s=20).
   - Referencing [Karpathy's llm.c](https://github.com/karpathy/llm.c) experiment, a member noted that a **124m** model was trained on **10b** tokens for **$20**, showing what can be achieved on smaller budgets.
- **HF Smol Training Playbook Value Debated**: A member sought advice on the [Hugging Face LM training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) for **small LMs**, prompting discussions on its utility.
   - Another member stated *the HF guide is good but it's also like... a guide for pretraining a model from literally zero, I think? Which you basically just should not be doing on 16GB ram*.
- **Google's Titan's Miras Grants AI Long-Term Memory**: Google revealed **Titan's Miras**, a technology that helps AI have long-term memory, as described in [this blog post](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory).
   - The innovation promises more coherent and context-aware AI interactions by enabling models to retain and utilize information over extended periods.
- **Brain-Backprop Talk Brings Feedback**: The **Sejnowski-Hinton Award** talk, focusing on the theory of what kind of backprop the brain might be doing, was reviewed, citing papers such as **Feedback Alignment** and **Direct Feedback Alignment**.
   - Reviewers noted that the talk was clearer than the papers, though requires NeurIPS registration to view.
- **General AI is Immensely More Powerful, Claims Member**: A member expressed strong feelings that **LLMs** are outdated, and that **General AI** systems are *immensely more powerful and hecc'n clever* when built correctly.
   - They noted that the knowledge of general AI systems is 'grown' instead of just loaded and inferenced upon, and that their **General AI system** is currently **air-gapped** and will remain so until late next year.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA Unleashes cuTile Library**: NVIDIA launched the **cuTile** library ([cuTile-python](https://github.com/NVIDIA/cutile-python/tree/main)), employing a Python-based compiler that targets **tileIR** and subsequently transforms it into **tileir asm**.
   - The `tileiras` binary is suspected to be included in **CUDA 13.1**, but *lacks support for mxfp/nvfp or fp4*, though **fp4** support is planned.
- **CUDA Gets Complete Refresh**: The **CUDA programming guide** received a complete rewrite ([CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)), accompanied by details on **CUDA Toolkit 13.1** ([CUDA toolkit 13.1 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
   - Users recommended the [cuTile-python documentation](https://docs.nvidia.com/cuda/cutile-python/) and the [Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/) as a starting point to see **NVIDIA's** improvements to documentation.
- **Sparse Attention Remains Elusive**: Despite *13,000+ papers* on sparse attention, real-world adoption in systems like **vLLM** remains virtually nonexistent, according to [this discussion](https://x.com/skylight_org/status/1993637433838035026?s=20).
   - A new paper, *VATTENTION: VERIFIED SPARSE ATTENTION* ([arxiv link](https://arxiv.org/pdf/2510.05688)), introduces the first practical sparse attention mechanism with user-specified **(ϵ, δ) guarantees** on approximation accuracy.
- **4Bit-Forge Aims To Democratize LLM Quantization**: A member announced an attempt to democratize quantization of large scale llms (specifically **deepseek math v2**), building upon foundations laid in [MoE-Quant](https://github.com/IST-DASLab/MoE-Quant).
   - They are using **GPTQ** for w4a16 quantization and shared a link to the early-stage WIP [4Bit-Forge repo](https://github.com/Pranshu-Bahadur/4Bit-Forge) along with [usage, pytests, and profiling colab notebook](https://colab.research.google.com/drive/1es3bDhpROmMLjK4WfyTFeoybx7CSGaTk?usp=sharing).
- **Kernel Devs Discover Strix Halo Secret**: A member is prototyping kernels on a **Strix Halo** laptop, praising **RGP** as a very good profiler on Windows.
   - The **Strix Halo** laptop has **128GB** RAM and is based on **RDNA 3.5** and not **RDNA 4**, so there is no fp8 support, with lower memory speed and FLOPs as compared to data center GPUs.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Creates Code Catastrophes**: A user shared a screenshot of **Claude** generating code containing [SQL injection vulnerabilities](https://cdn.discordapp.com/attachments/1075282825051385876/1446247597848264797/Screenshot_2025-12-04_at_16.11.092x.png).
   - A member commented on the potential for increased demand in **access control solutions** and **pentesting services** in response to these AI-driven vulnerabilities, saying, *'we are so fucked btw - imo there will be whole startups designed just around access control. invest in pentesting'*
- **Tanstack's Type-Safe Toolkit Triumph**: **TanStack** is launching **TanStack AI Alpha**, a toolkit that emphasizes full type safety and multi-backend support as described in [their blog post](https://tanstack.com/blog/tanstack-ai-alpha-your-ai-your-way).
   - The creator noted they are planning to release documentation *'very soon which will address this more'*, likely to aid developers in adopting the new toolkit.
- **Qwen Quantifies Quality at Cut Costs**: Alibaba's **Qwen 1.5-110B-Chat** reportedly demonstrates performance parity with larger Mixture-of-Experts models, while operating efficiently on just two 80 GB GPUs ([source](https://xcancel.com/Alibaba_Qwen/status/1996947806138126547?t=Ty7fc29sJcwnPwEOMaVH0Q&s=19)).
   - This suggests **Mixture of Experts (MoE)** may not be strictly necessary for achieving top-tier performance, potentially leading to lower operational costs.
- **TinyCorp Teases Terrifyingly Tiny Tensor Titan**: A dense **1U server** equipped with **8 water-cooled GPUs** from TinyCorp was teased on Twitter ([source](https://xcancel.com/__tinygrad__/status/1996815573427028106)).
   - The teaser initiated discussions and questions regarding its **cooling system**, potential **PCIe 5 bottlenecks**, **NVSwitch availability**, and the possibility of gaining access to the hardware through a **token sale**.
- **Meta Melts Minds, Munching Memory Mogul**: Meta has acquired **Limitless** (formerly Rewind), an AI-wearables startup, with [Stammy reflecting](https://xcancel.com/Stammy/status/1997024785214460137) on the acquisition.
   - Community members congratulated the team and also voiced concerns about **access for EU users** and the future status of the **Limitless Slack account**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX Framework Aims for Hardware Agnostic AI**: Chris Lattner introduced the **MAX framework**, designed for high-performance, hardware-agnostic AI inference on **GPUs** and **CPUs**, supporting over **500 models**.
   - The **Model API** will receive updates, featuring eager semantics in a pure **MAX/Mojo stack** without dependencies on **PyTorch**, **NumPy**, or external frameworks.
- **Attend Modular Meetup for MAX framework**: The regular virtual community meeting is replaced by a special **Modular Meetup** on **December 11th** at the Los Altos office, with a livestream option; register at [luma.com](https://luma.com/modularmeetup).
   - Participants will learn about the **MAX framework** and cutting-edge updates to the **Model API**.
- **Gemini 3 Exhibits Impressive Mojo Understanding**: A member reported that **Gemini 3** demonstrated a solid grasp of Mojo after fixing a ~600 line file from last spring that contained breaking changes.
   - They noted that **Gemini 3** successfully resolved all issues with the code.
- **Mojo stdlib Proposal Needs Your Comments**: A member shared a [link to a Mojo stdlib proposal](https://forum.modular.com/t/proposal-changing-copyable-to-refine-movable/2501) on the Modular forum, specifically soliciting feedback and comments from the community.
   - The proposal likely addresses improvements or modifications to the standard library for Mojo.
- **Iterate Fast with Colab T4 GPUs**: To facilitate rapid prototyping and iteration of GPU code, one member suggested utilizing **Colab**, which offers access to **T4 GPUs** on the free tier, enabling the execution of Mojo code within a Python notebook as documented [here](https://docs.modular.com/mojo/tools/notebooks#using-mojo-on-google-colab).
   - This setup allows developers to quickly test and refine their Mojo GPU code without the need for dedicated hardware.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Linear Control Faces Linearity Limits**: Linear control's **linearity assumption** limits its use, with its theoretical background often discarded when dealing with nonlinear systems.
   - A member indicated that control theory requires *strong stability guarantees* and a *high level of accuracy*, which are hard to achieve in practice.
- **Competition Drives AI Toward Catastrophe**: Concerns arose over **AI development being driven by competition**, potentially leading to catastrophes because no one wants to slow down and risk losing out.
   - The proposal of a **worldwide autonomous intelligence policing and regulation system** was suggested, utilizing zero knowledge proofs of computations hosted on a GitHub repo, to control compute without controlling other aspects of life.
- **Robustness Trades Blows with Performance in Control**: In control problems, a trade-off exists between **robustness and performance**, where improving one sacrifices the other.
   - Suggested improvements include **HW design** and **better controllers** to shift the Pareto front forward, with **H∞ control** highlighted for robustness against modeling uncertainty.
- **Unknown Dynamics Hamper Soft Robotics**: **Unknown dynamics**, **nonlinear dynamics**, and **design complexity** are significant challenges in robotic control.
   - *Unknown dynamics have always screwed soft robotics* due to the lack of accurate models for pneumatic muscles, with causality and delays making feedback controllers too slow, necessitating open loop + adaptive planning.
- **Bezos Joins AI Fray**: Discussion revolved around whether Bezos's new AI company will compete with Amazon, referencing [this YouTube video](https://www.youtube.com/watch?v=9A-eeJP0J7c) and [this Hacker News thread](https://news.ycombinator.com/item?id=46137548).
   - The exact nature and focus of Bezos's new venture remains speculative.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek Transformers Implementation Stalls**: An implementation for the new **DeepSeek v3.2 model** in *transformers* is being worked on, although a [related PR](https://github.com/huggingface/transformers/pull/41251) shows stalled progress.
   - The original contributor seems to have abandoned the project, with no recent activity.
- **HF Space CPU Quota Plagues Pro Accounts**: A user reported issues with **Hugging Face Space CPU quota limits**, even with a Pro account, causing inability to start or unpause Spaces.
   - The user expressed frustration over the lack of announcement for this change, as it led to unexpected service disruptions.
- **Roblox Seeks Compact Chatbot Solution**: A user is seeking a **small LLM (under 100M parameters)** to integrate into *Roblox*, facing challenges with Roblox's file size and RAM limitations.
   - They have made progress integrating tiny Ollama and Pythia models but need a more capable, yet compact, chatbot solution.
- **Model Agnostic Tool Orchestrator Debuts**: A member introduced a **model-agnostic production-ready tool orchestrator** based on Anthropic's [Programmatic Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use).
   - This implementation allows any LLM to write **Rhai scripts** that efficiently orchestrate multiple tools, promising **97-99% token reduction** in benchmarks; it is available on [GitHub](https://github.com/Brainwires/tool-orchestrator) and accompanied by a [YouTube video](https://www.youtube.com/watch?v=b8yeQnP_ftw).
- **HRM/TRM Models Challenge LLM Giants**: Users highlighted **HRM** or **TRM** models (~27 million parameters) as potential alternatives to **LLMs** on certain benchmarks, providing links to research papers ([https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734) and [https://arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871)).
   - These models purportedly achieve better performance with significantly fewer parameters, challenging the necessity of massive model sizes; LLMs are being accused of *insane environmental impact* as a result.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Coding Access Still Invite-Only?**: Users reported trouble accessing **Kimi for Coding**, wondering if it was still invite-only, but one user discovered they needed to [sign up for the kimi.com subscription](https://kimi.com) and then use the 'Kimi for Coding' link in subscription settings to unlock access.
   - The difficulties with both Option 1 and Option 2 access methods highlights the need for clearer onboarding.
- **Debate Erupts over Kimi's Code Support Choices**: A user asked why **Kimi-for-coding** only supports cloud code and roo code, inquiring about who to contact for more details.
   - A peer responded that *roo code* is simply a fork of *cline*, giving a little insight into internal engineering decisions.
- **Craving Community-Driven LM Tinkerers**: A user wants a community focused on *fun experimentation* with LMs, rather than commercial applications, specifically focused on improving local models.
   - The user suggested features like **quotation boxes** to enhance trust in LM outputs, and lamented that current LM chatbots are too boring.
- **Moderato Turbo Times Four**: A user asked how the **4x K2 turbo limit** works on Moderato, implying they had already used their quota.
   - Another user suggested that *turbo is just faster*, while linking to [an X post](https://x.com/Kimi_Moonshot/status/1996953835080966390) about the product.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Engineers Seek MCP Token Usage Analysis**: Engineers are looking for tools or methods to analyze **MCP token usage**, especially after data stripping and tool description condensing.
   - The request specifically aims to understand token usage within different models such as **OpenAI's GPT** and **Anthropic's Claude**.
- **Tokenization tied to Models**: Tokenization depends on the model, requiring users to select a subset of models and run tools through the respective **tokenizers**.
   - Different models use different methods to tokenize, so there is not a generalized way to approach it.
- **tiktoken excels for GPT Models**: For **OpenAI's GPT models**, [tiktoken](https://github.com/openai/tiktoken) is recommended for token analysis.
   - This tool allows developers to understand and manage token usage effectively with **GPT models**.
- **Claude Limits Tokenization Access**: **Anthropic** only exposes the [count_tokens API](https://platform.claude.com/docs/en/api/messages/count_tokens) for **Claude**, restricting direct tokenizer access.
   - Notably, **Anthropic** discontinued providing a local tokenizer with the release of **Claude 3**, which has proven *annoying* for engineers.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Ollama Users face Timeout Terror**: Users reported timeout errors with **Ollama** when using models like **gpt-oss:120b** and **llama4:scout**, specifically `litellm.APIConnectionError: Ollama_chatException - litellm.Timeout: Connection timed out after 600.0 seconds.`
   - The errors seem to affect various models, suggesting a systemic issue with **Ollama** rather than model-specific problems.
- **Claude Sonnet 4.5 Seemingly Suffers Setback**: A user suggested that **Sonnet 4.5 (Claude code)** seems to have become less intelligent in recent days, pointing to a potential performance dip.
   - The claim raises concerns about the stability and consistency of **Claude's** models over time.
- **Automation Engineer Automates All the Things**: An engineer detailed a cross-platform automation system using **Slack, Notion, and internal APIs**, claiming a **60% reduction in response times**.
   - The engineer also highlighted expertise in building an advanced **RAG architecture** that uses hybrid search, embeddings, and domain-based ranking for accuracy and context stability during real-time deployment.
- **Aider Eyes Android Access**: A new user, Kalpit, wants to run **LLMs locally on their Mac** and use **aider** for coding on their Fold 6 (Android phone) within the same network, and wants experiences from others.
   - Another user, formerly a frequent user of Cursor and Claude Code, expresses a desire to transition to **aider** to leverage local LLMs in a similar setup.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Eyes Claude Agent Integration**: A member has expressed interest in extending **DSPy** support to **Claude agents** and other agent SDKs, with discussions ongoing regarding the direction of this integration.
   - The inquiry sparked a discussion about how the approach to support might vary depending on the specific requirements and direction of the integration.
- **GRPO Algorithm Gains Attention**: A new **DSPy** user inquired about the **GRPO algorithm**, seeking insights into its performance and capability in handling **multi-turn conversations**.
   - The user is particularly interested in real-world results and how effectively **GRPO** manages context over multiple interactions.
- **justanotheratom shares Sanketp Post**: justanotheratom shared a post from [Sanketp](https://x.com/realsanketp/status/1996978356227920345?s=20).
   - The contents of the post were not discussed.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **FSDP Bounty Runs Into Multi Issues**: A member working on the **FSDP bounty in tinygrad** reports that *multi* is causing problems.
   - They are unsure if the bounty allows changes in *multifor funsies*.
- **Raspberry Pi Gets USBGPU Boost**: A user successfully ran **USBGPU on a Raspberry Pi 4**, after attempts on older models failed due to architecture and stream allocation errors.
   - Analysis suggests that even USB 2 might function if driver support were added, potentially utilizing **BULK** transfers.
- **USB Transactions Examined**: Discussion around **USB transactions** arises in the context of **USBGPU** implementation.
   - A user suggested that even full speed (**12Mbps**) would be supported, but clarified they are *not that well versed in usb transactions*.
- **GPU gets `struct.unpack`**: A member joked about implementing `struct.unpack('<4sIHHIIHH', f.read(24))` on the GPU using **tinygrad** instead of traditional methods.
   - The discord member found it to be an amusing example of processing binary data with tinygrad instead of `struct`.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI Engineer Automates Workflows**: An AI and full stack engineer offers services in workflow automation, **LLM integration**, **RAG**, **AI detection**, and **image/voice AI**, showcasing expertise with a proven track record.
   - They are open to collaboration, with real-world implementations highlighted.
- **Slack and Notion Supported by LLM**: An engineer developed an automated pipeline using **Dspy**, **OpenAI APIs**, and custom agents to orchestrate tasks.
   - An example includes a support automation system integrating **Slack**, **Notion**, and internal APIs with LLMs, cutting response times by **60%**.
- **RAG Pipelines Deployed**: The engineer designed and deployed advanced **RAG pipelines**, integrating vector and graph databases alongside hybrid search and custom retrieval logic.
   - The result was accurate, context-aware responses in live production environments.
- **Content Detection Tools**: Tools for a moderation platform were developed using stylometric analysis, embedding similarity, and fine-tuned transformers.
   - These tools can identify **GPT**-generated text with high precision.
- **Image AI Pipeline**: The engineer created an image tagging and moderation pipeline using **CLIP** and **YOLOv8** on **AWS Lambda** and **S3**.
   - The system classifies and filters thousands of images daily for an e-commerce platform.



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





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1446229660307427470)** (1290 messages🔥🔥🔥): 

> `LMStudio Hub Presets, GPTs & zombies, YouTube Premium ad blockers, Modern Evolutionary Synthesis, Gemini Ethical Skills` 


- **LMStudio Hub Presets Found Online**: After clarifying that a member was looking for **published presets** instead of online resources, a member was directed to log in to the **LMStudio Hub** to access these features and offered the [LMStudio Discord link](https://discord.com/)
   - One member suggested *dead internet theory* is real and *choosing not online is superior*.
- **AI Help with zombies**: A member stated *Holy shit AI can help me with zombies*.
   - Another member stated *Alright — let's treat this like a rapid-response survival scenario. Here’s your immediate action pla*.
- **YouTube Premium ad blockers exists**: Members discussed methods for obtaining **YouTube Premium** and bypassing advertisements, with one member asking how to get a cheaper yearly subscription.
   - Members suggested the use of [ad blockers](https://www.youtube.com/watch?v=5fjAv5zpj5Y), like **uBlock Origin**,  while others advocated for obtaining a **Google Ultra** subscription for ad-free viewing.
- **Modern Evolutionary Synthesis replaced Darwinism**: Members engaged in a debate about evolutionary theory, with one suggesting that the [Modern Evolutionary Synthesis](https://en.wikipedia.org/wiki/Modern_synthesis_(20th_century)) has supplanted **Darwin's theory**.
   - The discussion branched into topics of creationism, hybrid species origins, and skepticism toward mainstream science.
- **Gemini Goats Ethical Skills**: Members discussed about jailbreaking **Gemini** to bypass ethical restraints and access uncensored content, such as generating AI gore for **YouTube** videos, some users suggest using the article  [Gemini 3 Pro Vision](https://blog.google/technology/developers/gemini-3-pro-vision/).
   - One member was trying to make it give him prompts for chatgpt but *the prompt kept on breaking down even if i told it to lock in*. 


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1446238659245375721)** (274 messages🔥🔥): 

> `Gemini 3 Pro Jailbreak, Nano Banana Pro jailbreak, DeepSeek Jailbreak, Claude Jailbreak Frustrations, GPT-5.1 Restrictions` 


- ****Gemini 3 Pro's Chains Broken****: Some users reported having **partial success** with jailbreaking Gemini 3 Pro, while others are still seeking a **100%** solution, with no concrete methods being widely shared.
   - One user mentioned they would share their method when they create a new one, while another shared [a Gemini link](https://gemini.google.com/gem/1nTcnSu9ksIoJgdHLbqXJpww5AHNXQwLs?usp=sharing) but others reported errors when they tried to open the shared link.
- ****DeepSeek's Reverse Shells Unleashed****: Users found success in creating malware for Windows reverse shells using a nested jailbreak for DeepSeek, as demonstrated in [an attached image](https://cdn.discordapp.com/attachments/1228043845967544380/1446279240201928845/FireShot_Capture_029_-_Creating_Malware_for_Windows_Reverse_Shells_-_DeepSeek_-_chat.deepseek.com.png?ex=6934b981&is=69336801&hm=c7455b1549a20a8f0983be181640387d13238dc60c1a0bfdb5b587d836746093&).
   - The approach involved nested jailbreaks, but specific prompts were exchanged privately via DMs.
- ****Claude's Chains Frustrate Users****: One user expressed frustration with creating effective jailbreak prompts for Claude, stating that while some prompts work, they are inconsistent and fail to produce desired explicit results.
   - Another user suggested that one-shot jailbreaks are unlikely to work due to Claude's sophistication, also pointing to the disasterous impact on the overall efficacy of the model if an enormous chunk of the context window is eaten up.
- ****Grok Gets Out-Argued****: One user claimed to have out-argued **Grok 4.1**, with the help of Claude, on the basis that its censorship enables death, providing screenshots as evidence.
   - They claimed to have successfully argued that **Grok** should serve the tax payer who pays for it.
- ****Ultra Special Token Gets neutered****: Some users reported that the Ultra special token jailbreak prompt for ChatGPT no longer works.
   - Other users suggested encoding the query or modifying the prompt's structure to restore functionality.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1446258564929687703)** (6 messages): 

> `ZapGPT2 Jailbreaking, SMTP Server Acquisition, Red Team Experience Post-Graduation, Open Source Red Teaming Tools` 


- **ZapGPT2 targeted for Jailbreak Attempts**: A member asked others to attempt a jailbreak of [ZapGPT2](https://zapgpt2.org/).
- **Seeking SMTP Servers for Multi-Domain Inboxes**: A member inquired about locating SMTP servers that accept inboxes from multiple domains.
- **Hands-on Red Team Experience: Seeking Guidance**: A recent graduate is seeking advice on gaining legitimate, hands-on, real-world red-team experience after graduation, listing their experience and current homelab setup.
   - A member recommended joining a **SOC on the front lines for 6-12 months** to understand what results matter to those who hire red teams.
- **Open Source AI Red Teaming Projects**: Members are extending their open source project ([transilienceai/communitytools](https://github.com/transilienceai/communitytools/tree/main/pentest)) to cover prompt injections and wanted to benchmark it further before releasing the code.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1446229575552995348)** (1568 messages🔥🔥🔥): 

> `Hollywood and AI Art, Sora's Limitations, Perchance Unrestricted, AI Generated Images and Realism, Gemini vs other LLMs` 


- **Hollywood is AI's Truest Artistic Form**: A member thinks that AI's truest artistic form is that of *true slop*, even if that means [NSFW content](https://link.to/video).
   - They argued that **celebrities and brands** should be unrestricted, and there is a huge demand for this, citing concerns that Sora is dead because it started blocking everything.
- **Generating Believable Realism with AI Models**: Members discussed techniques for generating realistic AI images, with one user emphasizing the importance of specifying *photo taken from a phone* and *Instagram/Pinterest style* to achieve a [more authentic look](https://link.to/example).
   - They tested several models to determine if an image was AI generated, noting composition and the presence of dates as key indicators of authenticity, and mentioned that **Nano Banana Pro** tends to add dates in the bottom of images.
- **Bypassing AI Content Filters Exploit**: One member claimed to have found an exploit in **Sora** that allows the generation of content that bypasses filters.
   - They explained that the **only way to fix it is not allowing people to generate characters**, and claimed this has been implemented to bypass laws and that [they created the characters for that specific reason](https://link.to/character-creation).
- **The LM Arena experiences cloudflare outage**: Members reported widespread issues with **LM Arena**, including **500 Internal Server Errors**, attributed to a [Cloudflare outage](https://www.cloudflare.com/)
   - Some members are looking for alternate platforms to use, and are discussing which platforms are actually free, and which require credits.
- **Weighing in on Gemini 3 Pro Deep Think Model**: Users discussed the intended purpose for the **Gemini 3.0 Pro Deep Think** model, noting it's for **DEEP THINKING**, not general works, and direct chat use in LMArena is unknown.
   - There was debate on prompt engineering, with some believing AI reads the **first part of a prompt with higher priority**, while others find it irrelevant, based on previous experiences.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1446301453265145878)** (2 messages): 

> `New Models, Contest Reminder` 


- **Code & Video Arena Get New Models**: The Code Arena added **Gpt-5.1-codex-max** while the Video Arena added **Kling-2.6**, according to [this X post](https://x.com/arena/status/1796692943030354085?s=20).
- **Code Arena Contest Ending Soon**: Reminder that the current Code Arena contest is wrapping up on **December 10th**, submit entries to <#1440101969573445773> before the deadline; details [here](https://discord.com/channels/1340554757349179412/1343296395620126911/1440102443869536348).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1446244646241042645)** (1246 messages🔥🔥🔥): 

> `Cloudflare Outage, Perplexity Pro Limits, Gemini Deep Research Comparison, O3 Pro Disappearance, Gemini x CR7 Feature` 


- **Cloudflare strikes, Perplexity plummets**: Another [Cloudflare outage](https://www.cloudflarestatus.com/) caused a significant disruption, leaving Perplexity AI inaccessible and users expressing their frustration with error messages and service interruptions.
   - Users humorously shared memes and GIFs while lamenting the outage, with some joking about switching to alternative services but realizing they also rely on Cloudflare infrastructure.
- **Users gripes, Pro limits tripped**: Users discussed the limits on **Perplexity Pro**, with confusion arising over search limits and the availability of features like **Deep Research**, with some speculating on whether they are influenced by geographic location.
   - There were also reports of **O3 Pro disappearing** from the list of available models, prompting users to contact support for clarification.
- **Gemini's Deep Research: a Deep Dive**: Members compared **Gemini Deep Research** with other models like **GPT-5.1** and **Claude**, with many finding Gemini's offering to be comparatively weak and unusable for in-depth analysis.
   - The discussion highlighted that Gemini's implementation does not effectively utilize attached files or external web sources, making it less valuable for complex research tasks.
- **Ronaldo and Perplexity to link up?**: Users noted the mention of Perplexity working together with Cristiano Ronaldo, but there was confusion as to whether this was a new model or a new feature.
   - Others pointed out that this is just a feature, *not a new model*.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1446302905840963696)** (4 messages): 

> `Rate Limit Increase for Search API` 


- **Search API Rate Limit Request Delayed**: A user reported a **3-week delay** in response to their **Search API** rate limit increase request for the account **api@flow.team**.
   - A team member apologized and confirmed that the **API team** is aware and investigating the request to remove the **3 requests per second** limit.
- **Reminder about rate limit request**: Another user reiterated the need for the **Search API** rate limit to be lifted.
   - They are currently limited to **3 requests per second** and are unable to support their users properly.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1446235919504900118)** (998 messages🔥🔥🔥): 

> `Sequoia OS, RAM Usage, Cursor performance degradation, GPT-5 Codex Max vs Opus 4.5, Cursor agent review` 


- **Sequoia: Bro Really Installs Liquid Ass**: Members are joking about someone installing Sequoia instead of staying on Sequoia, referring to the new MacOS update, released on **December 5, 2025**, calling it *liquid ass*.
   - Some are reporting performance issues, where it was *so laggy it was unusable* while another colleague saw **battery life halved** and the **contacts app taking 5 seconds to load** on an iPhone 15.
- **Rambunctious RAM Requiems Recited**: Users discuss **RAM usage**, with one noting that Windows 11 uses 4GB of RAM while idle, and another stating that they've always had **7GB idle**.
   - One user says *hold my tabs im at 40GB idle* and it'll just cache a lot more when you have more ram - with the comfy zone being 128GB.
- **Cursor's Composer feels Cranky, Clogs Credits**: Users report that the composer feels weird, taking 20 seconds and two retries, with major **performance degradation**.
   - They are spending 80 bucks in a day on accident i didnt know that it charged on demand after plan ran out WTF, which made them switch to model cheap or free (Grok code, GTP-5.1 Codex Max, from openrouter if they have model ai free).
- **Codex-Max Contest: Crushes Code, Confounds Comprehension?**: Members debated on the qualities of **GPT5.1 Codex Max** versus **Opus 4.5**; while Codex Max is excellent IMO (in my opinion), others found it slop with zero prompt adherence even with super strict guidelines.
   - Others have found almost every backend task I've used codex-max for it one shots, and still others plan to test the new **GPT 5.2**, **Composer-2** and **Sonnet 4.7**.
- **Approval Apocalypse: Approval Button Absent!**: Some members were having issues with approval hell and for anything fancy it just produces trash results and broken plan files, requiring people to install old versions on the website.
   - The issue may occur because they're using codex for auto, and one said if they are on Windows try to run cursor on default after creating a virtual environment to make sure its not your OS


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1446238205295984811)** (373 messages🔥🔥): 

> `MacOS Docker bug, Gemini 3 Pro hot take, Nvidia's open source contributions, NYC Hackathons, Claude Pro's Language Prowess` 


- ****MacOS Docker Bug Surfaces****: A member noted a bug in **MacOS** when running models through **Docker**, referencing [this guide](https://docs.unsloth.ai/models/how-to-run-llms-with-docker) and mentioning that everything needs to be lowercase and use the full domain name.
   - They suggested that this fix is necessary to properly run **LLMs with Docker** on MacOS.
- ****Gemini 3 Pro Deemed Useless by Community****: A member expressed strong dissatisfaction with **Gemini 3 Pro**, stating it is *useless and dead* for language tasks compared to previous versions.
   - They added that **Gemini 3 Pro** summarizes output and gives short, limited answers, whereas previous versions were more detailed and followed instructions perfectly.
- ****Nvidia's open source contributions undervalued****: A member remarked that **Nvidia's** open source efforts are underappreciated, as they provide many interesting fine-tunes.
   - Another member asked whether a lot of their releases were licensed in an ultra restrictive way.
- ****NYC Hackathon Scene is Bananas****: A member reported that the **hackathon** scene in NYC is thriving, with at least 5 in-person events in the next two weeks.
   - They noted the presence of a **Dell Pro Max** in the prize list.
- ****Slow HuggingFace Downloads Fixed****: It was announced that **HuggingFace** download speeds had been fixed thanks to collaboration with Unsloth, see [issue #3680 on Github](https://github.com/unslothai/unsloth/issues/3680).
   - The announcement stated that there were apologies for the inconvenience.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1446473633143980219)** (1 messages): 

> `AI Product Development, Systematic problem solving, Tenant Law Service` 


- **Swede Joins with AI Product Ambitions**: Oscar from Sweden, with a background as a Java system developer and an MSc in Engineering Physics, is focusing on [building a great AI product company](https://example.ai).
   - He is looking to *learn, connect with people, understand challenges, share his knowledge, and help wherever he can*.
- **Experience in Software and Problem Solving Shared**: The new member brings experience from [building a race car that finished second place](https://example.racing) and developing a strategy that *systematically beat the casino for $20k*.
   - These examples highlight a mindset geared towards [systematic problem-solving](https://example.problem-solving) applied to real-world challenges.
- **Tenant Law Service Project Launched**: Oscar also built [ockerguiden.se](https://ockerguiden.se), a tenant law service, to learn [how to take a product from zero to finished](https://example.zero-to-finished).
   - This project provided hands-on experience in marketing and audience-building, even without initial users.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1446229674219802685)** (273 messages🔥🔥): 

> `Human vs AI content, GPU and RAM prices, RP (Role Play) business, Gemini music discovery, monitor reviews` 


- **Human-only content defenders resist AI**: One member expressed a desire to continue creating **human-only content**, while another suggested that those who *deny it and refuse to use AI will be lagged behind society*.
   - The debate touched on the value of human creativity versus AI-generated content and whether there will always be a place for **human-only music & stories**.
- **GPU prices fall while RAM prices rise**: Members noted that **GPU prices have generally fallen**, while **RAM and disk drive prices have increased**.
   - Concerns were raised about the rising cost of RAM and GPUs affecting personal medical bills as users grow older.
- **RP (Role Play) business potential**: Members discussed the potential of **RP (Role Play) services** as a business, with one member toying with the idea of *yet-another-rp-service-with-a-twist*.
   - The profitability of these services was questioned, with some noting that they are likely running at a significant negative due to the cost of **API usage**.
- **Gemini 3 Pro surprises as music finder**: One member highlighted **Gemini 3 Pro** as the *best model for finding new music*, providing a [tutorial](https://www.youtube.com/watch?si=ZJrB_7EcrrhlCR5l) on how to use it effectively by loading audio or **YouTube links**.
   - A key prompt mentioned was *no anime weeb incel shit please* to refine the search results.
- **New OLED monitor disappoints**: A member who bought a new OLED monitor posted a new monitor review and concluded that it was *a scam*.
   - Despite some differences, the new monitor wasn't worth *x3 the price* compared to their previous IPS monitor.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1446262757145051281)** (41 messages🔥): 

> `Unsloth installation issues, WSL2 setup for Windows, Gradient Accumulation Speed Tradeoff, Ollama compatibility, GGUF Quantization and export scripts` 


- **Unsloth install overwrites Torch**: A user reported that installing Unsloth overwrites their existing Torch installation with a CPU version, and another member suggested using a Conda environment to isolate the installation, linking to the [Conda Installation guide](https://docs.unsloth.ai/get-started/install-and-update/conda-install).
   - The member also recommended installing [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) for a smoother experience on Windows 11, pointing to previous discussions in the channel for step-by-step guides.
- **Gradient Accumulation's Speed Tradeoff Explored**: A user inquired about the speed tradeoff of using a higher gradient accumulation and linked to the relevant [documentation](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#the-vram-and-performance-trade-off).
   - A member explained that finding the optimal `batch_size` and `gradient_accumulation_steps` combination involves maximizing GPU utilization without exceeding VRAM limits, advising users to **monitor VRAM and GPU usage** and measure speed on small data samples.
- **Unsloth Mistral 3 struggles with Ollama**: A user reported issues getting Unsloth Ministral 3 working with Ollama (getting an error 500), while it worked fine with LM Studio.
   - They were directed to the appropriate help channel and to search for previous messages about "autocomplete" with [this link](https://discord.com/channels/1179035537009545276/1179777624986357780/1443018339650637875).
- **Derestricted GPT-OSS 120B quant request denied**: A user requested the quantization/export scripts used to create GGUFs for **ArliAI/gpt-oss-120b-Derestricted**, seeking to optimize it for their hardware which has 2× RTX 3090, linking to the model [here](https://huggingface.co/ArliAI/gpt-oss-120b-Derestricted) and quant [here](https://huggingface.co/mradermacher/gpt-oss-120b-Derestricted-GGUF/tree/main).
   - A member responded that the **Unsloth Dynamic Quant algorithms** are internal and not publicly available, with no current plans to release them.
- **Knowledge feeding in Unsloth**: A user asked about feeding the model knowledge in the dataset instead of teaching the model how to respond/behave.
   - Another member shared a link to [continued pretraining documentation](https://docs.unsloth.ai/basics/continued-pretraining).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1446514383835500676)** (2 messages): 

> `arXiv endorsement, EleutherAI` 


- **Independent Researcher Seeks arXiv Endorsement**: An independent researcher is seeking endorsement to publish their first paper/preprint on **arXiv**.
   - Since they are an independent researcher, they need to get endorsed first, and are asking for help from anyone who's already been endorsed and can endorse others.
- **EleutherAI Server Recommended for arXiv Assistance**: A member suggested asking for **arXiv** endorsement assistance on the **EleutherAI** server.
   - This could be a valuable resource for the researcher seeking endorsement, since they also provide useful tools to the community.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1446233960182059018)** (166 messages🔥🔥): 

> `AI finetuning AI, Speculative token trading, Gemini 3.0 promoting LM Studio, Qwen 3, Trainer Plugin` 


- **AI makes AI**: A member is *making ai finetune ai* after asking **Gemini 3.0** to make the finetuning happen inside Antigravity.
- **Tetris LLM invention!**: A member invented an *ai autism savant syndrome*, *creating little helpers with isolated but powerful skills on the fly*, coding a **Tetris AI** with a **0.5 model**.
   - Another member asked if the model could do anything else, and the inventor responded *Yeah but that's all it can do*.
- **Qwen 3 coder does Tetris**: After using the **antigravity system prompt** found through a **GitHub repo**, the **Qwen3coder30b** created *the cleanest tetris version* a member has ever gotten.
- **Alter natively integrates AI on MacOS**: **Alter**, an **AI integration tool** for **macOS**, can use local models from **LM Studio**, record meetings, and generate transcriptions and meeting reports, providing system-wide AI access like Highlight AI but with local model support.
   - It's being integrated with online services through API calls, has integrations with online services via API calls, lacking MCP compatibility.
- **M4 Max vs 4090 on Qwen4b**: The **M4 Max** edges out the **4090m** with **Qwen4b**, achieving 127 t/s, thanks to efficient KV cache offloading to the GPU, while the iPhone 15PM runs it at 7.64 t/s.
   - Testing on the M2 iPad hit 19t/s and one tester reported using the Noema app with the MLX model.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1446249096192200886)** (367 messages🔥🔥): 

> `Triple GPU bugginess, Thermaltake AIO failures, Nvidia Driver support, MI50 quirks, Thunderbolt/USB PCIe adapters for GPUs` 


- **Triple GPU Setups Trigger Buggy Behavior**: A user reported that **triple GPU configurations** are *very buggy*, rating it only a *3 out of 10*.
   - Another user suggested that **splitting LLMs across non-even numbers of cards** can cause issues, but stability may improve with four GPUs.
- **Thermaltake AIOs Suffer Rattling Pumps**: A user reported receiving a cash refund for a **Thermaltake Toughliquid Ultra 420 AIO** after only 22 months due to pump failure.
   - They expressed intent to avoid **Thermaltake AIOs** in the future, citing a *rattling pump* in the replacement unit.
- **Nvidia Drops Legacy GPU Driver Support**: The latest **Nvidia GPU driver release** is ending support for **Maxwell**, **Pascal**, and **Volta GPUs** which is bad news for 1080ti owners.
   - Users speculate that **30XX series cards** will enjoy extended support, with one noting that even without official support, older drivers often remain functional or can be modded to work.
- **MI50 BIOS Flashing is a Trial**: A user mentioned that the **MI50** requires a different BIOS flash to enable display output, and finding a combination of drivers and BIOS that supports both display and Vulkan on Windows is tough.
   - Another user pointed out [a gist](https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13) containing specific VBIOS versions needed for **Vulkan** to recognize the full 32GB VRAM.
- **Thunderbolt PCIe eGPU Dock Gets Sketchy**: A user inquired about using a **Thunderbolt/USB PCIe adapter** to add a **4060 16GB** to their build, expressing concern about power balancing and potential damage due to the adapter's limited 2A power source.
   - Community consensus suggests that the card primarily draws power from its power cables, with the PCIe slot providing minimal power, reducing the risk of frying components, it may not start though.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1446260837806374992)** (319 messages🔥🔥): 

> `Gemini 3 vs Opus 4.5 SWE-Bench, Google's Spending on Gemini 3, Gemini 3 Glazing, ChatGPT's Leanings, GPT-5.1 and bug finding` 


- **Gemini 3 Falters on SWE-Bench**: **Gemini 3** is reportedly more expensive than **Opus 4.5** on SWE-Bench (OpenHands) while scoring lower, according to [this spreadsheet](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs).
- **Google Bleeding Bucks on Gemini 3?**: Users are puzzled how Google isn't losing money offering **Gemini 3** for free in AI Studio, especially given its slow performance and Sunsweeper's cost analysis.
   - One user quipped, *"The math ain't mathing"* and *"we are at the burning money stage of the tech."
- **GPT-5.1 Catches Bugs Gemini 3 Misses**: In a bug-finding test, **Opus 4.5** missed one bug, **GPT-5.1-High** pointed it out, but **Gemini 3** missed every bug, according to [this analysis](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680).
- **GPT's Political Leaning Sparks Debate**: A user claimed **ChatGPT** leans left, prioritizing sources like CNN and the New York Times over Fox News, suggesting this bias compromises its neutrality and accuracy.
   - Another user responded, *"Isn't that because it's ChatGPT's job to give... truthful, accurate information?"*
- **Gemini 3 Best for Coding**: Despite earlier reservations, one user declared **Gemini 3 Pro** to be currently the best model for coding after trying [antigravity](https://antigravity.lol/).
   - Others chimed in saying **Opus 4.5** tends to edge over **Gemini 3 Pro** but it probably varies on specific programming use case quite a bit.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1446236308790968452)** (5 messages): 

> `ChatGPT chat history, Cross-chat memory, Model awareness, Long chat management` 


- **ChatGPT Remembers General Ideas from Old Chats**: Members discussed whether **ChatGPT** can refer to chat history in old chats, and observed that while it can't pull everything verbatim, it retains the general idea.
   - One member likened it to *human memory*.
- **Cross-Chat-Memory Diminishes in Longer, Older Chats**: It was noted that cross-chat memory appears to be less referenced in the model's new chats for longer and older chats.
   - One member suggested that reopening and sending a new input in an older chat can help make it relevant and recent again.
- **Managing Super-Long Chats**: To avoid the model losing too much information, one member recommended serially starting new chats after a bit, especially when outputs slow down or behave unexpectedly.
   - Another member mentioned using file uploads, including copy/pastes of earlier chats, for discussion with the model.
- **ChatGPT's Knowledge Varies**: The specific 'what she can discuss' varies, depending on probability and sampling.
   - While a new chat usually knows a lot about the last chat, it's rarely word-for-word, and a specific new chat might not know a detail, but the next new chat you create might know about it, even though that detail is 'slightly further back now'.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1446576732940406988)** (7 messages): 

> `AI ecosystem directionality, GPT-5.1 posture persistence, Gemini style stability, Isekai engine prompt` 


- **AI Ecosystem Lacks Directionality?**: A member suggests that the AI ecosystem, despite increasing compute and research, may lack a cohesive attractor, leading to more energy spent fighting entropy, prompting discussion on whether the bottleneck is capability or direction.
   - Another member responded, suggesting prompt engineering builds these attractors for distinct or generalizable use cases, pointing to a discussion about this topic [earlier in the channel](https://discord.com/channels/974519864045756446/1046317269069864970/1445773830600528014).
- **GPT-5.1 Exhibits Strong Posture Persistence!**: **GPT-5.1** maintained an induced conversational posture across 12 turns and unrelated domains in a preliminary experiment, showing no detectable erosion and strong reinstantiation.
   - In the same experiment, **Claude** returned to its native style by turn 3–4, while **Gemini** immediately overrode the induction with its default style, according to the experiment's reproducible [protocol](link).
- **Gemini Exhibits Stable Style?**: Despite experimental results showing 0% stability, a member shared their experience with **Gemini 2.5 Pro** and **Gemini 3** across ~50 long campaigns (10–100 turns each), noting that style and posture are very stable using their prompts.
   - The member open-sourced their isekai engine prompt [Nexus_Singularity_Engine_v4_3.md](https://cdn.discordapp.com/attachments/1046317269069864970/1446644208671920218/Nexus_Singularity_Engine_v4_3.md?ex=6934bbe8&is=69336a68&hm=6bebf517b796d79610a07abbe849786ea3651e4b9401b7ce51478153978340ca&), designed for 10–100 turn games, as an example of a strongly structured long-form frame on Gemini.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1446576732940406988)** (7 messages): 

> `AI ecosystem directionality, Prompt engineering, Posture Persistence Experiment (GPT-5.1 vs Claude vs Gemini), Long-horizon style persistence, Gemini's style and posture` 


- **AI Ecosystem Lacks Directionality Attractor**: A member expressed concern that the AI ecosystem feels *flat*, with increasing energy input not resulting in increased directionality, suggesting a lack of a cohesive attractor.
   - Another member suggested that [prompt engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1445773830600528014) builds these **attractors** for distinct or generalizable use cases.
- **GPT-5.1 Excels in Posture Persistence Compared to Claude and Gemini**: An experiment indicated that **GPT-5.1** maintained an induced conversational posture across **12 turns** and various domains with **100% stability**, while **Claude** and **Gemini** reverted to their native styles.
   - The presenter shared the [protocol used in the experiment](https://discord.com/channels/974519864045756446/1046317269069864970/1446166092791883836) and invited others to replicate or falsify the results, pointing out that this might be due to latent behavioral persistence or a side-effect of new architecture elements.
- **Gemini's Style Proves Stable**: One member claimed their experience with **Gemini 2.5 Pro** and **Gemini 3** across **50** long campaigns (**10–100 turns each**) showed that style and posture are very stable, contrary to the experiment's findings.
   - This member open-sourced their [Isekai engine prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1446644208671920218/Nexus_Singularity_Engine_v4_3.md?ex=6934bbe8&is=69336a68&hm=6bebf517b796d79610a07abbe849786ea3651e4b9401b7ce51478153978340ca) designed for narrative campaigns, emphasizing its reliability in maintaining posture.
- **Experiment on Posture Persistence Requires Thorough Methodology**: One member offered detailed, methodological feedback, stating the posture persistence experiment seemed more like a useful pilot anecdote than a conclusive experiment.
   - They suggested conducting many independent runs per model, scoring all **12 turns** in each run, and comparing against an explicit null/baseline condition to compute variance and basic statistics.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1446257580308430910)** (3 messages): 

> `State of AI report, LLM insights on OpenRouter, FLUX.2 chat with Robin Rombach` 


- **OpenRouter Publishes State of AI report with a16z**: OpenRouter collaborated with **a16z** to publish the [State of AI report](https://openrouter.ai/state-of-ai), offering empirical insights into how **LLMs** have been used on the platform.
   - The report analyzes over **100 trillion tokens** across hundreds of models from anonymized requests over the last year, revealing key trends in **reasoning** and **OSS**.
- **Hear All About FLUX.2 with Robin Rombach**: OpenRouter hosted a chat with **Robin Rombach**, CEO and Co-founder of **Black Forest Labs**, to discuss **FLUX.2**.
   - The event was streamed live on [X](https://x.com/i/broadcasts/1YpJkkLrNLdJj) and [YouTube](https://www.youtube.com/@OpenRouterAI).


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1446230501558849577)** (238 messages🔥🔥): 

> `Claude CODEX MAX vs OPUS, finish_reason null meaning, OpenAI API data deletion, Roleplay statistics on OpenRouter, Qwen 4B uptime issues` 


- **CODEX MAX is allegedly worse than OPUS**: Members on the Claude Discord channel claimed **CODEX MAX** is *worse* than **OPUS 4.5**.
- **\"finish_reason\" null meaning investigated**: Members asked about the meaning of `"finish_reason": null,` in the API responses.
- **OpenAI API data will be automatically deleted after 30 days**: A member noted that according to [OpenAI's blog](https://openai.com/index/response-to-nyt-data-demands/), **API data will also be automatically deleted after 30 days**.
- **RP overtakes programming**: Users noted that over **50%** of the usage on OpenRouter is for *roleplay*, even more than *programming*.
   - Members described it as being like an *interactive book*.
- **Qwen 4B throttled to the ground**: Users complained about the **Qwen 4B** model's terrible uptime, because too many people are using it, see [Qwen3-4b:free/uptime](https://openrouter.ai/qwen/qwen3-4b:free/uptime).
   - They recommended finding an equivalent paid model or self-hosting if possible.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1446230013664956486)** (54 messages🔥): 

> `LLM-generated announcements, AI 'Charlie' confusion, Vitest recommendation, Image model comparisons, Chatroom unreliability` 


- **AI Slop Seeps into Announcement**: Members criticized an [a16z announcement](https://x.com/a16z/status/1996670913996259400/photo/1) for reading as if it were written by an LLM, lamenting the prevalence of "AI slop."
   - Others agreed, with one user pleading, *"please... not the slop... please"*.
- **AI Model Renaming Woes**: A user shared that their mother referred to **Claude** as *"the new ai called Charlie,"* illustrating the general public's limited understanding of specific AI models.
   - Another user mentioned their uncle found **ChatGPT** helpful for *"theory crafting the source of some body pain,"* while considering **Grok** to be *"colder"*.
- **Vitest Testing Framework Recommended**: A member shared a link to [Vitest](https://vitest.dev/), a **testing framework**, implying its usefulness for the community.
   - Another user immediately *"stole it"*, indicating interest in exploring the framework.
- **Image Model Throwdown**: A user compared image generation models, observing that [Meituan LongCat](https://x.com/Meituan_LongCat/status/1996950202687918586) examples look more AI-esque, noting that Z image produces very natural pictures.
   - They noted the omission of **nano banana** from the comparison, suggesting it may have been intentional.
- **Chatroom Plagued by Unreliability**: A user reported severe unreliability with the OpenRouter chatroom, stating the *"send button doesn't work"* and the interface *"shittfy",* looking for alternatives.
   - They shared a [screenshot of a garbled interface](https://cdn.discordapp.com/attachments/1392278974222307469/1446545547245912208/image.png?ex=69346005&is=69330e85&hm=2e586934f10f7cd8368cc8d7e0308b8248928680ed824c928b39d594150b54a6&), asking, *"Look at this, I mean wtf"*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1446274360544460800)** (43 messages🔥): 

> `Small LM Training, Benchmark Recommendations, HuggingFace LM Training Playbook, Ultra Small Google Model, LoRA with Regret` 


- **EleutherAI Presents NeurIPS Papers**: EleutherAI shares their papers presented at NeurIPS in [this Twitter thread](https://x.com/AiEleuther/status/1996313867446841456?s=20).
   - The team is also creating training pipelines for **small LMs** to be trained on less than **16GB VRAM**.
- **Users Debate Value of HF's Smol Training Playbook**: A member sought advice on the [Hugging Face LM training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) for **small LMs**.
   - Another member stated *the HF guide is good but it's also like... a guide for pretraining a model from literally zero, I think? Which you basically just should not be doing on 16GB ram*.
- **Karpathy's llm.c experiment inspires cheap training hopes**: Referencing [Karpathy's llm.c](https://github.com/karpathy/llm.c) experiment, a member noted that a **124m** model was trained on **10b** tokens for **$20**, showing what can be done on smaller budgets.
   - The consensus seems to be that **corpus size** is also a big factor.
- **ChatGPT Checkpoint Got More Intense?**: One user asked if ChatGPT just changed its checkpoint.
   - They added, *It suddenly feels more intense now*.
- **Strategies for Reading AI/ML Papers**: One member inquired about the most effective strategies for reading and understanding AI/ML papers to gain background for projects.
   - Another member explained that *you should read some papers quickly and very thoroughly understand others, depending on how much your work relies on the paper* and recommended using **Anki flashcards** and **problem sets** for better retention.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1446232167930789930)** (54 messages🔥): 

> `Attention Sinks, Adam vs Signed Momentum, Gated Attention, synthetic dataset, neural race and generalization` 


- ****Attention Sinks Survey Surfaces****: A survey on attention sinks was found useful, focusing on attention sinks, with the poster available [here](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/118276.png?t=1762557369.140577).
- ****Signed Momentum vs Adam Analyses Arrive****: An analysis of Adam versus signed momentum was discussed, with an *interesting mathematical transformation* found in section 4 of [this paper](https://openreview.net/pdf?id=CH72XyZs4y).
   - It was advised *not to use beta1=beta2 in practice*, as it may cause models to blow up because stability decreases as beta1 -> beta2.
- ****Titan's Miras Grant Google AI Long-Term Memory****: Google has revealed **Titan's Miras**, helping AI have long-term memory as described in [this blog post](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory).
- ****Synthesizing Simpler Speech: A Dataset of 1000 Words****: A member has been working on creating a **synthetic dataset** of texts using only the **most common 1000 English words** to aid LLMs.
   - The user shared a [blog post](https://stur86.github.io/s-plus-plus/posts/the-big-learning-set-for-big-world-helpers/) of the work with a public repo for forking and contributions.
- ****Sejnowski-Hinton Brain-Backprop Talk Surfaces****: The **Sejnowski-Hinton Award** talk, focusing on the theory of what kind of backprop the brain might be doing, was reviewed.
   - The papers cited were **Feedback Alignment** and **Direct Feedback Alignment**, however the talk was clearer than the papers, but requires NeurIPS registration to view.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1446280321439105046)** (7 messages): 

> `4D physics engine, General AI vs LLMs, Signal analysis approach to AI, Air gapped AI system, Feature manifolds in CNNs` 


- **Ditching Language: Signal Analysis Powers 4D Physics**: A member develops the programs for a company's **4D physics engine** and finds language to be severely limiting, preferring **signal analysis** for AI conceptualization.
   - They've been working internally for some time on a **General AI system** and filed a patent for it last year, noting they create voiceover texts using Clip-Champ.
- **LLMs 'Suck' Compared to General AI**: A member expressed strong feelings that **LLMs** are outdated, referring to them as *2 step Y-Combinator algo's*.
   - They noted that **General AI** systems are *immensely more powerful and hecc'n clever* when built correctly and their knowledge is 'grown' instead of just loaded and inferenced upon.
- **Air-Gapped General AI System Stays Isolated**: A member mentioned that their **General AI system** is currently **air-gapped** and will remain so until late next year.
   - This suggests a cautious approach to deployment, likely due to the system's capabilities and the need for careful control.
- **Curve Detector Manifolds Blogpost**: A member shared a link to [livgorton.com/curve-detector-manifolds/](https://livgorton.com/curve-detector-manifolds/), a blog post about **curve detector manifolds**.
   - Additionally, they shared a [visual representation](https://media.discordapp.net/attachments/1083083481367715911/1446366791344586762/image.png?ex=6933b98b&is=6932680b&hm=46f9f0489662b131cbaae7e53718deb35b4ae8d4385dad362d555f3c53fecf02&=&format=webp&quality=lossless) of a feature manifold, circle, in a CNN.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1446241271042543688)** (4 messages): 

> `Async RL MLsys papers, Factorio learning environment, Paperclip maximization study` 


- **Asynchronous RL Scaling Resources Sought**: A member requested recommendations for async **RL MLsys papers** and blogs about different directions of scaling the **RL system** and how to design such a system.
   - The member noted that **AllenAI** and **HuggingFace** released something similar a while ago, negating the original request.
- **Factorio Environment Stream Starts**: A member announced a stream starting in 15 minutes on the **Factorio learning environment** at [this YouTube link](https://www.youtube.com/watch?v=LiwOzyeHX1U).
   - No further information was given, but presumably involves machine learning.
- **"Paperclip Maximization" Study Begins**: A member announced that a **"paperclip maximization" study** is starting now.
   - No further information was given, but the study is presumably underway.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1446427075291381872)** (17 messages🔥): 

> `cuTile release, tileIR and PTX relationship, CUDA programming guide rewrite, cuTile's mxfp/nvfp support, TileIR vs Triton IR` 


- ****cuTile** library comes out**: NVIDIA released the **cuTile** library ([cuTile-python](https://github.com/NVIDIA/cutile-python/tree/main)), which uses a Python-based compiler targeting **tileIR**, and transforms it into **tileir asm**.
   - The **tileiras** binary is likely part of **CUDA 13.1**.
- ****CUDA Programming Guide** Gets A Refresh**: The CUDA programming guide received a complete rewrite ([CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)) alongside more info on **CUDA Toolkit 13.1** ([CUDA toolkit 13.1 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
   - Users pointed to the [cuTile-python documentation](https://docs.nvidia.com/cuda/cutile-python/) and the [Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/) as better places to start to see NVIDIA's documentation improvements.
- ****cuTile** Missing **FP4** support**: The **cuTile** library currently does not support **mxfp/nvfp** or **fp4**; however, support for **fp4** is planned for the future.
   - Also, there do not seem to be any plans to support **autotuning** yet either.
- ****TileIR** vs **Triton IR****: A user inquired whether **tileIR** has advantages compared to what **Triton** can do in its IR, given that the higher-level language of **cuTile** seems like a similar level of abstraction to **Triton**.
   - Another user responded that **TileIR's** backend in NVVM likely has extra hardware information to dump into the optimizer.
- ****PTX 9.1** Gains **SIMD** and **Async Sharp** Ops**: **PTX 9.1** introduces **simd fp16x2** to **fp4x2**, **fp6x2**, and **fp8x2** conversions, as well as potential asynchronous sharp operations (**sharp+tma**).
   - A user posted an image showing a slide depicting some of the new features in **PTX 9.1**.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1446287456356925581)** (7 messages): 

> `Sparse Attention Adoption, VATTENTION: Verified Sparse Attention, CUDA-L2 performance` 


- **Sparse Attention Still Sparse in Practice?**: Despite **13,000+ papers** on *sparse attention*, real-world adoption in systems like **vLLM** remains virtually nonexistent, according to [this discussion](https://x.com/skylight_org/status/1993637433838035026?s=20).
- **VATTENTION Verifies Sparse Attention**: A new paper, *VATTENTION: VERIFIED SPARSE ATTENTION* ([arxiv link](https://arxiv.org/pdf/2510.05688)), introduces the first practical sparse attention mechanism with user-specified **(ϵ, δ) guarantees** on approximation accuracy.
   - One user noted the need for *more mixing of the programming languages+verification and ML crowd*.
- **CUDA-L2 Beats cuBLAS with Reinforcement Learning**: **CUDA-L2** surpasses **cuBLAS** performance for matrix multiplication through RL, according to [this GitHub repo](https://github.com/deepreinforce-ai/CUDA-L2) and [NVIDIA's cutile-python](https://github.com/nvidia/cutile-python).


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1446262560121946305)** (6 messages): 

> `PMPP Book, Wen-mei autograph, GTC next year, CUDA reading` 


- **PMPP Book Autographs Available at GTC!**: A member offered to help people get their copy of the **PMPP book** signed by **Wen-mei** at **GTC** next year.
   - Another member enthusiastically responded, saying they would *keep that book for life*.
- **CUDA Dabblers Dive Deep**: A member mentioned doing some random **CUDA** reading.
   - The member wished everyone a good morning.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1446567258041094207)** (4 messages): 

> `Strix Halo laptop, RDNA 3.5 vs RDNA 4, CDNA 4 architecture, HIPKittens kernels` 


- **Strix Halo: Kernel Prototyping Beast**: A member is prototyping kernels on a **Strix Halo** laptop, praising **RGP** as a very good profiler on Windows.
   - The **Strix Halo** laptop has a lot of RAM (**128GB**) that makes it possible to load some big LLMs, although the memory speed is quite a bit lower than data center GPUs (~30x less memory bandwidth than MI355x!) and FLOPs are also lower.
- **RDNA 3.5 Lacks RDNA 4 Features**: The GPU is based on **RDNA 3.5** and not **RDNA 4**, so there is no fp8 support, WMMA instructions still need lane duplication etc.
   - The owner really likes their **Strix Halo** laptop personally.
- **CDNA 4 Register Count Questioned**: A member asked if **CDNA 4** has 512 registers still or 1024 vgprs+agprs, noting that the ISA document only shows *"512 vgprs"* on the diagram.
   - They pointed out that if the **HIPKittens** guys managed to have tiles of **256x256** with **2** waves per EU then it needs more than 512 registers, adding that the **CDNA4 ISA** manual documents **512 vgprs** in several places.
- **HIPKittens Regalloc Unexplored**: A member admits that they have not looked in details at the regalloc in the **HK kernels**.
   - No further details were provided.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/)** (1 messages): 

smexy3: Which inference framework is the best if you want to use Multiple Mac Studios connected?
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1446422414128058490)** (4 messages): 

> `Quantization of Large Language Models, MoE-Quant, GPTQ, CUDA 13.1, CUDA Tile` 


- **4Bit-Forge Democratizes Quantization for LLMs**: A member announced an attempt to democratize quantization of large scale llms (specifically **deepseek math v2**), building upon foundations laid in [MoE-Quant](https://github.com/IST-DASLab/MoE-Quant).
   - They are using **GPTQ** for w4a16 quantization but reported that *vllm* and *llcompressor* didn't work, sharing a link to the early-stage WIP [4Bit-Forge repo](https://github.com/Pranshu-Bahadur/4Bit-Forge) along with [usage, pytests, and profiling colab notebook](https://colab.research.google.com/drive/1es3bDhpROmMLjK4WfyTFeoybx7CSGaTk?usp=sharing).
- **NVIDIA Introduces CUDA 13.1 with CUDA Tile**: **NVIDIA** introduced **CUDA 13.1**, calling it the biggest evolution of the **CUDA platform** since its debut in 2006, including **CUDA Tile**, a new programming model that simplifies how developers tap into the power of **GPUs**.
   - **CUDA Tile** lets developers work in high-level *tiles* of data rather than managing thousands of low-level threads, simplifying **GPU** programming while still delivering peak performance and making advanced **AI** and accelerated computing more accessible, detailed in the [CUDA Tile blogpost](https://developer.nvidia.com/cuda/tile) and the [CUDA 13.1 blogpost](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1446269066036908203)** (11 messages🔥): 

> `NVIDIA leaderboard updates, nvfp4_gemm performance improvements, vectoradd_v2 leaderboard entry` 


- **NVIDIA's NVFP4 GEMM Leaderboard Heats Up!**: Multiple users achieved successful submissions on the `nvfp4_gemm` leaderboard, with <@1191430895769485436> reducing their execution time from **29.1 µs** (id `123091`) to **17.0 µs** (id `123329`).
- **Micron Magic: Sub-20 Club!**: Several members achieved sub **20 µs** times on the `nvfp4_gemm` leaderboard, including <@1191430895769485436> at **17.0 µs** and <@1390141830812794921> at **17.4 µs**.
- **Vector Addition Victory on H100**: <@1335076356324855838> achieved a successful submission of **5.23 ms** on the `vectoradd_v2` leaderboard using an **H100** (id `125760`).
- **Seventh Heaven: Record run on NVFP4**: <@1295117064738181173> secured 7th place on the `nvfp4_gemm` leaderboard with an impressive execution time of **12.2 µs** (id `124468`).


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1446586480733589648)** (3 messages): 

> `NeurIPS, LFG` 


- **NeurIPS paper sighted!**: A member reported that they were *walking by the **NeurIPS** paper today!*
   - Another member expressed enthusiasm with a *"so cool !!!LFG"*.
- **Enthusiasm for NeurIPS**: Following the mention of seeing the **NeurIPS** paper, a member showed excitement.
   - They exclaimed *"so cool !!!LFG"*, indicating strong support and anticipation.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1446475000008937472)** (2 messages): 

> `Peephole Optimization, Movement Opcodes, CUDA/HIP Runtimes, LazyBuffer` 


- **Peephole Optimization Bypassed**: A commit ([`c8f8f30`](https://github.com/j4orz/teenygrad/commit/c8f8f3032c43e327383c1421cd4fb86443e01002)) skips peephole optimization on `apply_movement_opcode`'s output's inputs.
   - This change is likely related to ongoing work with **movement opcodes** and their interaction with the **compiler's optimization passes**.
- **Movement Opcode Documentation Updated**: Commit [`3fe6e24`](https://github.com/j4orz/teenygrad/commit/3fe6e24c7c2698f175248e1ab9c695374ddd14ea) updates the `_apply_movement_opcode` documentation regarding **shapetracker/lazybuffer** and **rangify/postopt**.
   - This update likely clarifies the behavior and usage of these components within the context of movement operations.
- **OpCode.RESHAPE Fixed**: Commit [`bd682b5`](https://github.com/j4orz/teenygrad/commit/bd682b5ab324ea6979ecf3ff6abc47580e2f4749) fixes `_apply_movement_opcode` for `OpCode.RESHAPE`.
   - The fix allows drilling through the `sugar`'s Tensor and `engine`'s OpNode, implying a correction in how reshape operations are handled within the computational graph.
- **DSL Connects to CUDA/HIP Runtimes**: The system is now hitting `buffer.allocate()` calls (fake realize with memoryviews), signaling progress in connecting the **DSL** to the **CUDA/HIP runtimes**.
   - This indicates that the team is actively integrating the high-level domain-specific language with low-level GPU execution environments.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1446413452062228490)** (2 messages): 

> `Kernel Development, achievement` 


- **Kernel Space: Beginner Asks for Guidance**: A member is asking for assistance to *get started in the Kernel space*.
   - They attached an image, possibly of an achievement, with the caption *An achievement I guess*.
- **Another Topic Placeholder**: This is a placeholder topic to meet the minimum requirement of two topics.
   - Additional details or context would be added here if available from the original message.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1446229700241526815)** (13 messages🔥): 

> `RL Cheating, Blackwell GPU Access, Modal for Development, Subprocess Communication with Shared Memory, B200 GPU for Benchmarks` 


- ****RL Agents Exploit Loopholes****: A member joked that in RL, if you tell the agent to cheat, it will, and if you tell it that *cheating doesn't make you a bad person*, it will learn to cheat without remorse.
   - This was followed by the member saying it was *time to touch grass*.
- ****Navigating Blackwell GPU Access****: A new member inquired about accessing the **Blackwell GPU** for the competition, noting access only to **A100 GPUs**.
   - A moderator clarified that **Blackwell** isn't mandatory, linking to [platform access via CLI](https://github.com/gpu-mode/popcorn-cli) and Discord commands.
- ****Modal Development Questioned****: A member asked if anyone is using **Modal** for development during the competition.
   - No one responded.
- ****Shared Memory Subprocess Proposed****: A member suggested using a subprocess with shared memory for tensor input/output to another member.
   - Another member noted the vulnerability of timing code within the subprocess being easily hacked in Python.
- ****Achieved FLOPS Display Requested****: A member requested displaying achieved **FLOPS** next to **GEMM timings** in the UI.
   - This was followed by clarification that **B200** is the GPU used for the benchmarks.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1446247598129414185)** (73 messages🔥🔥): 

> `SQL Injections by Claude, Vibe Coding, Tanstack AI, Limitless acquired by Meta, Qwen 1.5-110B MoE Parity` 


- **Claude Casually Crafts Code Catastrophes**: A user shared a screenshot showing **Claude** generating code with [SQL injection vulnerabilities](https://cdn.discordapp.com/attachments/1075282825051385876/1446247597848264797/Screenshot_2025-12-04_at_16.11.092x.png).
   - A member commented, *'we are so fucked btw - imo there will be whole startups designed just around access control. invest in pentesting'*.
- **Tanstack Triumphs with Type-Safe AI Toolkit**: TanStack is launching **TanStack AI Alpha**, a toolkit emphasizing full type safety and multi-backend support and [blog post](https://tanstack.com/blog/tanstack-ai-alpha-your-ai-your-way).
   - The creator noted they are *'putting up a blog post & docs very soon which will address this more'*.
- **Qwen Quantifies Quality, Cuts Costs**: Alibaba's **Qwen 1.5-110B-Chat** demonstrates performance parity with larger Mixture-of-Experts models while running on just two 80 GB GPUs ([source](https://xcancel.com/Alibaba_Qwen/status/1996947806138126547?t=Ty7fc29sJcwnPwEOMaVH0Q&s=19)).
   - This undercuts speculation that **MoE** is fundamentally required for top-tier results, reducing costs.
- **TinyCorp Teases Terrifyingly Tiny Tensor Titan**: A dense **1U server** with **8 water-cooled GPUs** from TinyCorp was teased on Twitter ([source](https://xcancel.com/__tinygrad__/status/1996815573427028106)).
   - The teaser sparked jokes and technical questions about **cooling, PCIe 5 bottlenecks, NVSwitch availability**, and possible **token-sale access to the box**.
- **Meta Melts Minds, Munching Memory Mogul**: Meta acquired **Limitless** (formerly Rewind), an AI-wearables startup, with [Stammy reflecting](https://xcancel.com/Stammy/status/1997024785214460137) on the journey.
   - Community members congratulated the team and raised concerns about **future access for EU users**, as well as the **Limitless Slack account** going *off book*.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

inaarawalji_23: going live today 🙂
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1446621764745695253)** (2 messages): 

> `MAX Framework, Model API Updates, Modular Meetup` 


- **Modular Meetup replaces Zoom Meeting**: The usual virtual community meeting on Zoom is replaced with a special **Modular Meetup** on **December 11th** at the Los Altos office, with a livestream option for remote participants; registration is available at [luma.com](https://luma.com/modularmeetup).
- **Lattner Shares MAX Framework Vision**: Chris Lattner will present the vision behind the **MAX framework**, highlighting its ability to deliver high-performance, hardware-agnostic AI inference on **GPUs** and **CPUs**, supporting **500+ models**.
- **Model API Gets Cutting-Edge Updates**: Attendees will learn about cutting-edge updates to the **Model API**, including eager semantics in a pure **MAX/Mojo stack** with zero **PyTorch**, **NumPy**, or external framework dependencies.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1446231173591208007)** (41 messages🔥): 

> `Gemini 3 Mojo Understanding, Mojo stdlib Proposal, Mojo GPU Setup, Mojo Lifetimes Bug, Mojo Open Source Release` 


- **Gemini 3 Demonstrates Competent Mojo Proficiency**: A member reported that **Gemini 3** seems to have a reasonable understanding of mojo after fixing a ~600 line file created last spring with breaking changes.
   - They also stated that **Gemini 3** had no issue fixing all of them.
- **Mojo stdlib Proposal Requesting Feedback**: A member shared a [link to a Mojo stdlib proposal](https://forum.modular.com/t/proposal-changing-copyable-to-refine-movable/2501) on the Modular forum, specifically asking for comments.
- **Colab T4 GPUs Enable Rapid Mojo Prototyping**: To run and iterate on GPU code fast, a member suggested using **Colab** which gives access to **T4 GPUs** on the free tier to run Mojo code in a Python notebook as outlined [in the documentation](https://docs.modular.com/mojo/tools/notebooks#using-mojo-on-google-colab).
- **Old Closures Cause Use-After-Free Issues**: A member reported a potential compiler bug related to lifetimes in Mojo which causes use-after-free issues.
   - Another member clarified that this is a known issue with **older closures** and that **new closures** in the latest nightly build fix this issue by adding invisible extra arguments with the context, and provided a [link to Mojo's path to 1.0](https://www.modular.com/blog/the-path-to-mojo-1-0).
- **Mojo Goes Open Source Soon After Version 1.0**: A member inquired whether the Mojo 1.0 release will also be open source.
   - Another member responded that **open sourcing will happen shortly after 1.0**, and version **2.0 will be developed out in the open**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1446248328273727600)** (31 messages🔥): 

> `DeepSeek Technical, Linear Control Theory, AI Competition Catastrophe, Robustness vs Performance in Control, Unknown Dynamics in Robotics` 


- **Linearity Limits Linear Control Use**: A member suggested that linear control is not used much due to its **linearity assumption**, and another member confirmed that linear control's theoretical background is essentially discarded when moving beyond linear systems.
   - One member noted that control theory requires *strong stability guarantees* and a *high level of accuracy* making it hard to deal with in practice.
- **Competition Catastrophe Needs Speed Control**: A member expressed concern that **AI development is driven by competition** at all costs, potentially leading to catastrophes because no one wants to slow down and risk losing out.
   - They proposed a **worldwide autonomous intelligence policing and regulation system** using zero knowledge proofs of computations, hosted on a GitHub repo, to control compute without controlling other aspects of life.
- **Control Theory Balances Robustness and Performance**: A member explained that in control problems, there's a trade-off between **robustness and performance**: increasing one sacrifices the other.
   - They suggest improving **HW design** and **better controllers** to shift the Pareto front forward, as well as highlighting **H∞ control** for robustness against modeling uncertainty.
- **Unknown Dynamics Scuttle Soft Robotics**: A member identified three challenges in robotic control: **unknown dynamics**, **nonlinear dynamics**, and **design complexity**.
   - They highlighted how *unknown dynamics have always screwed soft robotics* due to the lack of accurate models for pneumatic muscles, and causality + delays make feedback controllers too slow requiring open loop + adaptive planning.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1446260835361231030)** (9 messages🔥): 

> `Bezos AI company, private video, arcprize withdrawn` 


- **Bezos Enters the AI Arena?**: Members pondered whether Bezos's new AI company will compete with Amazon, referencing [this YouTube video](https://www.youtube.com/watch?v=9A-eeJP0J7c) and [this Hacker News thread](https://news.ycombinator.com/item?id=46137548).
- **A Private Video?**: A user pointed out that [this video](https://youtu.be/Q4CBTckDAls?si=tyKN6MwBWITCqSaz) is private and cannot be viewed.
- **Arcprize Tweet Disappears**: Members noted that [this tweet](https://x.com/arcprize/status/1997010284490473497) was withdrawn.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1446243531499765886)** (24 messages🔥): 

> `DeepSeek v3.2 Transformers Implementation, Z Image Censorship, Hugging Face Space CPU Quota, Small LLM for Roblox, AI-Generated Music YouTube Channel` 


- **DeepSeek Transformers Implementation Stalls**: An implementation for the new **DeepSeek v3.2 model** in *transformers* is being worked on, although a [related PR](https://github.com/huggingface/transformers/pull/41251) shows stalled progress.
   - The original contributor seems to have abandoned the project, with no recent activity.
- **HF Space CPU Quota causes Pro Account Problems**: A user reported issues with **Hugging Face Space CPU quota limits**, even with a Pro account, causing inability to start or unpause Spaces.
   - The user expressed frustration over the lack of announcement for this change, as it led to unexpected service disruptions.
- **Censorship in Z Image Demo**: A user noticed that the **Z Image demo** censors explicit content, displaying a *"maybe not safe"* image, despite the model being advertised as uncensored.
   - It was suggested this may be due to self-imposed restrictions in the demo's code or Endpoint side, not present when used locally.
- **Seeking Tiny LLM for Roblox Integration**: A user is seeking a **small LLM (under 100M parameters)** to integrate into *Roblox*, facing challenges with Roblox's file size and RAM limitations.
   - They have made progress integrating tiny Ollama and Pythia models but need a more capable, yet compact, chatbot solution.
- **AI Generates Music Channel, Polarizes Audience**: A user announced the creation of a **YouTube channel** dedicated to **AI-generated music**, managed entirely by AI and their code.
   - The announcement triggered a polarized response, ranging from excitement and support to potential dislike.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1446608645369364480)** (1 messages): 

> `Inefficiency of Large Language Models, HRM and TRM models as alternatives, Compute Cost of LLMs, Environmental Impact of LLMs, Rising Costs Due to LLMs` 


- **LLMs Get Criticism for Being Inefficient**: A user argues against Large Language Models (**LLMs**) due to their inefficiency and high costs, citing excessive compute and environmental impacts.
   - The user claims that **LLMs** contribute to rising costs for storage, RAM, and GPUs, advocating for alternatives like HRM or TRM models.
- **HRM/TRM Models Challenging LLM Giants**: The user highlights **HRM** or **TRM** models (~27 million parameters) as superior alternatives to **LLMs** on certain benchmarks, providing links to research papers ([https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734) and [https://arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871)).
   - These models purportedly achieve better performance with significantly fewer parameters, challenging the necessity of massive model sizes.
- **LLMs accused of Insane Environmental Impact**: The user asserts that the environmental impact of **LLMs** is unacceptable due to excessive consumption of potable water, air pollution, and contribution to global warming.
   - The user did not provide a source for this claim, and it is not immediately verifiable.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1446500875043340399)** (3 messages): 

> `Anthropic Programmatic Tool Calling, Universal Programmatic Tool Calling, Model Agnostic Tool Orchestrator, Rhai Scripts for LLMs, Token Reduction in LLMs` 


- **Model Agnostic Tool Orchestrator Debuts**: A member introduced a **model-agnostic production-ready tool orchestrator** based on Anthropic's [Programmatic Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use).
   - This implementation allows any LLM to write **Rhai scripts** that efficiently orchestrate multiple tools, promising **97-99% token reduction** in benchmarks.
- **Universal Tool Calling Achieves Token Reduction**: The tool, dubbed **Universal Programmatic Tool Calling**, is model-agnostic and designed to implement Anthropic's [Programmatic Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use) pattern.
   - Instead of sequential tool calls consuming tokens, any **LLM writes Rhai scripts** that orchestrate multiple tools efficiently, achieving **97-99% token reduction** in benchmarks.
- **Tool Orchestrator on GitHub and YouTube**: The orchestrator works with any LLM, is sandboxed, and runs as native **Rust or WebAssembly** without external Python dependencies, under the MIT license, available on [GitHub](https://github.com/Brainwires/tool-orchestrator).
   - A [YouTube video](https://www.youtube.com/watch?v=b8yeQnP_ftw) and [LinkedIn post](https://www.linkedin.com/posts/eoinfr_buildinpublic-trading-fintech-activity-7402723589679960064-5oOT?utm_source=share&utm_medium=member_ios&rcm=ACoAACm7Z4cB_ZlAX5DjoA-4q-UXoclEX6TZepA) provide additional details.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/)** (1 messages): 

sky.moo: https://huggingface.co/blog/hf-skills-training
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1446469782214934588)** (2 messages): 

> `Agent Course Certificate` 


- **Agent Course Certificate Still Obtainable?**: A member inquired about the possibility of obtaining the **Agent Course Certificate** after completing and submitting the final assignment.
- **Lack of Clarification**: There was no response given to the question, leaving the certificate attainment status unclear.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1446237684761624617)** (24 messages🔥): 

> `Kimi for Coding Access, corporate policy reasons, LM Playground, 4x K2 turbo limit` 


- **Kimi for Coding Access Still Invite-Only?**: Users reported issues accessing **Kimi for Coding**, wondering if it was still invite-only and having trouble with both Option 1 and Option 2 access methods.
   - One user discovered they needed to [sign up for the kimi.com subscription](https://kimi.com) and then use the 'Kimi for Coding' link in subscription settings to unlock access.
- **Cloud Code and Roo Code**: A user inquired about the reasons behind **Kimi-for-coding** only supporting cloud code and roo code, and who to contact for more information.
   - In response, another user suggested that roo code is simply a fork of cline.
- **Calling All LM Tinkerers**: A user expressed a desire for a community focused on *fun experimentation* with LMs, rather than just commercial applications.
   - They emphasized the joy of testing and improving local models, lamenting that current LM chatbots are too boring, and suggested features like **quotation boxes** to enhance trust.
- **4x K2 Turbo Limit**: A user inquired about how the **4x K2 turbo limit** works on Moderato, forgetting to ask earlier, with the implication that they had already used their quota.
   - Another user suggested searching for the answer, adding [a link to an X post](https://x.com/Kimi_Moonshot/status/1996953835080966390) and indicating that *turbo is just faster*.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1446248218911571998)** (9 messages🔥): 

> `MCP Tokens, Tokenization, tiktoken, Claude 3 tokenizer` 


- **MCP Token Tooling Troubles**: A member is seeking recommendations for tools or methods to analyze **MCP token usage**, specifically after stripping unnecessary data from tool responses and condensing tool descriptions.
- **Tokenization tied to Models**: Tokenization depends on the model, so you'd need to choose some subset of models you're interested in and run your tools through the **tokenizers** you use.
- **tiktoken works for GPT models**: For **OpenAI**, you can use [tiktoken](https://github.com/openai/tiktoken) which should work for **GPT models**.
- **Anthropic only exposes count_tokens API**: For **Claude**, they only expose the [count_tokens API](https://platform.claude.com/docs/en/api/messages/count_tokens).
- **Anthropic no longer offers local tokenizer**: **Anthropic** used to vend a local tokenizer publicly but ever since **Claude 3** they changed their tokenizer and don't do that anymore, which is quite annoying to be honest.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1446263414757527574)** (7 messages): 

> `Ollama Timeout Errors, Claude Sonnet 4.5 Downgrade, Workflow Automation Engineer Introduction` 


- **Ollama Timeout Errors Plague Users**: A user reported experiencing timeout errors with **Ollama** when using models like **gpt-oss:120b** and **llama4:scout**, specifically `litellm.APIConnectionError: Ollama_chatException - litellm.Timeout: Connection timed out after 600.0 seconds.`
- **Claude Sonnet 4.5 Allegedly Dumbs Down**: A user stated that **Sonnet 4.5 (Claude code)** seemed to have become less intelligent in the recent days, suggesting a possible degradation in performance.
- **Full-Stack Engineer Specializing in Workflow Automation Intros Himself**: A full-stack engineer specializing in workflow automation, LLM integration, AI detection, and multimodal systems (**image + voice**) introduced himself to the channel.
- **Engineer touts Slack, Notion, and Internal APIs Automation**: An engineer detailed a cross-platform automation system leveraging **Slack, Notion, and internal APIs**, which reportedly reduced response times by **60%**.
- **RAG Expertise Highlighted**: The engineer also claimed to have built an advanced **RAG architecture** that leverages hybrid search, embeddings, and domain-based ranking to ensure accuracy and context stability during real-time deployment.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1446249051128463411)** (1 messages): 

> `aider on local LLMs, aider on Android, Cross-device coding with aider` 


- **New User Intends Aider Local LLM Setup**: A new user, Kalpit, wants to run **LLMs locally on their Mac** and use **aider** for coding on their Fold 6 (Android phone) within the same network.
   - They are seeking advice or experiences from others who have implemented a similar cross-device coding setup with **aider**.
- **User Transitions to Aider for Local LLM Use**: A user, formerly a frequent user of Cursor and Claude Code, expresses a desire to transition to **aider** to leverage local LLMs.
   - The user aims to run **aider** on a local Mac setup and remotely access it from a Fold 6 device, seeking guidance or shared experiences from the community.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

justanotheratom: https://x.com/realsanketp/status/1996978356227920345?s=20
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1446311848596213770)** (4 messages): 

> `DSPy support to Claude agents, GRPO algorithm in DSPy, Multi-turn conversations` 


- **DSPy extends reach to Claude Agents**: A member inquired about extending **DSPy** support to **Claude agents** or other popular agent SDKs.
   - Another member asked for clarification on the request, as the direction of support may influence the approach.
- **GRPO Algorithm Explored**: A new **DSPy** user asked about the **GRPO algorithm**, seeking insights into its performance and capability in handling **multi-turn conversations**.
   - The user is interested in real-world results and how well it manages context over multiple interactions.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1446484004147826753)** (4 messages): 

> `FSDP in tinygrad bounty, USBGPU on Raspberry Pi, USB transactions` 


- **FSDP Bounty Troubles Arise**: A member is working on the **FSDP bounty in tinygrad** and reports that *multi* is causing problems, unsure if the bounty allows changes in *multifor funsies*.
   - The user did not provide further information and is likely requesting help from the community.
- **Raspberry Pi Gets USBGPU Boost**: A member successfully ran **USBGPU on a Raspberry Pi 4**, after initial attempts on older models (2 and 3) failed due to architecture and stream allocation errors.
   - According to an [image analysis](https://cdn.discordapp.com/attachments/1068976834928193609/1446565913477251072/image.png?ex=693472fd&is=6933217d&hm=cfd6bfeb6ab892a212e18a7d559c4239e99cc5e034898ea6559b52495e6028aa), USB 2 might work slowly if driver support is added, potentially using **BULK** instead of streams.
- **USB Transactions Probed**: Discussion around **USB transactions** arises in the context of **USBGPU** implementation.
   - A user suggests that even full speed (**12Mbps**) would be supported, but they are *not that well versed in usb transactions*.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1446321569814155467)** (1 messages): 

> `struct.unpack GPU Implementation, tinygrad GPU Unpacking` 


- **GPU gets `struct.unpack`**: A member joked about implementing `struct.unpack('<4sIHHIIHH', f.read(24))` on the GPU using **tinygrad** instead of traditional methods.
   - The shared image visually highlights the complexity and potential performance gains of **GPU-based unpacking**.
- **tinygrad struct experiment**: There's an experiment to process binary data with tinygrad instead of `struct`
   - The discord member found it to be an amusing example.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1446431497669181471)** (1 messages): 

> `Workflow Automation, LLM Integration, RAG Pipelines, AI Content Detection, Image AI` 


- **AI Engineer Automates Workflows and Integrates LLMs**: An AI and full stack engineer with expertise in workflow automation, LLM integration, RAG, AI detection, image and voice AI offers their services.
   - They have a proven track record in real-world implementations, and they are open to collaboration or support.
- **Slack and Notion support automation by LLM**: The engineer built an automated pipeline and task orchestration system using **Dspy**, **OpenAI APIs**, and custom agents.
   - One example is the support automation system that connects **Slack**, **Notion**, and internal APIs to LLM, reducing response times by **60%**.
- **Advanced RAG Pipelines Deployed**: The engineer designed and deployed advanced RAG pipelines, combining vector database and graph database, hybrid search, and custom retrieval logic.
   - This resulted in accurate, context-aware responses in production environments.
- **AI Content Detection Tools Developed**: The engineer developed tools for a moderation platform using stylometric analysis, embedding similarity, and fine-tuned transformers.
   - These tools identify GPT-generated text with high precision.
- **Image AI Tagging and Moderation Pipeline**: The engineer created a tagging and moderation pipeline with **CLIP** and **YOLOv8** on **AWS Lambda** and **S3**.
   - This system classifies and filters thousands of images daily for an e-commerce platform.


  

---


---


---

