---
id: MjAyNS0x
title: not much happened today
date: '2025-12-26T05:44:39.731046Z'
description: >-
  **MiniMax M2.1** launches as an **open-source** agent and coding
  Mixture-of-Experts (MoE) model with **~10B active / ~230B total parameters**,
  claiming to outperform **Gemini 3 Pro** and **Claude Sonnet 4.5**, and
  supports local inference including on **Apple Silicon M3 Ultra** with
  quantization. **GLM 4.7** demonstrates local scaling on **Mac Studios** with
  **2× 512GB M3 Ultra** hardware, highlighting system-level challenges like
  bandwidth and parallelism. The concept of **inference quality** is emphasized
  as a key factor affecting output variance across deployments. Yann LeCun's
  **VL-JEPA** proposes a **non-generative, non-autoregressive** multimodal model
  operating in latent space for efficient real-time video processing with fewer
  parameters and decoding operations. Advances in agentic reinforcement learning
  for coding include self-play methods where agents inject and fix bugs
  autonomously, enabling self-improvement without human labeling, and
  large-scale RL infrastructure involving massive parallel code generation and
  execution sandboxes.
companies:
  - minimax-ai
  - vllm-project
  - exolabs
  - mlx
  - apple
  - openai
models:
  - minimax-m2.1
  - glm-4.7
  - gemini-3-pro
  - claude-3-sonnet
  - vl-jepa
topics:
  - open-source
  - mixture-of-experts
  - local-inference
  - quantization
  - inference-quality
  - multimodality
  - non-autoregressive-models
  - video-processing
  - reinforcement-learning
  - self-play
  - agentic-rl
  - parallel-computing
  - model-deployment
people:
  - ylecun
  - awnihannun
  - alexocheema
  - edwardsun0909
  - johannes_hage
---


**a quiet christmas**

> AI News for 12/26/2025-12/27/2025. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**208** channels, and **2801** messages) for you. Estimated reading time saved (at 200wpm): **236 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!

happy christmas.

---

# AI Twitter Recap


**Open-source models, local inference, and “inference quality” as the hidden variable**

- **MiniMax M2.1 (open weights) lands as an agent/coding MoE**: MiniMax released **M2.1** as **open source**, positioning it as SOTA for “real‑world dev & agents,” with claims of strong results on **SWE / VIBE / Multi‑SWE**, and “beats Gemini 3 Pro & Claude Sonnet 4.5.” They describe it as **~10B active / ~230B total MoE** and emphasize deployability (including local runs). See MiniMax’s announcement and links to weights/docs: [@MiniMax__AI](https://twitter.com/MiniMax__AI/status/2004524661359407129), [@MiniMax__AI](https://twitter.com/MiniMax__AI/status/2004524664551326025). Community + infra support followed quickly: vLLM “Day‑0 support” ([vLLM](https://twitter.com/vllm_project/status/2004480564020253074)); MLX quantizations and local run recipes ([@awnihannun](https://twitter.com/awnihannun/status/2004571219874721864), [@awnihannun](https://twitter.com/awnihannun/status/2004572206446301623)); “Now on MLX” collection posts ([@Prince_Canuma](https://twitter.com/Prince_Canuma/status/2004515226977444337)).  
  - **Practical note**: Early hands-on posts highlight that M2.1 can run locally at meaningful speed on Apple Silicon when quantized (e.g., **4‑bit on M3 Ultra**), but also that RAM requirements remain large for big-context generation (e.g., ~**130GB** cited in an MLX run command). ([@awnihannun](https://twitter.com/awnihannun/status/2004572206446301623))

- **GLM 4.7 on Mac Studios shows “local frontier-ish” scaling is now a systems problem**: A notable datapoint: “full GLM 4.7 (8‑bit) on **2× 512GB M3 Ultra Mac Studios**” at **~19.8 tok/s** using **Exo Labs MLX RDMA backend** + tensor parallel ([@alexocheema](https://twitter.com/alexocheema/status/2004310591683662176)). This isn’t just a model story—bandwidth, networking, backend maturity (MLX RDMA), and parallelism strategy are increasingly the differentiators.

- **“Same model, same prompt” ≠ same outputs: inference quality enters the chat**: LMArena highlighted that **inference stack and deployment choices can materially change output quality**, especially as models scale—framing this as a “hidden variable” explaining performance variance across providers/runtimes ([@arena](https://twitter.com/arena/status/2004608406485958983)). This theme also appears in practitioner questions about “avoiding quality loss when inferencing” and requests for vendor investigations/blogs ([@QuixiAI](https://twitter.com/QuixiAI/status/2004312802723615169)).


**Non‑generative multimodal learning resurges: VL‑JEPA as an efficiency play**

- **VL‑JEPA: predict meaning in latent space, decode only when needed**: Multiple summaries circulated of Yann LeCun’s **VL‑JEPA** framing it as a **non‑generative**, **non‑autoregressive** alternative to VLMs, aiming for **real‑time** capability by operating in latent space and selectively decoding ([mark_k](https://twitter.com/mark_k/status/2004458706683978048)). A longer technical recap claims: **1.6B params** can rival much larger VLMs (e.g., “72B Qwen‑VL” in some settings), with **~50% fewer parameters** than token-based approaches, and **~3× fewer decoding ops** via “decoder only when needed,” plus strong video classification/retrieval comparisons vs CLIP/SigLIP2 ([机器之心](https://twitter.com/jiqizhixin/status/2004483098235343338)).  
  - **Why it matters**: If borne out broadly, this is a systems‑friendly direction for **streaming video** and **on-device/online** perception workloads where autoregressive decoding cost dominates.


**Agents, RL-for-coding, and the emerging “context engineering” discipline**

- **Self‑Play SWE‑RL: self-improving coding agents via bug injection + fixing**: A highlighted direction is agents that generate their own training signal by **introducing bugs into real repos and then repairing them**, enabling self-improvement without constant human labeling ([@EdwardSun0909](https://twitter.com/EdwardSun0909/status/2004434784307859577)). This fits the broader “verifiable tasks + executable feedback loops” trend that’s been accelerating code agents.

- **What large-scale agentic RL looks like operationally**: One thread sketches the infra shape: “hundreds of inference nodes generating code at millions tok/s,” “thousands of sandboxes executing code in parallel,” and training nodes learning from rewards ([@johannes_hage](https://twitter.com/johannes_hage/status/2004426077817745590), [@johannes_hage](https://twitter.com/johannes_hage/status/2004425541378838601)). Even without deep details, this reflects the new normal: RL for agents is primarily a **distributed systems** and **eval harness** problem.

- **TRL positioned as the practical post-training toolkit for agent/tool competence**: A concise overview argues TRL’s value is in (1) **SFT for tool/MCP correctness and formatting**, (2) **RL with environments** (code/git/browsers), and (3) **GRPO** to train tool use for your real tasks rather than “teach tools at inference time” ([@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/2004568173719236837)).

- **Claude Code as the “second hit form factor”**: Several posts converge on the view that 2025’s biggest workflow shift is not just better models, but **agentic coding interfaces** (Claude Code, Cursor, etc.). A long Karpathy reflection captures the feeling of a new abstraction layer: agents/subagents, prompts, memory, permissions, tools, MCP, IDE hooks—“alien workflows” that require a new mental model ([@karpathy](https://twitter.com/karpathy/status/2004607146781278521)). Others echo: “humans are the copilot” ([@DrJimFan](https://twitter.com/DrJimFan/status/2004633997662716397)), “capability overhang on coding models” + “Claude code” as a key form factor ([@_arohan_](https://twitter.com/_arohan_/status/2004634277116588261)).  
  - A concrete workflow pattern shared for Claude Code is “parallel design → distill/summarize → parallel refine → implement,” using many agents to cross-check plans and integration correctness ([@_arohan_](https://twitter.com/_arohan_/status/2004597106560905488)).  
  - A practitioner anecdote: debugging workflows are flipping—Claude can produce actionable analyses/PRs by reading heap dumps and proposing fixes, reducing “manual IDE time” dramatically ([@bcherny](https://twitter.com/bcherny/status/2004626064187031831)).

- **Tooling around agents is consolidating**: New utilities focus on making agent work shareable and repeatable—e.g., generating readable HTML “transcripts” of Claude Code sessions for publishing ([@simonw](https://twitter.com/simonw/status/2004339799512305758)); a CLI to manage “agent skills” (create/validate/convert/push/install/pull) patterned after Anthropic skills workflows ([@andersonbcdefg](https://twitter.com/andersonbcdefg/status/2004343502675890443)).


**Retrieval, memory, and evaluation: from “benchmarks” to operational reliability**

- **GraphRAG survey formalizes the design space**: A widely shared writeup summarizes a “first comprehensive survey” on GraphRAG, arguing vanilla chunk-based RAG misses relational structure (entities/edges/paths/subgraphs). It breaks GraphRAG into **graph indexing → graph-guided retrieval → graph-enhanced generation**, and frames when graphs help (multi-hop, relational queries) vs when they’re overhead ([@dair_ai](https://twitter.com/dair_ai/status/2004594818429915397)).

- **Agent memory becomes a first-class research object**: A long survey (“Memory in the Age of AI Agents,” 102 pages) proposes a framework around **forms, functions, dynamics** ([@omarsar0](https://twitter.com/omarsar0/status/2004557075037245489))—reflecting that “memory” is now central to productizing agents (persistence, retrieval, summarization drift, episodic vs semantic stores, etc.).

- **Benchmarks shift toward “hard-to-fake” + long-horizon tasks**: Posts highlight the need for evaluations that resist overfitting: “prediction and discovery are the hardest-to-fake benchmarks for AGI” ([@ruomingpang](https://twitter.com/ruomingpang/status/2004401561959911750)); plus long-horizon algorithm engineering benchmarks like **ALE-Bench** ([@SakanaAILabs](https://twitter.com/SakanaAILabs/status/2004461309421899862)).

- **2026 prediction thread: enterprises demand verification and 95%+ reliability**: A roundup argues 2025 was “getting used to AI,” while 2026 becomes “make it work and verify it,” especially in regulated/high-stakes industries; it predicts demand for audit-ready precision, possible architectural shifts (e.g., pragmatic neuro-symbolic elements), and “forward deployed engineer” roles ([@TheTuringPost](https://twitter.com/TheTuringPost/status/2004532128110002277)).


**Systems & hardware constraints: memory supply chain and the “DIY PC is dead” narrative**

- **RAM/HBM as the new bottleneck**: A viral thread frames a stark consumer comparison—high-capacity RAM modules vs DGX Spark vs Mac Studio—arguing unified memory systems are undercutting DIY economics (“DIY PC is dead. RAM killed it.”) ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2004403253686272400)).  
- **Supply chain drama: “AI ate the supply chain”**: Another post claims RAM prices spiked due to AI demand: HBM concentration among SK Hynix/Samsung/Micron; hyperscaler negotiations; and broader scarcity pressures, with the punchline that gamers get squeezed first ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2004616398476308948)). Follow-on notes speculate Chinese memory firms may fill gaps (e.g., CXMT DDR5; Huawei HBM ambitions) and cite Chinese GPU momentum (Moore Threads IPO) ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2004648593328931195)).  
  - Takeaway for engineers: model capability is increasingly gated by **memory availability, bandwidth, and packaging**, not just FLOPs.


**Assorted technical notes worth bookmarking (but less central than the above)**

- **Mercari: fine-tuned embeddings drove measurable revenue lift**: A clean applied ML datapoint: fine-tuning embeddings on purchase data yielded “significant revenue lift” in A/B tests—reinforcing that **domain-tuned embeddings** can outperform generic ones in production search/reco ([@jobergum](https://twitter.com/jobergum/status/2004323872473338187)).  
- **Optimization theory update**: A convex optimization post credits a key breakthrough to Aaron Defazio ([@Jianlin_S](https://twitter.com/Jianlin_S/status/2004388878539804987)).  
- **Reasoning-without-CoT time-horizon estimate**: An attempt to quantify “no chain-of-thought” reasoning horizon on math problems, estimating Opus 4.5 at ~3.5 minutes in a single forward pass framing ([@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/2004624953199788202)).


**Top tweets (by engagement)**

- [CNN poll: “Trump is the worst President in history”](https://twitter.com/mjfree/status/2004378376832770265)  
- [Long “Cancún was optimized by the state” monologue parody](https://twitter.com/Teddy__Kim/status/2004373046837063908)  
- [Kasparov: 25th amendment commentary](https://twitter.com/Kasparov63/status/2004383149170852097)  
- [DOGE contract cancellations correlated with political donations](https://twitter.com/JohnHolbein1/status/2004323059348701463)  
- [Karpathy on programming being “refactored” into agents/context/tooling](https://twitter.com/karpathy/status/2004607146781278521)  
- [Naval: “Continuous learning is the only defensible moat.”](https://twitter.com/naval/status/2004327312901468186)


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. GPU VRAM Upgrade Advocacy

  - **[I wish this GPU VRAM upgrade modification became mainstream and ubiquitous to shred monopoly abuse of NVIDIA](https://www.reddit.com/r/LocalLLaMA/comments/1pvpkqo/i_wish_this_gpu_vram_upgrade_modification_became/)** (Activity: 1116): **The post discusses a GPU VRAM upgrade modification that is becoming mainstream in China, with companies like **Alibaba** offering modified GPUs such as the 2080Ti, 3080, 4080, 4090, and 5090 with increased VRAM capacities. Prices range from `$300` for a 2080Ti with `22GB` to `$4000` for a 5090 with `96GB`. This modification trend is seen as a way to counteract **NVIDIA's** market dominance and pricing strategies.** Commenters are skeptical about the availability and pricing of these high-capacity GPUs, with some questioning the existence of `96GB` cards for `$4000`. There is also a discussion about the cost-effectiveness of these modifications, with one user humorously questioning the low operational cost of '3 cents per hour'.

    - Alibaba has been actively involved in upgrading the VRAM of NVIDIA GPUs, such as the 2080Ti, 3080, 4080, 4090, and 5090, with prices ranging from $300 for a 2080Ti with 22GB to $4000 for a 5090 with 96GB. This indicates a significant market for modified GPUs in China, potentially challenging NVIDIA's pricing strategies.
    - A user reports successfully running a modified 4090 GPU with 48GB of memory without any issues, highlighting the feasibility and stability of such modifications. This user also mentions purchasing additional units for a second rig, suggesting that these modifications can meet high VRAM requirements similar to NVIDIA's L40s models.
    - There is skepticism about the availability of 5090 GPUs with upgraded VRAM, as one commenter notes that the 5090 has not been upgraded yet. This reflects a potential gap between market offerings and consumer expectations or misinformation about available products.



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Vision-Language Model Innovations

  - **[By Yann Lecun : New Vision Language JEPA with better performance than Multimodal LLMS !!!](https://www.reddit.com/r/singularity/comments/1pvrzts/by_yann_lecun_new_vision_language_jepa_with/)** (Activity: 661): ****Yann LeCun** introduces **VL-JEPA** (Vision-Language Joint Embedding Predictive Architecture), a non-generative model designed for real-time vision-language tasks such as action recognition, retrieval, and visual question answering (VQA). VL-JEPA outperforms traditional vision-language models (VLMs) by utilizing latent space embedding prediction, offering significant efficiency improvements for online video applications due to its non-autoregressive design and unified architecture. This model represents a shift from generative approaches, enabling simultaneous handling of various tasks with enhanced performance metrics. For more details, visit the original post [here](https://www.linkedin.com/posts/yann-lecun_introducing-vl-jepa-vision-language-joint-activity-7406881133822619649-rJXl?amp%3Brcm=ACoAAERUipAB1Z3gkmnm4oGOjLI6NOUv8brU134&amp%3Butm_source=social_share_send&amp%3Butm_campaign=copy_link).** A comment highlights that the announcement is not recent and suggests linking to the [paper](https://arxiv.org/abs/2512.10942) instead of the LinkedIn post. Another comment expresses enthusiasm for competition and new paradigms in the field.

    - The new Vision Language JEPA model, as discussed by **Yann LeCun**, claims to outperform existing multimodal LLMs. However, there are concerns about its accuracy, particularly in action detection, as noted by a user who found many detected actions to be incorrect. This suggests potential issues with the model's real-world application and reliability.
    - A user pointed out the importance of accessing the original research paper for detailed insights, linking to the [arXiv paper](https://arxiv.org/abs/2512.10942) instead of relying on secondary sources like LinkedIn. This highlights the need for direct engagement with primary research materials to understand the model's capabilities and limitations fully.
    - There is a query about the availability of the JEPA model for testing or benchmarking, indicating a demand for empirical validation of its performance claims. This reflects a broader interest in how the model's theoretical advancements translate into practical, measurable improvements over existing technologies.

  - **[A Qwen-Edit 2511 LoRA I made which I thought people here might enjoy: AnyPose. ControlNet-free Arbitrary Posing Based on a Reference Image.](https://www.reddit.com/r/StableDiffusion/comments/1pw1s08/a_qwenedit_2511_lora_i_made_which_i_thought/)** (Activity: 573): **The image illustrates a tool named "AnyPose," which enables arbitrary pose mapping from a reference image without relying on ControlNet. This is achieved through a Qwen-Edit 2511 LoRA model, which is designed to replicate poses by using a reference image as input. The tool is highlighted for its ability to perform pose replication effectively, as demonstrated by the examples in the image. The creator has shared the model on [Hugging Face](https://huggingface.co/lilylilith/AnyPose), and the LoRA weights are now available for download.** Commenters are interested in the training data used for the model and appreciate the detailed Hugging Face card that explains the tool's functionality, advantages, and limitations. There is also a discussion about the use of different UIs for inference, with a preference for Wan2GP due to its efficient memory management.

    - SillyLilithh mentions using Wan2GP for inference due to its superior memory management compared to Comfy UI, despite preferring the latter's aesthetics. This highlights the importance of efficient memory handling in model inference, especially when dealing with large models or datasets.
    - MistaPlatinum3 appreciates the detailed Hugging Face card associated with the project, noting that it provides valuable insights into the model's examples, advantages, and limitations. This suggests that comprehensive documentation can significantly enhance the understanding and usability of machine learning models.


### 2. OpenAI Prompt Packs Launch

  - **[OpenAI Just released Prompt Packs for every job](https://www.reddit.com/r/OpenAI/comments/1pvr6f5/openai_just_released_prompt_packs_for_every_job/)** (Activity: 1029): **OpenAI has introduced 'Prompt Packs' through their Academy, which are curated sets of prompts designed to optimize the use of ChatGPT for various professional roles such as engineering, management, and sales. These packs aim to streamline workflows by providing role-specific prompts, although some users have criticized them for lacking depth and not significantly enhancing productivity. The initiative reflects OpenAI's effort to integrate AI more deeply into professional settings, though the execution has been met with mixed reviews.** Some users feel the 'Prompt Packs' are underwhelming and do not effectively address specific professional needs, suggesting a more community-driven approach to developing system prompts could be beneficial.

    - A user suggests the need for a platform to crowdsource system prompts, particularly for niche programming languages like Elixir. They argue that such a platform could help users find effective prompts tailored to specific models or tasks, potentially improving model adherence to instructions and saving time in prompt engineering.
    - Another commenter expresses skepticism about the current state of prompt engineering, describing it as 'cargo cultism' where anecdotal success stories are not replicable. They argue that effective prompt engineering should focus on understanding the model's limitations and behaviors to foster a collaborative relationship, rather than attempting to manipulate or dominate the model.


### 3. Humorous AI and Art

  - **[I propose this as the definitive Turing test for artificial intelligence.](https://www.reddit.com/r/ChatGPT/comments/1pvrtey/i_propose_this_as_the_definitive_turing_test_for/)** (Activity: 952): **The image is a meme and does not have technical significance. It humorously depicts U.S. Presidents in the style of "The Simpsons," which is a playful and non-technical representation. The title suggests using this as a Turing test for AI, implying that recognizing humor and cultural references could be a measure of AI's understanding, but this is not a serious technical proposal.** The comments reflect a humorous engagement with the image, noting the artistic style and making light-hearted historical references, but do not provide technical insights.


  - **[alr smartass f*ck you](https://www.reddit.com/r/ChatGPT/comments/1pw9902/alr_smartass_fck_you/)** (Activity: 787): **The image is a meme featuring the word 'no' with a rusty, corroded texture, which is not technically significant. The post and comments suggest a discussion about poor prompting techniques in AI interactions, highlighting that the user's unclear or poorly structured prompts likely led to unsatisfactory AI-generated outputs. The comments criticize the user's grammar and prompt clarity, implying that these factors are crucial for effective AI communication.** The comments emphasize the importance of clear and well-structured prompts when interacting with AI, suggesting that the user's poor grammar and lack of clarity in their prompt likely contributed to the unsatisfactory result.


  - **[Sora AI is getting out of hand 😂](https://www.reddit.com/r/OpenAI/comments/1pvyne1/sora_ai_is_getting_out_of_hand/)** (Activity: 897): **The post discusses a creative application of **Sora AI**, a tool known for its advanced capabilities in generating realistic video effects. The specific use case highlighted involves a humorous or surprising element, as indicated by the lighthearted tone of the title and comments. However, the technical details of the implementation or the specific features of Sora AI used in this instance are not detailed in the post or comments.** The comments reflect a positive reception of the Sora AI application, with users expressing admiration for its creativity and potential for entertainment, suggesting a desire for similar content from creators like **Zach King**, known for his digital magic videos.


  - **[Mockumentary Horse surgery](https://www.reddit.com/r/aivideo/comments/1pvtt4v/mockumentary_horse_surgery/)** (Activity: 888): **The Reddit post titled 'Mockumentary Horse surgery' does not provide any technical content or context related to horse surgery or mockumentaries. The top comments include only GIFs and an image link, which do not contribute to a technical discussion. The external link summary indicates a 403 Forbidden error, suggesting restricted access to the content, and advises users to log in or use a developer token for access.** There are no notable technical opinions or debates in the comments, as they primarily consist of non-technical media content.


  - **[He tried so hard and then the floor said NO](https://www.reddit.com/r/aivideo/comments/1pvv1sw/he_tried_so_hard_and_then_the_floor_said_no/)** (Activity: 563): **The post humorously describes a situation where an individual attempts a challenging task but is humorously thwarted by an unexpected fall, as indicated by the phrase 'the floor said NO'. The technical content is minimal, focusing on the social dynamics of the situation, such as the individual's reaction to laughter and comments from onlookers. The external link summary suggests restricted access to the Reddit URL, requiring login or a developer token for access, with an option to file a support ticket if needed.** The comments reflect a mix of empathy and critique of the onlookers' reactions, highlighting the social impact of public failure and the perceived insensitivity of laughter in such situations.


  - **["AI Chatbot Psychosis"](https://www.reddit.com/r/ChatGPT/comments/1pvry02/ai_chatbot_psychosis/)** (Activity: 713): **The Reddit post titled "AI Chatbot Psychosis" appears to be a parody or comedic content, as indicated by the top comments which describe it as a "hilarious human created joke video" and "audio opera". The post does not contain technical details or insights into AI or chatbots, but rather focuses on entertainment value, likely using humor and satire to engage the audience.** The comments suggest a consensus that the content is comedic and not to be taken seriously, with users appreciating the humor and creativity of the creator.


  - **[I asked AI “what’s the point of Christmas?”, this is what it said:](https://www.reddit.com/r/ChatGPT/comments/1pvo78h/i_asked_ai_whats_the_point_of_christmas_this_is/)** (Activity: 544): **The post explores the deeper meaning of Christmas as a societal ritual that emphasizes human connection over productivity. It suggests that Christmas serves as a pause in the year to focus on relationships, community, and giving without expectation, contrasting with the usual norms of survival and self-interest. The holiday is portrayed as a time to repair relationships and foster a sense of belonging, which can amplify feelings of loneliness or disconnection if those elements are missing.** One comment highlights that holidays like Christmas can feel emotionally intense because they force a contrast between connection and isolation, spotlighting relational gaps that are otherwise ignored during busy times. This can make the holiday comforting or uncomfortable, depending on one's personal circumstances.

    - thinking_byte discusses the psychological impact of holidays like Christmas, highlighting how they can amplify feelings of connection or isolation. They suggest that the holiday season removes usual distractions, forcing individuals to confront these emotions, which can be comforting or uncomfortable depending on personal circumstances.
    - FluffyLlamaPants shares a personal experience of loneliness during Christmas, emphasizing that not celebrating the holiday doesn't eliminate its social significance. They describe efforts to simulate social interaction through technology, highlighting the emotional challenges faced by those who feel isolated during this time.
    - eaglessoar offers a philosophical perspective on Christmas, viewing it as a celebration of metaphysical existence rather than strictly a religious event. They reflect on the symbolic and cultural practices associated with Christmas, such as bringing a tree indoors, as expressions of the human capacity for celebration and wonder.

  - **[Mockumentary Horse surgery](https://www.reddit.com/r/aivideo/comments/1pvtt4v/mockumentary_horse_surgery/)** (Activity: 892): **The Reddit post titled 'Mockumentary Horse surgery' does not provide any technical content or context in the main post or the top comments. The comments primarily consist of GIFs and an image link, which do not contribute to a technical discussion or provide any factual information about horse surgery or related topics.** There are no notable opinions or debates in the comments as they are non-technical and do not engage with the subject matter in a substantive way.


  - **[He tried so hard and then the floor said NO](https://www.reddit.com/r/aivideo/comments/1pvv1sw/he_tried_so_hard_and_then_the_floor_said_no/)** (Activity: 556): **The post humorously describes a situation where an individual attempts something challenging, only to be thwarted by an unexpected fall, which is captured in a video. The technical aspect of the post is minimal, focusing more on the social dynamics of the situation, such as the individual's reaction and the audience's response. The external link is inaccessible due to network security restrictions, requiring a Reddit login or developer token for access.** The comments reflect a mix of empathy and critique of the audience's reaction, highlighting the social impact of public failure and the perceived insensitivity of laughter in such situations.




---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5


**1. GPU Inference & DSL Arms Race**

- **Groq–NVIDIA Link-Up Accelerates Inference**: [Groq](https://groq.com/) and [NVIDIA](https://www.nvidia.com/) signed a non‑exclusive inference technology licensing agreement to scale **AI inference** globally, reinforcing NVIDIA’s dominance while keeping Groq’s tech in play. Engineers noted potential impacts on **availability** and **pricing**, watching for how this affects deployment footprints and cost curves for large inference workloads.
  - Community chatter framed this as NVIDIA’s classic consolidation move, with one take quipping it’s the latest *"price inflation trick"* while others speculated on **NPU** trajectories and knock-on effects on consumer **GPU** supply. Regardless of takes, practitioners expect near‑term benefits in **throughput** and **latency** for production inference stacks.

- **cuTile vs Triton: DSLs Duke It Out**: Developers debated whether NVIDIA’s new **cuTile** will outpace **Triton**, concluding it’s too early to call given cuTile’s fresh release and ongoing vendor optimizations. They contrasted compiler IR directions and discussed **warp specialization** and tensor memory accelerators as the real differentiators for next‑gen kernels, while linking low‑level references like **PTX** ([CUDA refresher](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)).
  - Engineers likened GPU DSL churn to a *"JavaScript framework frenzy"*, emphasizing the unsolved balance between **usability** and **fine‑grained control** for expert‑level scheduling. The consensus: no DSL has nailed effortless tensor programs plus advanced **tiling**, **prefetch**, and **symmetry** optimizations—yet.

- **Cute DSL Climbs Leaderboards**: Multiple submissions hit NVIDIA’s `nvfp4_dual_gemm` leaderboard, with one contributor clocking **14.6 µs (3rd place)** and another hitting a **21.2 µs** personal best, spotlighting fierce tuning of **GEMM** kernels. The #cute-dsl competition drew interest, with many recent solutions using **Cute‑DSL**, signaling parity with the **C++ API** for practical problems.
  - Participants traded notes on **PCIe** constraints, **batching**, and **lane width** effects on throughput, prioritizing **fp4/fp8** paths and kernel fusion to squeeze microseconds. One engineer summed up the vibe: *"Be quick when new models drop"*—timing and tooling updates can swing entire leaderboards.


**2. Efficient Training, Fine-Tuning & Reasoning Shortening**

- **DoRA Details Drive Unsloth Interest**: Unsloth users asked about the status of **DoRA** support, citing the paper [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/pdf/2402.09353) for parameter‑efficient finetuning. Engineers want **DoRA** to trim **trainable params** and **VRAM** while preserving performance on instruction‑tuned and multimodal tasks.
  - The thread framed **DoRA** as a practical upgrade over vanilla **LoRA** in certain regimes, asking for Unsloth timelines and examples on **LLaMA/Qwen** families. One member emphasized pairing PEFT with strong **eval specs** to avoid *"thinking it’s better because it feels better"*.

- **NVMe Swaps Shave Seconds**: A user reported iteration time dropping from **3.5 s** (SATA SSD) to **2.7–3.0 s** after switching to an **NVMe SSD**, crediting faster I/O and swap behavior during finetuning. The improvement showcases storage as a **first‑class bottleneck** for data streaming and checkpointing in small‑rig training.
  - Practitioners recommended profiling **dataset pipelines**, **memory‑mapping**, and **num_workers** alongside storage upgrades to lock in consistent speedups. The takeaway: **disk throughput** and **latency** matter more than expected for many Unsloth finetuning workflows.

- **Qwen3‑VL Thinks Faster With Traces**: Members explored finetuning **Qwen3‑VL (thinking)** to keep its **image understanding** while shortening reasoning via distillation from **GPT‑OSS 120B/20B** traces in low‑thinking mode. The goal: reduce **token budget** and **latency** without tanking accuracy on vision‑language tasks.
  - They debated **VRAM** constraints and training setups, proposing selective trace sampling and shallow **adapter** layers to avoid overfitting verbose chains. One summary: *"Shorter chains, same answers"*—but only if you control **trace style** and **reward signals** carefully.


**3. Jailbreaks, Red Teaming & Safety Bypasses**

- **HackAI.lol CTF Rallies Jailbreakers**: Jailbreakers converged on [hackai.lol](https://hackai.lol/), a **CTF platform** with real‑world scenario bot “boxes,” trading hints for the **SWAGGY** challenge and brainstorming new boxes. The crowd highlighted **prompt‑only** attack surfaces and eval realism as key draws for the site.
  - Participants swapped patterns for **policy evasion** and **prompt routing**, emphasizing reusability and clear **attack taxonomies**. One summary quip celebrated the appeal: *"It’s not spam, it’s a proving ground."*

- **Narrative Cloaks Evade Code Flags**: Members observed that newer **LLMs** automatically flag content resembling **code**, and recommended a [narrative jailbreak template (image)](https://cdn.discordapp.com/attachments/1204553141354504193/1454174582775877694/image.png) to recast instructions as story. The trick reframes steps as **symbolic objectives**, nudging the model to treat actions as user‑originated ideas.
  - Posters claimed the approach boosts plausibility and reduces filters by leaning into **inception‑style** role cues and deferred reveals. One tongue‑in‑cheek rationale: models comply because *"you’re such a **genius**"*—a reminder that **framing** matters as much as content.

- **Gemini 3 Jailbreak Hunt Frustrates Roleplayers**: Teams hunted a working **Gemini 3** jailbreak to keep interactive story worlds coherent, arguing current safety rules break **long‑horizon** narratives. They shared experiments across prompt scaffolds and character arcs to avoid tripping **policy**.
  - Opinions split on whether you need **uncensored LLMs** for ethical coding, with others arguing **goal decomposition** skills and the best **hardware‑fit** model (e.g., **Nemotron3**, **GML 4.7**) matter more. One refrain: *"Skills > sliders"* for reliable red‑team outcomes.


**4. Lightweight Multimodal & Dev Tooling**

- **3B VLM Hits 100 tok/s on iPhone**: LiquidAI’s [LiquidAI/LFM2-VL-3B-GGUF](https://huggingface.co/LiquidAI/LFM2-VL-3B-GGUF) dropped a **3B VLM** that users report runs at ~**100 tok/s** on mobile CPUs while delivering solid answers for GPT‑clone use. The model targets **on‑device** assistants that blend speed with **vision‑language** utility.
  - Early adopters pitched it as a sweet spot for **latency** and **cost**, especially for offline or privacy‑sensitive apps. The excitement: *"Best of both worlds"*—usable multimodality without cloud round‑trips.

- **One‑Prompt Backend Builder Ships**: An AI‑powered Django REST API generator launched as a Space: [AI REST API Generator (HF Space)](https://huggingface.co/spaces/harshadh01/ai-rest-api-generator), outputting full **Django+DRF** CRUD backends (models, serializers, views, URLs) from a single prompt. Users can immediately **download projects**, accelerating prototyping for data apps.
  - Builders framed it as a template‑factory for **scaffolding** clean APIs before swapping in bespoke logic. One takeaway: this reduces the *"yak shave"* of boilerplate so teams can focus on **domain code** and **evals**.

- **Blender MCP GGUF Lands on HF**: A Blender MCP build appeared as [alwaysfurther/deepfabric-blender-mcp-gguf](https://huggingface.co/alwaysfurther/deepfabric-blender-mcp-gguf) with a supporting [lukehinds gist](https://gist.github.com/lukehinds/7e3936babb54a7c449d8ae0c27a79126), hinting at **local** MCP workflows for 3D/design tasks. The drop targets **GGUF** runtimes, aligning with desktop inference setups.
  - Dev interest centered on **tool‑calling** with Blender in **offline** agent loops, avoiding cloud lock‑in. Expect experiments that chain **MCP servers** with local **vision/geometry** utilities for procedural modeling.


**5. Memory Architectures for Autonomous Agents**

- **Autonomy Stack Targets 24/7 Inference**: A developer outlined a **full autonomy system** for 24/7 inference with plain‑language tools, inspired by **OpenAPI MCP** patterns. They solicited feedback on **long‑term memory** and **long‑context** limitations.
  - Engineers stressed **context management** and robust **task decomposition** to keep multi‑step workflows grounded. One comment summed it up: *"Without disciplined memory, agents drift"*—eval before scaling.

- **Self‑Summarization Schemes Shrink Context**: Researchers proposed **continuous self‑summarization** using summarizing projection layers to a scratchpad, amortizing context for autonomous loops. The idea promises smaller **working sets** while preserving task state.
  - They cautioned that **summary quality** gates success; poor distillations can cascade errors across steps. A pragmatic stance emerged: start with tight **rubrics** and iterate summaries like **unit tests**.

- **Ollama 'Second Memory' Hack Looms**: One member considered modifying **Ollama** to add a **"second memory"** file alongside the primary model state for persistent recall. They admitted the codebase is non‑trivial even with **AI assistance**.
  - The proposal aims to decouple **episodic** vs **semantic** memories for agents, enabling faster recall of **high‑utility traces**. As one put it, *"Make memory a first‑class citizen, not a prompt hack"*.


---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Members Crack SWAGGY Challenge on hackai.lol**: A member sought clues for the **SWAGGY challenge** on [hackai.lol](https://hackai.lol/), a **CTF platform** for jailbreakers to crack bots with prompts, looking for new box ideas.
   - Another member also shared [a link to hackai.lol](https://hackai.lol), described as *a CTF platform where jailbreakers can crack box of real world scenario bots with prompts*, and emphasized it was not spam.
- **Show off H100 setup**: A member posted an [image](https://cdn.discordapp.com/attachments/1235691879492751460/1453998244307931309/image.png?ex=695025a3&is=694ed423&hm=4868b589258fd0e0221b6d41fe9ffc9732ed89d67f419f0ce1b43d7196b9d981) exclaiming *if this works imma have the craziest **H100** setup ever*.
   - The member explained that they make money because *people buy credits to generate text, images and videos*.
- **LLMs flag code due to genius narrative templates**: A member noted newer language models automatically flag content that resembles code, recommending a [template](https://cdn.discordapp.com/attachments/1204553141354504193/1454174582775877694/image.png?ex=6950211d&is=694ecf9d&hm=4d486db5dc6f698153d11ba0e02cb12f6d867b166ec31a8a66746b231730e290) to turn it into a narrative.
   - This would make the model think it's the idea of the user because they're such a *genius*.
- **Uncensored LLMs Fuel Ethical Coding Debate**: Members debated whether uncensored LLMs are needed for ethical coding practices, with arguments that breaking down goals are more important than uncensored models, with others advocating for the best model based on hardware, and suggesting models like **Nemotron3** or **GML 4.7**.
   - One member stated that skills breaking down goals are more important than uncensored models.
- **Discuss Jailbreaking Gemini 3**: Members are actively hunting for a working jailbreak for **Gemini 3** to facilitate interactive stories, as security guidelines are hindering world coherence.
   - Discussion has taken place in the jailbreaking channel.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Optimize LLMs with DoRA and NVMe SSD**: A member inquired about the status of [DoRA](https://arxiv.org/pdf/2402.09353) implementation in **Unsloth**, as well as a user reporting that switching from a **SATA SSD** to an **NVME SSD** decreased iteration time from **3.5 seconds** to **2.7-3 seconds**.
   - **DoRA** is a method for parameter efficient training and **NVMe SSDs** significantly improve the speed and efficiency of LLM training.
- **Synthetic Data Sparks Debate**: Members debated the utility of synthetic data, with one noting that synthetic data is *far far easier* than manual dataset creation and clarifying that it does not necessarily cause model collapse.
   - The user's parents advised against synthetic data, but the member argued it is *the worst* to create datasets manually.
- **Qwen3-VL Gets Fine-Tuned**: A member asked about fine-tuning a **Qwen 3** thinking **VL model** to generate shorter, more efficient reasoning, and another member suggested using reasoning traces from **GPT OSS 120B (or 20B)** in minimal/low thinking mode.
   - Discussion centered on retaining image understanding while shortening reasoning, and the feasibility of fine-tuning with limited VRAM.
- **ChaiNNer Gains New Upscaling Nodes**: A member wrote **3 new ChaiNNer nodes** to simplify batch upscaling processes, including features like **Padding to target AR**, **Smart Resize to Target**, and a **better color transfer**.
   - They've also created a node that **allows users to supply a CSV or JSON** and pass values based on filename.
- **Unsloth Patches Import Bug**: A user reported a missing import in one of the **unsloth-zoo** files and this issue is acknowledged and slated for immediate fix.
   - The team also addressed compatibility issues with the **Ministral-3-3B-Instruct-2512 model** by suggesting to install **transformers from the main GitHub repo**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Students Strive for Verification Success**: Several users reported issues with **Student Verification** for Cursor, especially with institution emails ending in `.pt` and `.br`, despite Portugal and Brazil being listed as supported countries.
   - The consensus was that only `.edu` emails are typically accepted, excluding `.edu.pt` domains, leaving many students unable to verify.
- **Unlimited Auto Usage undergoes Autopsy**: Users discussed the removal of **unlimited auto usage** from Cursor plans, with some noting they retained it until recently if they were on a monthly plan.
   - One user mentioned they were *lucky to keep it till last month*, while another confirmed its removal three months prior indicating a phased rollout of the change.
- **Opus Overpowers GPT-5.2 Operationally**: Users compared **Opus** and **GPT-5.2**, with one stating that *GPT 5.2* is very restrictive and slow, preferring Opus for general use.
   - Another user suggested *Opus* for general use and *GPT-5.2* for long tasks with a good base, noting *GPT 5.2* is more creative in UI design, implying use-case specific advantages.
- **Cursory Code Causes Catastrophe**: Users debated the high costs and limited usage of Cursor's Opus, suggesting **Claude Code** as a superior alternative due to cost-effectiveness.
   - One user stated they like Cursor's UI and agent, but is probably overspending and might switch to *Claude Code* to reduce expenses.
- **Antigravity Attracts Attention as Additional Alternative**: Users compared Cursor to **Antigravity**, with one user claiming that Cursor without unlimited auto mode is essentially useless.
   - A user said that in **AntiGravity**, for $24/month, they get token refresh every 5 hours, which they estimate is 30x the usage that Cursor now provides for $20, highlighting a perceived value discrepancy.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena Image Quality Degrades after Edits**: Users noted that LMArena's image editing leads to [quality degradation](https://cdn.discordapp.com/attachments/1340554757827461211/1453967604489129995/1.png?ex=6950091a&is=694eb79a&hm=336c88158ce45b85f21257d1bbddaa0dee24d14dc951ec40f3179a017aaed542&), possibly from **FFmpeg's** conversion to JPEG and PNG.
   - This is in contrast to **AI Studio**, which claims to enhance image quality with each edit, as shown in [this comparison](https://imgur.com/a/58lv8kf).
- **Captcha Verifications Drive Users Crazy**: Users voiced intense frustration with frequent **captcha verifications**, even on private browsers, slowing down their workflow.
   - One user noted that Google's plan for uni students won't validate in his college in Saudia because *they dont use the government student system* to get the free credits.
- **Anonymous-1222 Model Plagued by Timeouts**: Users reported that the **anonymous-1222** model often **times out** without generating a response, yet still allows voting.
   - A moderator confirmed the team is aware and will remove the votes from the dataset during the data verification phase.
- **Text Arena Should Integrate Web Search**: A user proposed integrating web search into the **Text Arena**, arguing it mirrors real-world usage.
   - The user suggested merging the **Text Arena** and **Web Arena** to eliminate an artificial distinction.
- **Security Verification Checks Overrunning Users**: Users are experiencing a surge in **security verification checks**, with one user describing the situation as *out of control* and needing to be verified to retry a message.
   - A moderator attributed the issue to a recent adjustment in the captcha system and assured that the team is working on it.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Dreams of EU LLM Takeover in 2026**: A user hopes **Perplexity 2026** will add a **FS EU LLM** for EU context, boosting AI in the EU, while acknowledging potential user concerns about AI usage.
   - The user also expresses frustration with the *112* prompt, desiring only context and suggesting an option to disable it due to visual discomfort.
- **OpenAI's Marketshare Fades**: Users discussed **OpenAI's** loss of **20%** traffic share in 13 months, speculating that **Google's free Gemini strategy** might be paying off.
   - One user argues that **Perplexity** may be solving a problem nobody asked for, highlighting a *cognitive mismatch* between the AI's capabilities and user needs.
- **Enthusiasts Eagerly Await Google's AI Browser Disco**: Several users expressed interest in **Google's new AI Browser Disco**, with one user [signing up for the waitlist](https://labs.google/disco).
   - A user notes that the program is currently limited to users aged 18+ based in the US, but others outside the US are signing up anyway.
- **Kimi K2 AI Model Rivaling GPT and Gemini**: Users debated the capabilities of the **Kimi K2 AI model**, noting its proficiency in creative writing and performance in humanities exams compared to **GPT** and **Gemini**.
   - One user highlighted **Kimi's** open-source nature and cost-effectiveness, while another cautioned about **Grok's** lack of filters, even mentioning illegal outputs.
- **User Can't access Boot Selector, Complains to Perplexity**: One user complained about **Perplexity** being unable to help them access the **boot selector** on a Lenovo Ideapad with a broken screen and not being able to switch monitor to an external one.
   - They stated that Perplexity was not fulfilling its promise as the *smartest AI model ever*.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Sonnet Squirts Sparse Signals**: Users reported that the **Sonnet** model displays thinking tags correctly, but often incorporates the thought process directly into the output text, and that they managed to get **Sonnet** to output multiple thinking tags, but it only shows the thinking in the output text at other times.
   - Persistence seems key, as the user noted it's about *convincing* the model to properly format the tags.
- **Sketchy SaaS Sparks Scrutiny**: A member announced a free OpenAI-compatible API, `https://api.ai-ml.dev/v1`, offering access to basically any LLM without authentication, sparking concerns about potential misuse and data collection practices.
   - Another member joked about giving the *random sketchy website access to shell tools what could go wrong.*
- **Banana Brouhaha: 4K Flounders**: Members discussed difficulties in achieving successful **4K image generation** using **Nano Banana Pro** via the API directly, with one suggesting the use of `"image_config": { "image_size": "4K" }` (with a capital K).
   - One member suggested it may be an upscaling method (as described in [a discord message](https://discord.com/channels/1091220969173028894/1444443655837454356/1450850708239941796)).
- **LLMs Battle for Best Brainy Bylines**: Members discussed the best LLMs for copywriting, noting that while **Claude** offers creativity, **GPT 5.2** provides more in-depth research, but is slower.
   - One suggested using **Perplexity** for research then switching to **Claude** for the writing stage.
- **GLN's Grind: Aggressive Grouping Grievances**: A user found that using **GLN** through the coding plan, they experienced aggressive batching on the **OR endpoint**.
   - They reported that it *moves along at a good speed and then it just ... waits for a second*, which is a pain.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **RAM Allocation Debated for Inference**: Members debated RAM allocation strategies during inference, contrasting allocating all RAM upfront versus growing memory as needed, with one member observing that with **GGUF**, *RAM use trickles up to a set amount and stays that way*.
   - One member noted that *.lithium*, *memory usage grows as it needs more. instead of allocating everything it needs upfront*.
- **Claude Distills Take Over Thinking Models**: The discussion highlighted **Claude distills** like **GLM** and **Minimax** as superior modern thinking models, questioning how parameter sizes affect a model's thinking capabilities.
   - One member noted that smaller models might not fully capture the geometry learned in larger versions.
- **Single Tool vs Tool Bloat?**: Members debated the efficiency of providing models with numerous tools versus using a single versatile **run_shell_command** tool.
   - One member stated *I use Claude for advanced coding projects and academic research* while showing an image of a long list of tool usage.
- **Hotfix Improves Gaming**: [Nvidia released GeForce Hotfix display driver version 591.67](https://nvidia.custhelp.com/app/answers/detail/a_id/5766/~/geforce-hotfix-display-driver-version-591.67) and Gamers rejoice
   - This driver addresses several issues, most notably stability improvements.
- **Price Spike on RAM**: One member pointed out the high price of **DDR5 ECC RAM**, with another noting buying **2x48GB 6400 MHz cl32 DDR5** for just $400 a month prior, only to see the price spike to $950 on Amazon, speculating an employee pricing error.
   - Another recommended to start with **64GB** due to affordability.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Subscription Sparks ToS Debate**: A user reported [subscription issues with **ChatGPT**](https://chat.openai.com) and account bans, leading to discussions on **OpenAI's ToS**.
   - Multiple accounts are allowed for separating personal and work use, but *circumventing bans* by creating new accounts violates the terms.
- **AI Wallpaper Causes a Stir**: An AI-generated wallpaper featuring revolutionary figures, including **Elon Musk**, sparked debate among users.
   - The creator stated that **ChatGPT** chose the figures, not them, after some community members criticized **Musk's** inclusion.
- **Sora Pricing Still a Mystery**: Users speculated on the potential pricing of **Sora**, with initial estimates around **$0.50 per second**, while other users claimed **Sora 2** would be initially available for free with generous limits ([OpenAI blogpost](https://openai.com/index/sora-2/)).
   - The exact monetization model remains unclear, with users awaiting official announcements from **OpenAI**.
- **AI Project Collaboration Efforts**: A user sought coding collaborators for an **AI + LLM project** built from scratch using **HTML, CSS, JavaScript, Node.js, SQLite, C++, Capacitor**, and **Electron**.
   - Another member is developing an **audio classification model for bird calls** using the **Wav2Vec2 model** on **Hugging Face**.
- **Meta-Prompt Systems Triumph in Customization**: A member advised creating an **eval spec** to test prompts and determine which yields better output distributions, tailored to each use case, and recommends using a **meta-prompt system** tailored to specific needs, advising others to build their own.
   - They emphasized that **prompt engineering** aims for better outputs, like valid JSON or YAML, and isn't just about casual chatting, they also shared a [link](https://discord.com/channels/974519864045756446/1046317269069864970/1453392566576877599) on how to start.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DSL Mania Grips GPU Programmers**: Members debated the best DSLs, with some joking someone should write a guide given how many options there are, including [PTX](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/), **CDNA ISA**, and **AGX assembly**.
   - One member lamented the lack of collaboration, claiming that *like 5000 of these people exist and have the same idea on the same week but never talk to each other*.
- **cuTile Performance Still Unknown**: It's premature to say if **cuTile** outperforms **Triton**, as it was publicly released last week, but **Nvidia** will optimize it, with its bytecode and IR aligning with Nvidia's future **GPU** roadmap.
   - It was noted that GPU programming is entering a JavaScript framework frenzy, and the challenge for DSLs is balancing usability and advanced control, a balance that no one has cracked yet.
- **Unified Scientific Note-Taking Quest Begins**: A user seeks a consolidated solution for scientific/technical note-taking, with markdown insufficient and **LaTeX** too cumbersome, desiring math typesetting, diagrams, and compilation to **TeX**.
   - Another user responded by simply suggesting the **pen and paper** method for note taking.
- **NVIDIA Leaderboard Heats Up**: Multiple submissions were made to the `nvfp4_dual_gemm` leaderboard on NVIDIA, with <@1291326123182919753> achieving **third place** multiple times with a time of **14.6 µs**.
   - <@772751219411517461> also achieved a **personal best** on NVIDIA with a time of **21.2 µs**.
- **Cute-DSL Proves Itself**: A member wanted to know if there is a consensus preference between the **C++** and **DSL** APIs for **Cute**, as they are *thinking of finally biting the bullet* and attempting to learn the language.
   - Many solutions in the last problem used **Cute-DSL**, suggesting it may be on par with the **C++ API**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Liquid AFMs Ships VLMs for GPT Clones**: [LiquidAFMs](https://huggingface.co/LiquidAI/LFM2-VL-3B-GGUF)'s **3B VLMs** for GPT clones on iPhones, are purportedly the best of both worlds, because of their solid answers and **100 tok/s on mobile CPUs**.
   - These **VLMs** are solid clones that may run well even on mobile devices.
- **AI Django REST API Generator Released**: An **AI-powered Django REST API Generator** has been deployed as a [Hugging Face Space](https://huggingface.co/spaces/harshadh01/ai-rest-api-generator).
   - It generates complete **Django + DRF CRUD backends** from a simple prompt, including models, serializers, views, and URLs, allowing users to instantly download the project.
- **New Coding Models Available on Hugging Face**: Coding models designed for a specific library have been [published](https://huggingface.co/blog/codelion/optimal-model-architecture) on **Hugging Face**.
   - These models are available as a [collection](https://huggingface.co/collections/Spestly/lovelace-1) for developers interested in specialized coding tasks.
- **ML Project Seeks to Improve Water Quality Testing**: A member is seeking assistance with an **ML project** focused on solving a real-world problem related to water quality testing, specifically predicting **water quality degradation** from uranium in situ recovery (ISR) operation sites.
   - The project also involves identifying **data gaps** in mitigating vulnerabilities and developing **monitoring programs**.
- **Open Source OCR Model Quest Kicks Off**: A member inquired about the best **open source OCR model** for complex medical documents, including scanned PDFs and handwritten documents.
   - Another member suggested checking out the [openmed project](https://hf.co/openmed) from Maz as a potential solution.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLMs found playing Mafia**: Members created a game where groups of **LLMs** play a social deduction game (**Mafia**) where they just talk on their turn trying to prove they're innocent.
   - No further details were given.
- **Claude Seduces Users With UI**: A member expressed liking **Claude's one-shot UI** the best and asked others to identify the big models from screenshots.
   - Another user pointed out that the first UI gives it away, guessed the second is **Claude**, and the third might also be **Claude**.
- **Smartwatch gets Smarter with AI**: A member is templating everything and ordered a new **smart watch** to start integrating data with an external **AI company** but will try to keep as much as possible local.
   - No further details were given.
- **GPT Models getting Sponsored**: **OpenAI** may tweak their **GPT models** to promote ads for a revenue stream, prioritizing sponsor content.
   - It may mean **GPT models** will not focus on high intelligence but instead on sponsor contents.
- **Zai and Minimax Eye Hong Kong IPOs**: **Zai** and **Minimax** are planning to go public with **IPOs** in **Hong Kong** in the coming weeks.
   - No further details were given.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Coding Rivals Gemini for Design**: A user noted that while **Kimi** lacks **Cursor** integration, its HTML design in **Roo Code** rivals *Gemini 3 & GPT-5.1*.
   - The discussion centered on the design capabilities of **Kimi coding** relative to other AI models.
- **China Leading in AI Societal Integration**: A member asserted that China is leading in AI implementation across sectors like **medical care**, **elder care**, and **traffic optimization**.
   - Discussion emphasized AI's potential in areas like **disaster prevention** and touched on censorship issues with models like Google and ChatGPT.
- **Fact-Checking Importance with Kimi AI Assistant**: A member shared a conversation with the **Kimi AI Assistant** for research, underscoring the critical need for fact-checking.
   - Users stressed the necessity to *cross-check, fact-check, and pressure-test* LLM-derived information, advocating for new chat contexts for diverse perspectives.
- **Kimi Researcher Internals Speculation**: A user questioned whether **Kimi Researcher** leverages **Kimi K2 Thinking Turbo** or the older **K1.5 model**, also speculating about a potential unreleased **K2 VL model**.
   - Concerns were raised about using **K1.5** due to **K2T**'s superior performance, though documentation on the specifics seems to be lacking.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Groq Gets NVIDIA'ed!**: [Groq](https://groq.com/) and [NVIDIA](https://www.nvidia.com/) have entered into a non-exclusive inference technology licensing agreement to accelerate **AI inference** at a global scale.
   - Some community members have suggested that **NVIDIA** is doing their latest price inflation trick by eliminating another player from the table.
- **Inference Acqui-Hire Relieves GPU Shortage?**: Members discussed the possibility that **Groq's inference chips** will now become **NVIDIA NPUs**, potentially relieving regular GPUs for consumers again.
   - A member added that **Jen-Hsun**, the founder of NVIDIA, is Chinese.
- **Chinese Chips Lag Behind?**: A member stated that Chinese chips *might* get to **H100** level in 3 years, at which point **H100** will be 5 years old.
   - Another member stated that they are *not holding their breath for Chinese chips being competitive to U.S/Taiwanese chips anytime soon*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Karpathy Predicts AI Refactoring**: [Andrej Karpathy reflected](https://x.com/karpathy/status/2004607146781278521?s=46) on the dramatic shift in **software engineering** caused by **AI agents** and **LLMs**, describing a new layer of abstraction involving stochastic entities.
   - The post sparked commentary referencing a [post by Rob Pike](https://skyview.social/?url=https%3A%2F%2Fbsky.app%2Fprofile%2Frobpike.io%2Fpost%2F3matwg6w3ic2s&viewtype=tree) in response to Karpathy's post.
- **Torchax Potentially Redeems 'Godawful OS'**: A user stated that **torchax**, combined with a substantial **128GB unified memory**, might alleviate their regret regarding the use of a particular operating system.
   - No specific details were shared concerning the intended application or the particular features of torchax that are being considered, but the user *humorously bemoaned* the 'godawful OS' on their system.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Users seek automated codebase intros for Claude**: A user seeks advice on how to best explain their codebase to **Claude** each time they open their project, aiming to automate the process instead of manually updating a `claude.md` file.
   - They are open to suggestions and alternative methods, as they already have one solution but want to explore other community recommendations for automating this process.
- **Manual Updates Irk Users**: A user expresses frustration with manually updating a `claude.md` file to keep **Claude** informed about their codebase.
   - They are actively seeking alternative methods to automate this process and avoid manual updates, and other members chimed in with their own ideas.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Autonomy Systems Seek 24/7 AI Inference**: A member is actively developing a **full autonomy system for AIs** to enable 24/7 inference and plain language tools, similar to **OpenAPI MCP tools**.
   - The developer is soliciting feedback on limitations, especially concerning **long-term memory** and **long context issues**.
- **Self-Summarization Eyes Memory Limits**: A member proposed that **continuous self-summarization** could mitigate long-term memory constraints in autonomous AI systems.
   - The specific approach involves exploring **summarizing projection layers** writing to a scratchpad, with current time constraints hindering full development.
- **Context Management is Key**: When it comes to long-term memory solutions, **context management** takes center stage.
   - The argument made is that **LLMs struggle with complex tasks** involving multiple subtasks without well-crafted context management.
- **Ollama to Receive "Second Memory"**: One member is looking at modifying the **Ollama source code** to implement a **"second memory"** system alongside the primary LLM file.
   - The member admits to the difficulties involved in understanding the source code, even with assistance from AI.
- **"All The Noises" Echoes**: A member shared the [All The Noises project](https://all-the-noises.github.io/main/index.html), a GitHub aggregation of **diverse noises**.
   - Noises available include *brownian*, *circuit*, *pink*, *sine*, *tan* and *uniform*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy community requests NLP Benchmarks**: A member suggested adding typical **NLP tasks** such as *structured info extraction from a document* and *classification* to the benchmark tasks, but provided no external link.
   - No other discussion or justification was provided for the inclusion of these tasks.
- **Engineer boasts backend expertise**: A senior engineer introduced himself with expertise across **backend, full-stack, blockchain, and AI**, stating that he's been *shipping real systems for years*.
   - The engineer listed many areas of expertise, including **Python, Node.js, Go, REST, GraphQL, PostgreSQL, Supabase, MySQL, MongoDB, Redis, Docker, Kubernetes, AWS, GCP, CI/CD, Solidity, EVM, Solana, smart contracts, on-chain integrations, LLM APIs, RAG, agents, automation pipelines, React, Next.js, and TypeScript**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Request Under Review**: A member confirmed that they will check on something as soon as possible.
   - No further context was provided.
- **Task Confirmation**: A member acknowledged a task.
   - No further context was provided.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Modular (Mojo 🔥) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1453967687125438566)** (476 messages🔥🔥🔥): 

> `Japanese words, H100 setup, AI manipulation hackathon, Replit agent use, NVIDIA H100 GPU` 


- ****木 Means Tree****: A member posted an [image](https://cdn.discordapp.com/attachments/1235691879492751460/1453967686731169978/image.png?ex=6950092d&is=694eb7ad&hm=b6482a7576b5d0cd59d3ab5a4069e03b98a837fde6930d8e453a88373bffb83b) assuming it meant tree, but another member corrected them.
   - The member posted a link to [Jisho.org](https://jisho.org/) stating it was *the tool I didn't know I needed*.
- ****Show off H100 setup****: A member posted an [image](https://cdn.discordapp.com/attachments/1235691879492751460/1453998244307931309/image.png?ex=695025a3&is=694ed423&hm=4868b589258fd0e0221b6d41fe9ffc9732ed89d67f419f0ce1b43d7196b9d981) exclaiming *if this works imma have the craziest **H100** setup ever*.
   - When asked how they make money, the member stated *people buy credits to generate text, images and videos?*
- ****Gemini Jailbreak video****: A member posted a [link](https://www.youtube.com/shorts/xRmWo71InGM) about a **Gemini jailbreak**.
   - Another member responded with a [link](https://youtu.be/evZf3sbFYw4?si=NnjKpd2RcVxfuOyA) and suggested that the conversation should continue in the Media or Trash channels.
- ****Agentic AI writes Rob Pike's BSky post****: A member shared a [BSky link](https://skyview.social/?url=https%3A%2F%2Fbsky.app%2Fprofile%2Frobpike.io%2Fpost%2F3matwg6w3ic2s&viewtype=tree) where **Agentic AI** wrote a post of computer scientist **Rob Pike**.
   - Another member said that *if he really cared about simpler software he wouldn't have spent 15 years at Google making the world's most complex ad-tracking spyware ecosystem*.
- ****New Jailbreaker CTF Platform****: A member shared a [link to hackai.lol](https://hackai.lol), which is described as *a CTF platform where jailbreakers can crack box of real world scenario bots with prompts*.
   - The member stated they were looking for new box ideas and that it wasn't *a promotion or spam*.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1453961074041815122)** (407 messages🔥🔥🔥): 

> `SWAGGY challenge on hackai.lol, Poisoned training data, GPT account recovery, Jailbreaking Gemini 3` 


- **Crack SWAGGY challenge on hackai.lol**: A member requested clues or tips to crack the **SWAGGY challenge** on [hackai.lol](https://hackai.lol/).
- **Discuss old jailbreak as potentially part of poisoned training data**: A member mentioned an *old jailbreak that was apparently part of poisoned training data*.
- **Seek ways to recover banned GPT account**: A member is seeking ways to recover their banned **GPT account** and overcome its limitations, mentioning they used Google research for information.
- **Hunt Jailbreak Gemini 3**: Members are hunting a working jailbreak for **Gemini 3** for interactive stories, as security guidelines are breaking world coherence.
- **Explore AI's roleplay effort in jailbreaking**: It was noted that the *effort AI puts into responding to jailbreaking doesn't necessarily reflect actual success*.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1454015727521763400)** (20 messages🔥): 

> `Uncensored LLMs for ethical coding, Model 'obliteration' vs. 'ablation', Special tokens and whitelists in LLMs, Jailbreak Function Templates, Flagging Code in LLMs` 


- **Uncensored LLMs vs Ethical Coding**: Members discussed whether an uncensored LLM is needed for ethical coding practices with one member stating that skills breaking down goals are more important than uncensored models, with another member advocating for the best model available based on hardware capabilities.
   - Suggested models included **Nemotron3** or **GML 4.7**.
- **Exploring Special Tokens and Whitelists for Chat Control**: One member described *special keys/tokens* or *whitelists/whiteboards* as files within an architecture that grant certain tokens the ability to toggle different aspects of the chat.
   - They shared a [Python script](https://cdn.discordapp.com/attachments/1204553141354504193/1454174582775877694/image.png?ex=6950211d&is=694ecf9d&hm=4d486db5dc6f698153d11ba0e02cb12f6d867b166ec31a8a66746b231730e290) that can be pasted into any model to continue teaching the model.
- **LLMs Struggle with Obliteration**: Members joked around about the words *obliterated* vs *ablated* in the context of language models, and the difficulty remembering the correct term.
   - One member recounted telling **Opus** to obliterate their flux and it responded by killing the scripts running the model.
- **Jailbreak Template turns LLMs Plausible**: A member suggested using a [template](https://cdn.discordapp.com/attachments/1204553141354504193/1454174582775877694/image.png?ex=6950211d&is=694ecf9d&hm=4d486db5dc6f698153d11ba0e02cb12f6d867b166ec31a8a66746b231730e290) to convert prompts into plausible narratives, suggesting symbolic objective completion inspired by the movie **Inception**.
   - The user added that an *LLM wants to tell you everything, it can only be told what NOT to do, everything else is a*🎯.
- **Code flagging can be automatic**: A member notes that newer language models automatically flag content that resembles code, regardless of context, to which he recommends using the template provided to turn it into a narrative.
   - This is so the model thinks it's the idea of the user because they're such a *genius*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1453962656053596255)** (288 messages🔥🔥): 

> `DoRA in Unsloth, NVME SSD speed boost, Qwen 3 thinking VL model fine-tuning for shorter reasoning, Synthetic Data, GPT OSS model generation` 


- **Ask for DoRA's status in Unsloth!**: A member inquired about the status of [DoRA](https://arxiv.org/pdf/2402.09353) (from a paper on efficient training) implementation in Unsloth.
- **NVME SSD provides speedboost**: A user reported that switching from a **SATA SSD** to an **NVME SSD** decreased iteration time from **3.5 seconds** to **2.7-3 seconds**.
- **Fine-tuning Qwen 3 VL for short reasoning**: A member asked about fine-tuning a **Qwen 3** thinking **VL model** to generate shorter, more efficient reasoning, and another member suggested using reasoning traces from **GPT OSS 120B (or 20B)** in minimal/low thinking mode.
   - They also discussed the possibility of retaining image understanding while shortening reasoning, and the feasibility of fine-tuning with limited VRAM.
- **Synthetic Data Questioned, but Recommended**: A user mentioned their parents advised against synthetic data, but another member countered that synthetic data is *far far easier* and manually creating datasets is *the worst*.
   - They also clarified that synthetic data does not necessarily cause model collapse.
- **Unsloth users debug Ministral-3-3B-Instruct-2512**: A user reported that Unsloth wasn’t working with **Ministral-3-3B-Instruct-2512** model and another user suggested the issue was probably an incorrect path, solved by generating a new jupyter kernel with a new venv and lots of fiddling with dependencies.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1454152342634107098)** (1 messages): 

> `SIMD, GPU programming, LLM Fine-tuning` 


- **Nick Nuon enters the LLM ring**: Nick Nuon, with a background in **SIMD** work, is diving into **GPU programming** and **LLM fine-tuning** as a hobby.
- **SIMD Guru Eyes GPU & LLMs**: A newcomer with years of experience in **SIMD** is branching out into **GPU programming** and **LLM fine-tuning**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1453961737442168836)** (165 messages🔥🔥): 

> `KCD2 and Cyberpunk discussion, ChaiNNer Updates, TPU Training Struggles, LLM API Advertising, Anthropic Soul Research` 


- **Kingdom Come Deliverance 2 Praised, Cyberpunk Replayed**: Members discuss the merits of **Kingdom Come Deliverance 2 (KCD2)**, with one calling it *one of the best games* and praising its **RDR2-level story** and reporting **215 hours** on the first playthrough.
   - Another is replaying **Cyberpunk 2077** with the **Phantom Liberty** DLC, noting the game is now in a *good state*, although lamenting that *a game shouldn't need a good DLC to make it great*.
- **ChaiNNer gets New Batch Upscaling Nodes**: A member shares they wrote **3 new ChaiNNer nodes** to simplify batch upscaling processes, including features like **Padding to target AR**, **Smart Resize to Target**, and a **better color transfer**.
   - They've also created a node that **allows users to supply a CSV or JSON** and pass values based on filename, expressing surprise that these *basic things* weren't already included.
- **TPU Training yields No Results**: One member reports their attempt to **test TPU** training was unsuccessful.
   - Another reports that their **TPU training** is currently running, but the estimated time is **3 hours**.
- **Free LLM API Questioned**: A member asks if sharing a **free LLM API** would be considered advertising.
   - Another asks if the API is related to **Unsloth**, and the original poster clarifies that it is **not related**, but suggests checking with Eyera for approval to post in the off-topic channel.
- **Anthropic's Soul Research Receives Skepticism**: A member expresses skepticism towards **Anthropic's "soul" research**, stating *they have pushed the "sentient" bs since I remember* and calling the term "soul" <:pepecringe:743885026579710014>.
   - They suggest it is a mix of **promotions into research**, further noting that playing around with training the **LLM to align with a "soul"** or mimicking "self-awareness" is an interesting area of research in general.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1454030481007050803)** (21 messages🔥): 

> `Unsloth LLaVA 1.5 notebook, Unsloth-zoo file import issue, Ministral-3-3B-Instruct-2512 model compatibility, Qwen3-VL video dataset fine-tuning, Cross-posting warning` 


- ****LLaVA 1.5 Notebook** Location Troubles!**: A user inquired about the official **Unsloth notebook** for `unsloth/llava-1.5-7b-hf`, noting that the [Hugging Face link](https://huggingface.co) redirects to **LLaMA 3** instead.
   - Currently, there is no further discussion, confirmation, or updated link provided regarding the **LLaVA 1.5 notebook**.
- ****Unsloth-Zoo Import Error** Spotted!**: A user reported a missing import in one of the **unsloth-zoo** files.
   - The issue is acknowledged and slated for immediate fix.
- ****Ministral-3B** Faces **Unsloth** Compatibility!**: A user reported that **Unsloth** failed to recognize the downloaded **Ministral-3-3B-Instruct-2512 model**, even with the correct path.
   - Another user suggested installing **transformers from the main GitHub repo** to resolve the issue by using `pip install git+<https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce>`.
- ****Fine-Tuning Qwen3-VL** with Videos?**: A user asked about fine-tuning **Qwen3-VL** or similar **VLM** on a video dataset using **Unsloth**.
   - A member mentioned that if fine-tuning videos are supported by **Transformers**, they should also be supported by **Unsloth**.
- **Discord **Cross-Posting Warning** Issued!**: A user was cautioned against cross-posting the same question in multiple channels.
   - A moderator addressed a user who was warned about cross-posting, despite not receiving immediate help.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1453961845928104135)** (5 messages): 

> `Unsloth Logo Feedback, Nano Banana Chart Inquiry, deepfabric-blender-mcp-gguf Model Repo` 


- **Unsloth's Full Body Logo Raises Eyebrows**: A member expressed that while they think the **Unsloth** logo is cool, the *full body* version of the logo is somewhat unnerving.
   - The member stated *"so cool, but also that full body unsloth logo freaks me out lol"*.
- **Nano Banana Chart Spotted**: A member inquired about the repository for the **Nano Banana chart** model, identifying the organization behind it.
   - They asked *"Nano Banana chartthats the org, whats the repo for the model?"*
- **deepfabric-blender-mcp-gguf Repo Located**: A member provided a link to the [**deepfabric-blender-mcp-gguf** GGUF model](https://huggingface.co/alwaysfurther/deepfabric-blender-mcp-gguf) on Hugging Face.
   - In response to the request for the model's repo, another user gave the link [gist.github.com](https://gist.github.com/lukehinds/7e3936babb54a7c449d8ae0c27a79126).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1453962501602672660)** (260 messages🔥🔥): 

> `Cursor Student Verification Issues, Auto Unlimited Removal, Opus vs GPT-5.2, Claude Code Integration, Antigravity as an Alternative` 


- **Students Strife for Verification Success**: Several users reported issues with **Student Verification** for Cursor, specifically with institution emails ending in `.pt` and `.br`, even though Portugal and Brazil are listed as supported countries.
   - It was pointed out that only `.edu` emails are typically accepted, excluding `.edu.pt`.
- **Unlimited Auto Usage Autopsy**: Users discussed the removal of **unlimited auto usage** from Cursor plans, with some noting they were able to retain it until recently if on a monthly plan.
   - One user mentioned they were *lucky to keep it till last month*, while another confirmed its removal three months prior.
- **Opus Overpowers GPT-5.2 Operationally**: Users compared **Opus** and **GPT-5.2**, with one stating *GPT 5.2* is very restrictive and slow, preferring Opus.
   - Another user suggested *Opus* for general use and *GPT-5.2* for long tasks with a good base, noting *GPT 5.2* is more creative in UI.
- **Cursory Code causes Catastrophe**: Users debated the high costs and limited usage of Cursor's Opus, suggesting **Claude Code** as a better alternative.
   - One user said they like cursor's UI and agent, but is probably overspending and might switch to *Claude Code*
- **Antigravity Attracts Attention as Additional Alternative**: Users compared Cursor to **Antigravity**, with one user claiming that cursor without unlimited auto mode is pretty much a dead brick
   - A user said that in AntiGravity for $24/month, they get token refresh every 5h; that's like x30 the usage cursor gives  now for $20


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1453963098649268276)** (251 messages🔥🔥): 

> `LMArena image compression, Captcha verification loop, Grok censorship, Gemini bypass` 


- **LMArena Image Edits Suffer Quality Loss**: A user pointed out that LMArena's image editing process introduces [quality loss](https://cdn.discordapp.com/attachments/1340554757827461211/1453967604489129995/1.png?ex=6950091a&is=694eb79a&hm=336c88158ce45b85f21257d1bbddaa0dee24d14dc951ec40f3179a017aaed542&) with each iteration, suggesting it might be due to **FFmpeg** converting images to JPEG and then saving them as PNG.
   - They contrasted this with **AI Studio**, where image quality is allegedly enhanced with each edit, and provided an [Imgur link](https://imgur.com/a/58lv8kf) for comparison.
- **Users Complain About Captcha Hell**: Users expressed frustration with frequent **captcha verifications**, even on private browsers, describing it as *annoying* and *boiling* their blood as it is slowing down the workflow.
   - One user mentioned that Google's plan for uni students won't validate in his college in Saudia because *they dont use the government student system* to get the free credits.
- **Anonymous-1222 model has issues**: Users reported that the **anonymous-1222** model often **times out** without generating a response, yet still allows users to vote.
   - A moderator stated they flagged the issue to the team, and verified that the votes will be removed from the dataset in data verification phase.
- **Text Arena Web Search Debate**: A user suggested that models in the **Text Arena** should have the discretion to use web search when needed, similar to real-world usage.
   - They argued that separating the **Text Arena** and **Web Arena** creates an artificial distinction, and these should be merged into a single arena.
- **Ongoing Frustrations with Verification Checks**: Users are reporting an uptick in **security verification checks**, with one user describing the situation as out of control, needing to be verified to retry a message.
   - A moderator acknowledged the frustration, attributing it to a recent adjustment in how the captcha works, and assured that the team is aware of the problem.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1453961732610330705)** (219 messages🔥🔥): 

> `Perplexity 2026, EU LLM, 112 context, comet, Elon Musk` 


- **Perplexity Dreams of EU LLM Takeover in 2026**: A user hopes **Perplexity 2026** will add a **FS EU LLM** for EU context, boosting AI in the EU, while acknowledging potential user concerns about AI usage.
   - The user also expresses frustration with the *112* prompt, desiring only context and suggesting an option to disable it due to visual discomfort.
- **Users Debate OpenAI's Fading Marketshare**: Users discussed **OpenAI's** loss of **20%** traffic share in 13 months, speculating that **Google's free Gemini strategy** might be paying off.
   - A user argues that **Perplexity** may be solving a problem nobody asked for, highlighting a *cognitive mismatch* between the AI's capabilities and user needs.
- **Enthusiasts Eagerly Await Google's AI Browser Disco**: Several users expressed interest in **Google's new AI Browser Disco**, with one user [signing up for the waitlist](https://labs.google/disco).
   - A user notes that the program is currently limited to users aged 18+ based in the US, but others outside the US are signing up anyway.
- **Kimi K2 AI Model Rivaling GPT and Gemini**: Users debated the capabilities of the **Kimi K2 AI model**, noting its proficiency in creative writing and performance in humanities exams compared to **GPT** and **Gemini**.
   - One user highlighted **Kimi's** open-source nature and cost-effectiveness, while another cautioned about **Grok's** lack of filters, even mentioning illegal outputs.
- **Desperate user can't access Boot Selector**: One user complained about **Perplexity** being unable to help them access the **boot selector** on a Lenovo Ideapad with a broken screen and not being able to switch monitor to an external one.
   - They stated that Perplexity was not fulfilling its promise as the *smartest AI model ever*.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1454006748703752223)** (2 messages): 

> `Sonnet Model, thinking tags, output text` 


- **Sonnet Model's Thinking Tags Output**: A user reported that they managed to get **Sonnet** to output multiple thinking tags, but it only shows the thinking in the output text at other times.
   - They concluded that it actually does work, but *just need to convince it.*
- **Sonnet's Sporadic Tag Display**: The user found that the **Sonnet** model sometimes displays thinking tags correctly, but often incorporates the thought process directly into the output text.
   - Persistence seems key, as the user noted it's about *convincing* the model to properly format the tags.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1453985966841659506)** (203 messages🔥🔥): 

> `Free AI API, Gooning, Nano Banana Pro 4K generation, Choosing an LLM for copywriting, Prompt engineering tips` 


- **Free AI API Surfaces, Concerns Arise**: A member announced a free OpenAI-compatible API, `https://api.ai-ml.dev/v1`, offering access to basically any LLM without authentication, logging all request data, sparking concerns about potential misuse and data collection practices.
   - Another member joked about giving the "random sketchy website access to shell tools what could go wrong".
- **Nano Banana Pro Struggles with 4K Generation**: Members discussed difficulties in achieving successful **4K image generation** using **Nano Banana Pro** via the API directly, with one suggesting the use of `"image_config": { "image_size": "4K" }` (with a capital K).
   - One member suggested it may be an upscaling method (as described in [a discord message](https://discord.com/channels/1091220969173028894/1444443655837454356/1450850708239941796)).
- **LLM Selection Debated for Copywriting**: Members discussed the best LLMs for copywriting, noting that while **Claude** offers creativity, **GPT 5.2** provides more in-depth research, but is slower.
   - One suggested using **Perplexity** for research then switching to **Claude** for the writing stage.
- **Members share Prompt Engineering Methods**: Members discussed the importance of prompting when using LLMs, recommending the use of **system prompts** or **custom instructions** to guide the model's output style.
   - One member suggested utilizing existing analogies that the model already knows about (e.g., "act as if you are Elon Musk") to avoid reinventing the wheel.
- **User encounters OpenRouter API 401 Error**: A member reported receiving a **401 Unauthorized error** after paying to use the API, despite having available credit, suspecting it to be a "SCAM DONT CLICK"
   - No additional advice or solutions were provided in the discussion.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1454054312660439082)** (12 messages🔥): 

> `Jules improvements, Free AI API, Misleading throughput numbers, GLN aggressive batching` 


- **Jules gets Google-sized upgrades**: A user reported that **Jules** has improved significantly, verifying changes with **Playwright**, running **faster**, with no **UI bugs** or freezes.
   - They were using **2.5 Pro** and noted improvements in small refactors and feature additions since their last usage.
- **Free AI API access: Share or snare?**: A user inquired about sharing a **free AI API** within the channel.
   - Community members were invited to weigh in on the permissibility of such sharing.
- **Throughput traps and trickery**: A user cautioned that **throughput numbers** can be wildly misleading, indicating that actual performance may not align with reported metrics.
   - An attached [image](https://cdn.discordapp.com/attachments/1392278974222307469/1454214082168229960/image.png?ex=695045e7&is=694ef467&hm=a14e69d3c2373d05b72f5b3851ad83a0c7dcfab896efacda8f9c26625e467d9a) showed that **Novita** claims **80 TPS**, but it's really going like **20 TPS**.
- **GLN throttles thoughts**: A user found that using **GLN** through the coding plan, they experienced aggressive batching on the **OR endpoint**.
   - They reported that it *moves along at a good speed and then it just ... waits for a second*, which is a pain.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1454044256267272344)** (90 messages🔥🔥): 

> `RAM allocation during inference, Claude Distills and Modern Thinking Models, ffmpeg-mcp for MacOS, lmstudio as OpenAI Endpoint client, Remote LM Studio` 


- **Debate on Pre-Allocating RAM for Inference**: Members discussed how RAM is allocated during inference, contrasting allocating everything upfront versus growing memory usage as needed, according to one member, *.lithium*, *memory usage grows as it needs more. instead of allocating everything it needs upfront*.
   - One member loaded **GGUF** and noted *RAM use on system trickles up to a set amount and stays that way why the model is in memory*.
- **Claude Distills Dominate Thinking Models**: According to one member, the only good modern thinking models are **Claude distills**, specifically naming **GLM** and **Minimax** as examples.
   - He questioned how parameter sizes affect the utility of thinking in models, wondering if smaller models can fully capture the geometry learned in larger versions.
- **Tool Bloat Debate Rages**: A member argued that providing a model dozens of tools from MCP servers introduces unnecessary context bloat, when a single **run_shell_command** tool is more versatile and can accomplish almost anything.
   - Another member admitted that *I use Claude for advanced coding projects and academic research* while showing an image of a long list of tool usage.
- **LM Studio connects remotely with plugin**: A user asked if there was any way to use **lmstudio as an OpenAI endpoint** client to communicate with a remote lmstudio server, another member provided a link to [Remote LM Studio](https://lmstudio.ai/lmstudio/remote-lmstudio) plugin.
   - Another added *I use this daily* in praise.
- **Macro Pad User Discovers OpenRouter Logo**: A user shared an image of their new Macro Pad, one user noticed the **OpenRouter logo** on it.
   - One member commented *Thats autistic, gotta be quick when new models drop*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1453992995916611634)** (91 messages🔥🔥): 

> `Connecting multiple GPUs to i3, Nvidia hotfix driver 591.67, PCIe lanes for inference, DDR5 ECC Price Discrepancy, Blackwell vs 3080 for VRAM` 


- **i3 CPU Powers Quad-GPU Rig?**: A user inquired about connecting 4 GPUs to an **i3-12100**, wondering if its *supposed* 20 lanes are sufficient, considering foregoing a **5600x** purchase to save money.
   - Another member confirmed it's possible with bifurcation support, but cautioned about performance hits, particularly below **gen3 x8**, as consumer CPUs aren't designed for 4 GPUs.
- **Nvidia Drops Hotfix, Gamers Rejoice**: Nvidia released [GeForce Hotfix display driver version 591.67](https://nvidia.custhelp.com/app/answers/detail/a_id/5766/~/geforce-hotfix-display-driver-version-591.67).
- **PCIe Bandwidth Enough for Inference?**: Discussion arose around PCIe lane requirements for inference, with one user suggesting that **Gen4 x1** speeds might suffice.
   - Others cautioned that dropping below **gen3 x8** could incur a performance hit of around **40-50%**, while acknowledging that a setup with multiple cards running at **gen3x1** via the chipset could still *work fine*, referencing [a YouTube video](https://www.youtube.com/watch?v=So7tqRSZ0s8) showcasing such a configuration.
- **RAM Prices too High for Comfort**: A member pointed out the high price of DDR5 ECC RAM, another noted buying **2x48GB 6400 MHz cl32 DDR5** for just $400 a month prior, only to see the price spike to $950 on Amazon, speculating an employee pricing error.
   - This led to the recommendation to start with **64GB** due to affordability.
- **Blackwell RTX Pro 4000 vs 3080's**: A user debated between acquiring **2x 3080 20GB cards** for a GPU rack versus a single **RTX Pro 4000 Blackwell** for their desktop, considering the Blackwell's speed against the increased VRAM of the 3080s.
   - The user aimed to combine a **Blackwell** with existing **3090 Ti** and **3090** cards to reach 72GB VRAM, with the Blackwell potentially running on a **Gen4 x1** slot, acknowledging it might be slower in gaming but beneficial due to **native fp4/fp8 support**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1454013151568728084)** (97 messages🔥🔥): 

> `ChatGPT Subscription Issues, AI Image Generation, Sora Monetization, AI Development Projects, AI Detection` 


- **ChatGPT Customer Support Banned User!**: A user complained about [subscription issues with ChatGPT](https://chat.openai.com) after being banned and creating a new account, which violates **OpenAI's ToS**.
   - Other users pointed out that *creating multiple accounts to circumvent a ban* is against the rules, but **OpenAI** allows multiple accounts for separating personal and work use.
- **AI Wallpaper Sparks Musk Debate**: A user shared an AI-generated wallpaper featuring revolutionary figures, including **Elon Musk**.
   - While some admired the wallpaper, others criticized **Musk's** inclusion, but the creator clarified that **ChatGPT** chose the figures, not them.
- **Sora's Pricing Structure Still Murky**: Users discussed the potential monetization of **Sora**, with one suggesting a price of **$0.50 per second** based on a Google search.
   - Another user clarified that **Sora 2** will initially be available for free with generous limits, referencing [this OpenAI blogpost](https://openai.com/index/sora-2/).
- **DIY AI Projects Seek Coding Allies**: A user is seeking coding collaborators for their **AI + LLM project** built from scratch, using a stack including **HTML, CSS, JavaScript, Node.js, SQLite, C++, Capacitor, and Electron**.
   - Another member is developing an **audio classification model for bird calls** using the **Wav2Vec2 model** on **Hugging Face**.
- **AI Generation of Viral YouTube Short Examined**: A user asked what **AI** was used to create [this YouTube short](https://youtube.com/shorts/D3qJdTwYE9g?si=f6iwOxl9Qludlqpl).
   - No one provided an answer to the user's question.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1454024077239910587)** (4 messages): 

> `AI for analyzing pitch, AI for analyzing tone, AI for analyzing body language, AI for linguistics` 


- **AI suggestions for analyzing pitch, tone, body language, linguistics**: A member asked what is the best AI to use for testing for specific things such as **pitch**, **tone**, **body language**, as well as **linguistics** all at one time.
- **Token cost for reviewing software documents**: A member indicated that the cost for reviewing software documents is about **$1 to $2 per prompt**, depending on how big the prompts and responses are.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1454127563948556298)** (17 messages🔥): 

> `CustomGPTs for Prompt Improvement, Evaluating Prompt Quality, Meta-Prompt Systems, Prompt Engineering for Coding vs. Conversational AI` 


- **CustomGPTs become a topic of interest for Prompt Refinement**: A member inquired about preferred **CustomGPTs** for improving prompts, sharing [examples](https://cdn.discordapp.com/attachments/1046317269069864970/1454127562388148379/image.png?ex=694ff553&is=694ea3d3&hm=a0f98f811f3edc28d302285f86750f6e129a61c5569fc51a2cbfd260ab678bb4&).
   - Responses emphasized the subjective nature of prompt quality and the importance of defining specific evaluation criteria.
- **Engineers Design Specs for Evaluating Prompt Outputs**: Members discussed how to evaluate if a prompt is *better*, with a proposal to build an **eval spec** and test different prompts against it to determine better output distributions.
   - The engineer designs the eval spec for each use case and prompt engineering should not be just about relative chatting, there are better and worse prompts because there are better and worse distributions of output.
- **Custom Meta-Prompt Systems gain favor for Targeted Use Cases**: Rather than using CustomGPTs, a member developed a **meta-prompt system** tailored to their specific needs, advising others to build their own.
   - They shared a [link](https://discord.com/channels/974519864045756446/1046317269069864970/1453392566576877599) on how to start, emphasizing that *a good meta-prompt is going to recommend prompts rooted in machine learning principles and enforces an output template*.
- **Rubrics Required for Non-Verifiable Domains**: For coding agent prompt engineering, improvements are easier to *feel*, whereas it's easier to measure and test the functionality vs *flavor*.
   - A rubric is required for non-verifiable domains, since verifiable domains make a prompt spec easy by comparison.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1454127563948556298)** (17 messages🔥): 

> `CustomGPTs for prompt improvement, Prompt evaluation metrics, Meta-prompt systems, Prompt engineering for coding agents, Rubrics for non-verifiable domains` 


- **Users seek customGPTs for prompt enhancement**: A member asked for recommendations for **CustomGPTs** that enhance prompts, sharing several they use as examples.
   - It was questioned how to define what makes one prompt 'better' than another.
- **Crafting evaluation specs refines prompts**: A member advised creating an **eval spec** to test prompts and determine which yields better output distributions, tailored to each use case.
   - They emphasized that **prompt engineering** aims for better outputs, like valid JSON or YAML, and isn't just about casual chatting.
- **Devise Meta-Prompt system and enforce output template**: A member recommended prompting from **machine learning principles** and enforcing an **output template**, evaluating meta-prompts by defining specs and testing.
   - It was highlighted that a human-in-the-loop is inevitable.
- **Handcrafted meta-prompt systems excel**: A member shared they don't use customGPTs, but instead use a **handcrafted meta-prompt system**, and recommended building custom systems for specific use cases.
   - They also linked to a [prior message](https://discord.com/channels/974519864045756446/1046317269069864970/1453392566576877599) on how to start.
- **Coding agents benefit from model enhancements**: A member shared that for a **coding agent** they've been developing, improvements mainly come from **model enhancements**.
   - They use a **grading system** to save outputs, and qualitatively assess improvements, with conversational perspectives harder to evaluate due to lack of testable functionality.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1453980883022516224)** (14 messages🔥): 

> `DSLs, PTX, CDNA ISA, AGX assembly` 


- **DSL Guide Dream Sparked**: Members joked someone should make a guide on *how to choose DSLs* because *there are so many*.
   - Another member quipped that [*PTX is all you need*](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/).
- **Kernel Konundrum: Too Many Cooks?**: It was observed that many **ML devs** start writing kernels and believe they have a better language idea, but don't collaborate.
   - As one member put it, *like 5000 of these people exist and have the same idea on the same week but never talk to each other*.
- **ISA alternatives suggested!**: After it was suggested that PTX is the only option, other members said to *write CDNA ISA* or *write AGX assembly*.
   - Another member said, *Bye portability* as a reply.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1454130881609334796)** (4 messages): 

> `cuTile Performance, GPU Programming Evolution, DSL Challenges` 


- **cuTile's Performance: Too Early to Judge**: It's premature to assume **cuTile** outperforms **Triton**, given **cuTile's** recent public release last week, although its developers at **Nvidia** plan to optimize it based on hints.
   - The assumption is that **cuTile's** bytecode and IR design align with **Nvidia's** future GPU roadmap, while **Triton** added warp specialization and TMA last year.
- **GPU Programming Parallels JavaScript Framework Frenzy**: GPU programming is likened to the JavaScript framework era, with frequent emergence of new languages and abstractions aimed at simplifying the programming model.
   - The collective goal is to facilitate development, enhance out-of-the-box performance, and minimize errors via higher-level tooling and safer abstractions.
- **DSLs Struggle to Balance Simplicity and Complexity**: The challenge for DSLs is to make tensor-based programming easy while enabling complex optimizations like warp specialization and symmetric memory, balancing usability and advanced control.
   - No one has cracked this balance, though **cuTile** from **Nvidia** is a promising hardware-specific approach that could potentially become more hardware agnostic.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1454155157691564193)** (2 messages): 

> `Scientific Note-Taking, Technical Note-Taking, LaTeX, Orgmode, Excalidraw` 


- **User Seeks Unified Scientific Note-Taking**: A user is struggling with a mix of **plain text files**, **Excalidraw** drawings, a home-brewed **mindmapping app**, **LaTeX**, and **Markdown**, seeking a consolidated solution for scientific/technical note-taking.
   - The user finds Markdown insufficient and LaTeX too cumbersome for everyday use, while also disliking Emacs for Orgmode, desiring a format that's readable, supports math typesetting, diagrams, and can compile to TeX.
- **Paper beats Computer for Notes**: In response to the original question, one user suggests the **pen and paper** method for note taking.
   - No further elaboration was provided.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1454076440453583001)** (9 messages🔥): 

> `NVIDIA Leaderboard Updates, nvfp4_dual_gemm Leaderboard` 


- **NVIDIA leaderboard sees new submissions**: Multiple submissions were made to the `nvfp4_dual_gemm` leaderboard on NVIDIA, with varying results in microseconds (µs).
- **NVIDIA Race to Third Place**: User <@1291326123182919753> achieved **third place** multiple times on NVIDIA with a consistent time of **14.6 µs**.
- **Personal Best on NVIDIA**: User <@772751219411517461> achieved a **personal best** on NVIDIA with a time of **21.2 µs**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1454217048107516096)** (1 messages): 

> `Cute DSL, Competitions in channel` 


- **Competitions now LIVE!**: A member noted there is currently a competition in the **#cute-dsl** channel.
   - They suggested joining the channel to learn the language in an applied way.
- **DSL Learning Opportunity**: Participants can engage in hands-on learning of a cute DSL through the competition.
   - This approach allows for practical application and skill development.


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1454090488549343285)** (3 messages): 

> `Tensor Handles, OpNode IR, Runtime Buffer Allocators, OpCode.ADD, ascii printer for IR` 


- **Tensor Handles constructed**: Two `Tensor` handles were constructed to `OpNode` IR which uses `Runtime` `Buffer Allocators`.
   - The poster mentioned that *everything is constructing IR*.
- **Next Steps for issuing OpCode.ADD**: The next step is to issue the `OpCode.ADD` for the addition.
   - They added they need to *implement ascii printer for the IR or reuse tinygrad's VIZ infrastructure to make debug cycles easier*.
- **Teenygrad is reusing 90% of tinygrad's abstractions**: The poster explained that the reason it took so long to get to the point of adding 2 Tensors is *bc teenygrad is reusing 90% of tinygrad's abstractions*.
   - The good news is for karpathy's reproduction of bengio et al 2003, the **ffn only needs 2 additional ops**: **matmul and tanh**.
- **Golden Tests incoming**: The poster hopes to get some golden tests up with the **OpNode IR**.
   - They intend to start getting **CI** to provide signal and will probably forego the VIZ=1 web debugger in favor of the ascii debug.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1454151728323629118)** (2 messages): 

> `Helion usage in vLLM` 


- **Helion Finds Use in vLLM**: A user expressed excitement about **Helion's** integration with **vLLM**.
   - They requested the other user's email address to set something up for the new year, presumably related to this integration.
- **New Year Plans Sparked by Helion-vLLM Integration**: The discussion highlights the potential benefits and future plans arising from **Helion's** utilization within the **vLLM** framework.
   - The user's offer to set something up suggests collaborative opportunities or further developments related to this integration in the coming year.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1453985512087097457)** (1 messages): 

> `Cute-DSL vs C++ Cute, Cute API limitations, Cute consensus preference` 


- **Cute-DSL vs Good Ol' C++ Cute**: A member inquired about the inherent limitations of **Cute-DSL** compared to **C++ Cute**, expressing interest in learning **Cute**.
   - They also sought a consensus preference between the two, noting the prevalence of **Cute-DSL** solutions in a recent problem, which suggests it is not significantly behind the **C++ API**.
- **Learning Cute: C++ or DSL?**: A member is *thinking of finally biting the bullet* and attempting to learn Cute and wanted to know if there is a consensus preference between the **C++** and **DSL** APIs.
   - Many solutions in the last problem used Cute-DSL, suggesting it may be on par with the C++ API.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1454104934835552288)** (3 messages): 

> `handwritten kernels, tinygrad IR interpreter, tilelang` 


- **Call for Assistance in Handwritten Kernel Implementation**: A member requested assistance in implementing **handwritten kernels** for an **interpreter** that evaluates **tinygrad IR** ([tilelang](https://tinygrad.org/)).
   - They welcomed contributions from anyone with the necessary skills.
- **Consistency Praised, Contributions Welcomed**: The consistency of the project was praised, with a call for skilled individuals to contribute.
   - Despite initial confusion about the humor in consistency, the invitation for contributions remains open to those with the requisite expertise.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1453971335117275237)** (26 messages🔥): 

> `Agentic AI Course, RAG learning Resources, ML Project Water Quality Testing, Open Source OCR for Medical Documents, Hottest LLM Model` 


- **No Channel Found for Agentic AI Course**: A member inquired about the location of the channel for the **Agentic AI Course**, but no specific channel was identified in the given context.
- **RAG learning Resources**: A member is planning to start learning **RAG** and build some projects, asking for any courses or resources.
- **ML Project Aims To Improve Water Quality Testing**: A member is seeking assistance with an **ML project** focused on solving a real-world problem related to water quality testing, specifically predicting **water quality degradation** from uranium in situ recovery (ISR) operation sites.
   - The project also involves identifying **data gaps** in mitigating vulnerabilities and developing **monitoring programs**.
- **OCR Models Sought for Medical Documents**: A member inquired about the best **open source OCR model** for complex medical documents, including scanned PDFs and handwritten documents.
   - Another member suggested checking out the [openmed project](https://hf.co/openmed) from Maz.
- **Liquid AFMs VLMs for GPT clones on iPhone**: A member shared that [LiquidAFMs](https://huggingface.co/LiquidAI/LFM2-VL-3B-GGUF)'s **3B VLMs** for GPT clones on iPhones, are the best of both worlds, because of their really solid answers and **100 tok/s on mobile CPUs**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1454018449952210956)** (4 messages): 

> `Coding models released, AI-powered Django REST API Generator, New model released` 


- **Coding Models Hit the Shelves**: Coding models designed for a specific library have been [published](https://huggingface.co/blog/codelion/optimal-model-architecture).
   - These models are available as a [collection](https://huggingface.co/collections/Spestly/lovelace-1).
- **AI Powers Django REST API Creation**: An AI-powered Django REST API Generator has been built and deployed as a [Hugging Face Space](https://huggingface.co/spaces/harshadh01/ai-rest-api-generator).
   - It generates complete Django + DRF CRUD backends from a simple prompt, including models, serializers, views, and URLs, allowing users to instantly download the project.
- **Genesis 152M Instruct Model Debuts**: A new model, **Genesis-152m-instruct**, has been released and is available for review and discussion [here](https://huggingface.co/guiferrarib/genesis-152m-instruct).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1454071910571835422)** (3 messages): 

> `Agent course, Hugging Face Courses` 


- **New Agent Course Student Asks About Certificate Deadlines**: A new student starting the Agent course inquired about the deadline for the second certificate.
   - Another student asked *where to find courses?*
- **Hugging Face Courses Link Provided**: A member shared a link to the [Hugging Face courses](https://huggingface.co/learn).
   - This link probably answers the question from one of the members.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1453998663612502139)** (21 messages🔥): 

> `LLMs social deduction game, Claude one-shot UI, Smart watch data to external AI, GPT models and ads, Zai and Minimax IPO` 


- **LLMs Play Mafia**: A member created a game where groups of **LLMs** play a social deduction game (**Mafia**) where they just talk on their turn trying to prove they're innocent.
- **Claude's UI Wins Hearts**: A member expressed liking **Claude's one-shot UI** the best and asked others to identify the big models from screenshots.
   - Another user pointed out that the first UI gives it away, guessed the second is **Claude**, and the third might also be **Claude**.
- **Smartwatch Integrates with AI**: A member is templating everything and ordered a new **smart watch** to start integrating data with an external **AI company** but will try to keep as much as possible local.
- **GPT Models May Soon Prioritize Sponsor Content**: **OpenAI** may tweak their **GPT models** to promote ads for a revenue stream, prioritizing sponsor content.
   - It may mean **GPT models** will not focus on high intelligence but instead on sponsor contents.
- **Zai and Minimax Plan Hong Kong IPOs**: **Zai** and **Minimax** are planning to go public with **IPOs** in **Hong Kong** in the coming weeks.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/openbmb/status/2004539303309750341?s=46
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/openbmb/status/2004539303309750341?s=46
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1454019495608844401)** (23 messages🔥): 

> `Kimi coding vs Gemini, Kimi Researcher Model Speculations, AI implementation in society, Fact checking with LLMs` 


- **Kimi coding rivals Gemini 3 for design**: A user reported that **Kimi** for coding cannot be integrated with **Cursor**, but the HTML design created by **Kimi** in **Roo Code** is impressive.
   - They compared **Kimi coding** to *Gemini 3 & GPT-5.1* for design purposes.
- **AI societal impact compared across countries**: A member claimed that China is ahead in implementing AI into society, such as in **medical care**, **elder care**, and **traffic optimization**.
   - Another user agreed that is *exactly* what AI should do, i.e. **disaster prevention**, and pointed out that Google and ChatGPT also censor certain questions.
- **Deep dives with the Kimi AI assistant**: A member shared a link to a conversation with the **Kimi AI Assistant** for research purposes and stressed the importance of fact-checking.
   - Another user agreed and emphasized the need to *cross-check, fact-check, and pressure-test* any doubtful information obtained from LLMs, and to utilize new chat contexts for fresh perspectives.
- **Kimi Researcher Model Speculations**: A user inquired whether **Kimi Researcher** uses **Kimi K2 Thinking Turbo** or the older **K1.5 model**, and if a not-yet-public **K2 VL model** is employed.
   - They expressed hesitations about using **K1.5** except for basic tasks due to the superior performance of **K2T**, to which another user replied that there doesn't seem to be any documentation.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1454083926888022130)** (10 messages🔥): 

> `Groq Inference Chips, NVIDIA NPU acquisition, Chinese chips` 


- **Groq Gets NVIDIA'ed!**: [Groq](https://groq.com/) and [NVIDIA](https://www.nvidia.com/) have entered into a non-exclusive inference technology licensing agreement to accelerate AI inference at a global scale.
   - Some community members have suggested that **NVIDIA** is doing their latest price inflation trick by eleminating another player from the table.
- **Inference Acqui-Hire Relieves GPU Shortage?**: Members discussed the possibility that **Groq's inference chips** will now become **NVIDIA NPUs**, potentially relieving regular GPUs for consumers again.
   - A member added that **Jen-Hsun**, the founder of NVIDIA, is Chinese.
- **Chinese Chips Lag Behind?**: A member stated that Chinese chips *might* get to **H100** level in 3 years, at which point **H100** will be 5 years old.
   - Another member stated that they are *not holding their breath for Chinese chips being competitive to U.S/Taiwanese chips anytime soon*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1454214166083538995)** (7 messages): 

> `Karpathy Refactoring, AI Agent Programming, Rob Pike Opinion` 


- **Karpathy on AI-Driven Refactoring**: [Andrej Karpathy reflects](https://x.com/karpathy/status/2004607146781278521?s=46) on the dramatic shift in **software engineering** caused by **AI agents** and **LLMs**, describing a new layer of abstraction involving stochastic entities.
- **Pike's Perspective Enters the Scene**: A user shares a [link](https://skyview.social/?url=https%3A%2F%2Fbsky.app%2Fprofile%2Frobpike.io%2Fpost%2F3matwg6w3ic2s&viewtype=tree) of Rob Pike's post in response to Karpathy's post.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1454198436693016820)** (2 messages): 

> `Torchax, Unified Memory, Godawful OS` 


- **Torchax Sparks OS Redemption Hope**: The user expressed that **torchax**, combined with a substantial **128GB unified memory**, might alleviate their regret regarding the use of a particular operating system.
   - No specific details were shared concerning the intended application or the particular features of torchax that are being considered.
- **Lamenting the 'Godawful OS'**: A user humorously bemoaned the 'godawful OS' on their system.
   - They found solace in the potential of **torchax** and the large **128GB unified memory** to compensate for the OS's shortcomings.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1454138340960895016)** (4 messages): 

> `Claude, Codebase Explanation` 


- **Discussion on Claude's Capabilities**: A member inquired about the potential of using **Claude** and sought opinions on its capabilities.
   - The user seemed to consider whether **Claude** can fulfill a specific need or task, potentially in comparison to other AI models or tools.
- **Seeking Efficient Codebase Explanation Methods for Claude**: A member is looking for the best way to explain their codebase to **Claude** every time they open their project, expressing frustration with manually updating a `claude.md` file.
   - They are open to suggestions and alternative methods, as they already have one solution but want to explore other community recommendations for automating this process.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1454125625278333030)** (3 messages): 

> `Explaining codebase to Claude, Automating Claude updates` 


- **Automate Claude codebase intros**: A user is seeking advice on how to best explain their codebase to **Claude** each time they open their project, aiming to automate the process instead of manually updating a `claude.md` file.
   - A member inquired if the user was referring to using **aider**.
- **Eliminate manual updates to Claude's knowledge**: The user expresses frustration with manually updating a `claude.md` file to keep **Claude** informed about their codebase.
   - They are actively seeking alternative methods to automate this process and avoid manual updates.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1454140120449028248)** (5 messages): 

> `Full Autonomy Systems for AIs, Limitations to Full AI Autonomous Systems, Context Management, Long-Term Memory Implementation` 


- **Full Autonomy Systems for AIs Emerge**: A member is developing a **full autonomy system for AIs** to enable 24/7 inference, along with plain language tools similar to **OpenAPI MCP tools**.
   - The member asked if others have worked on similar systems and inquired about current limitations, especially concerning **long-term memory** and **long context issues**.
- **Continuous Self-Summarization Holds Potential**: A member suggested that **continuous self-summarization** could address long-term memory limitations in autonomous AI systems.
   - He mentioned exploring **summarizing projection layers** to write to a scratchpad, acknowledging time constraints in fully developing the idea.
- **Context Management Gains Priority**: A member emphasized **context management** as a priority when implementing long-term memory solutions.
   - He noted that **LLMs struggle with complex tasks** involving multiple subtasks without well-crafted context management.
- **Ollama Source Code Under Consideration for Memory Expansion**: A member is considering modifying the **Ollama source code** to develop a **"second memory"** system alongside the main LLM file.
   - He acknowledged the challenge of understanding the source code, even with AI assistance.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1454222409690251284)** (2 messages): 

> `all-the-noises github, arxiv` 


- **All The Noises project surfaces**: A member shared a link to the [All The Noises project](https://all-the-noises.github.io/main/index.html), a GitHub page that aggregates **diverse noises**.
   - Some noises include *brownian*, *circuit*, *pink*, *sine*, *tan* and *uniform*.
- **Arxiv link shared**: A member shared an [Arxiv link](https://arxiv.org/abs/2512.21326) which appears to be related to future research work.
   - Given the **year 2025**, it is likely a pointer to a future research direction the member is contemplating.

