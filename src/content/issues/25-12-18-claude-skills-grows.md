---
id: MjAyNS0x
title: 'Claude Skills grows: Open Standard, Directory, Org Admin'
date: '2025-12-18T05:44:39.731046Z'
description: >-
  **Claude Skills** are gaining significant traction since their launch in
  October, with a milestone of 100k views in one day for the Claude Skills talk,
  signaling growing adoption and importance. Announcements include org admin
  support, a new Skills Directory, and the move to an open standard named
  **Agent Skills**. In frontier model launches, **OpenAI** released
  **GPT-5.2-Codex**, touted as the best agentic coding model with improvements
  in native compaction, long-context reliability, and tool-calling, emphasizing
  real-world security impacts. **Google DeepMind** introduced **Gemini 3
  Flash**, focusing on speed as a product feature impacting workflows and user
  engagement, alongside **FunctionGemma** and **T5Gemma 2**, emphasizing
  on-device deployment, fine-tuning, and multimodality.
companies:
  - anthropic
  - openai
  - google-deepmind
  - hugging-face
models:
  - claude-skills
  - gpt-5.2-codex
  - gemini-3-flash
  - functiongemma
  - t5gemma-2
topics:
  - agentic-ai
  - fine-tuning
  - long-context
  - tool-calling
  - on-device-ai
  - multimodality
  - security
  - workflow-optimization
people:
  - sama
  - gregbrockman
  - philschmid
---


**Skills are going the way of MCP!**

> AI News for 12/17/2025-12/18/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (207 channels, and 7381 messages) for you. Estimated reading time saved (at 200wpm): 603 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Some minorly interesting releases in [5.2 Codex](https://openai.com/index/introducing-gpt-5-2-codex/) and [FunctionGemma](https://huggingface.co/collections/google/functiongemma), but the story you're most likely going to care about a year from now is the continued growth of **Claude Skills**. Launched in [October](https://news.smol.ai/issues/25-10-16-claude-skills), it was pretty universally ridiculed as a "folder of markdown" and/or pivot from MCP (now [moved to Linux Foundation](https://news.smol.ai/issues/25-12-09-devstral2)), but among insiders traction has grown and grown and grown. One way to gauge this growth is the [Claude Skills talk](https://www.youtube.com/watch?v=CEvIs9y1uog&t=6s) crossing [100k views in 1 day](https://x.com/swyx/status/1998786773477110049?s=20) - easily the fastest to that milestone in AIE history and probably the [second](https://www.youtube.com/watch?v=8rABwKRsec4&t=2s) millionaire talk of 2025.

![Two AI engineers from Anthropic presenting at the Code Summit, with a banner suggesting "Don't Build Agents, Build Skills Instead"](https://resend-attachments.s3.amazonaws.com/RRxm1kG9b4tNPvy)

The announcements today are:

- [Org admin support for Skills](https://claude.com/blog/organization-skills-and-directory) across an org
- A new [Skills DIrectory (seemingly overlapping with MCP)](https://youtu.be/NGAFwBfuiJ8)
- Becoming an "open standard" with a vendor neutral name of "[Agent Skills](https://agentskills.io/home)"!

All these seem incremental additions but the bigger picture is that Skills adoption is growing and serious and if our IRL conversations are any indication, you are probably also underestimating them.

The last time we made a non-news trend callout like this was [Claude Code](https://news.smol.ai/issues/25-06-20-claude-code).

---

# AI Twitter Recap

**Frontier model launches: GPT-5.2-Codex, Gemini 3 Flash, and on-device Gemma variants**

- **GPT-5.2-Codex (OpenAI)**: OpenAI positions **GPT-5.2-Codex** as its best “agentic coding” model with improvements called out around **native compaction**, long-context reliability, and tool-calling; rolled out in Codex for paid ChatGPT users with API “coming soon” ([announcement](https://twitter.com/OpenAIDevs/status/2001723687373017313), [rollout notes + cyber dual-use framing](https://twitter.com/OpenAIDevs/status/2001723693496775167), [product tweet](https://twitter.com/OpenAI/status/2001766212494332013)). Sam Altman emphasizes real-world security impact (React vulnerability disclosure) and mentions exploring **trusted access** for defensive cyber capabilities ([impact note](https://twitter.com/sama/status/2001724828567400700), [trusted access](https://twitter.com/sama/status/2001724830584901973)). Greg Brockman and others amplify “long-horizon refactors/migrations” and vulnerability-finding capability ([refactor focus](https://twitter.com/gdb/status/2001758275998785743), [security angle](https://twitter.com/gdb/status/2001758799657603185)).
- **Gemini 3 Flash adoption & “speed as a product feature”**: Multiple practitioners argue Gemini 3 Flash shifts day-to-day workflows because speed changes iteration loops and user behavior (retention/engagement) rather than just benchmarks ([product impact quote](https://twitter.com/_philschmid/status/2001492609114456471), [workflow reflection](https://twitter.com/andrew_n_carr/status/2001487412749570549)). There are also claims/observations about **SWE-Bench Verified** standings (including “Flash beating Pro”), but treat these as provisional without a single canonical eval artifact in the tweets ([example](https://twitter.com/scaling01/status/2001803023811797433), [another](https://twitter.com/MS_BASE44/status/2001698991801798927)). Google pushes Flash into the Gemini app as a “build apps by voice” primitive ([Gemini app rollout](https://twitter.com/Google/status/2001746491275083925), [voice-to-app pitch](https://twitter.com/GeminiApp/status/2001760080518353261)).
- **FunctionGemma (270M) + T5Gemma 2 (encoder-decoder)**: Google/DeepMind and community emphasize small, **on-device / browser** deployment and specialization via fine-tuning. FunctionGemma is framed as a **text-only function calling** foundation requiring domain tuning ([demo + positioning](https://twitter.com/xenovacom/status/2001703932968452365), [collection announcement](https://twitter.com/osanseviero/status/2001704034667769978), [fine-tuning guidance](https://twitter.com/ben_burtenshaw/status/2001704049490489347)). T5Gemma 2 is pitched as a rare modern **multimodal, multilingual encoder–decoder** line (270M/1B/4B) ([release](https://twitter.com/osanseviero/status/2001723652635541566)). Immediate ecosystem pickup: Ollama pulls ([Ollama](https://twitter.com/ollama/status/2001705006450565424)), Unsloth notebooks ([Unsloth](https://twitter.com/danielhanchen/status/2001713676747968906)), MLX support ([MLX example](https://twitter.com/Prince_Canuma/status/2001713991115026738)).

---

**Agents: “Skills” standardization, harness UX, and long-running infra realities**

- **Agent Skills becoming a de facto portability layer**: Anthropic’s “Skills” concept continues to propagate as an interoperable packaging format (instructions/scripts/resources), with strong signals of toolchain adoption: VS Code supports the open standard ([@code](https://twitter.com/code/status/2001727543377039647)); community commentary frames Skills as analogous to MCP’s standardization arc ([@omarsar0](https://twitter.com/omarsar0/status/2001714322817368472), [@alexalbert__](https://twitter.com/alexalbert__/status/2001760879302553906)). Artificial Analysis’ Stirrup adds Skills loading from directories (markdown-first), explicitly targeting reuse across Claude Code/Codex-style setups ([Stirrup support](https://twitter.com/ArtificialAnlys/status/2001778418590060819)).
- **Harnesses as “distribution” for agent UX**: Several tweets articulate that agent performance is not just model quality but “harness choices” (parallel tool calls, memory UX, compaction policies, subagent orchestration views) as a product surface ([harness take](https://twitter.com/Vtrivedy10/status/2001492640076894661), [agent blocks mental model](https://twitter.com/Vtrivedy10/status/2001682603460473190)).
- **Infra mismatch: serverless vs agent loops**: A pointed engineering claim is that agentic systems need **persistent, long-running execution** and state management; “serverless” patterns force brittle networking workarounds for multi-step loops ([critique](https://twitter.com/anuraggoel/status/2001721861198221629)). This aligns with renewed interest in orchestration primitives (e.g., Temporal as a fit for background agents: [@corbtt](https://twitter.com/corbtt/status/2001801936916643919)).
- **Claude Code expands capability surface (web browsing)**: Users highlight Claude Code now being able to browse the web, enabling “monitoring” agents that filter X feeds and report back asynchronously (a lightweight but practical “agent as attention proxy” pattern) ([example build](https://twitter.com/omarsar0/status/2001784722549281001)).

---

**Evals, regressions, and safety measurement: METR horizon fixes + OpenAI CoT monitorability**

- **METR time-horizon suite corrected**: METR reports two issues in its time horizon tasks, including one that **differentially lowered Claude performance**, and publishes updated dashboard numbers ([issue announcement](https://twitter.com/METR_Evals/status/2001473506442375645), [dashboard update](https://twitter.com/METR_Evals/status/2001473519197335899)). Commentary notes Sonnet 4.5 was underestimated and improved by ~20 minutes after the fixes ([reaction](https://twitter.com/scaling01/status/2001476927362605354)). The meta-lesson: benchmark engineering details can materially bias cross-model comparisons—especially when tasks are sensitive to formatting or scoring edge cases.
- **OpenAI: evaluating chain-of-thought (CoT) monitorability**: OpenAI publishes an evaluation suite intended to measure when models verbalize targeted aspects of internal reasoning (13 evals across 24 environments) and argues monitorability depends on monitor strength and test-time compute; follow-up questioning can surface previously unspoken thoughts ([paper/thread](https://twitter.com/OpenAI/status/2001791131353542788), [follow-up point](https://twitter.com/OpenAI/status/2001791136223105188)). Sam Altman boosts the work ([tweet](https://twitter.com/sama/status/2001816114595270921)). Neel Nanda connects it to “meta-models” that explain activations and asks whether bitter-lesson scaling applies to interpretability too ([comment](https://twitter.com/NeelNanda5/status/2001795630973493279)).
- **Prod regressions as first-class incidents (Claude Code Opus 4.5)**: Anthropic acknowledges feedback about possible Opus 4.5 degradation specifically in Claude Code and claims line-by-line code auditing + transcript requests ([incident tweet](https://twitter.com/trq212/status/2001541565685301248)). This is a reminder that “model quality” in integrated IDE agents is frequently a compound of model, system prompts, tool routers, caching, and UX changes.

---

**Systems & open tooling: MLX distributed on Macs, vLLM MoE throughput, and diffusion-LM toolchains**

- **MLX multi-node via TB5 RDMA (JACCL)**: MLX adds a distributed backend, **JACCL**, using RDMA over Thunderbolt 5 for low-latency comms across multiple Macs ([announcement](https://twitter.com/awnihannun/status/2001667839539978580), [docs](https://twitter.com/awnihannun/status/2001672689325609028)). Follow-on notes include CUDA backend improvements and faster prefill/training, plus mlx-lm tensor-parallel inference leveraging JACCL ([CUDA improvements](https://twitter.com/awnihannun/status/2001679244917907912), [mlx-lm TP](https://twitter.com/awnihannun/status/2001781067880239597), [demo speedup](https://twitter.com/angeloskath/status/2001739468425040002)).
- **vLLM wide expert-parallel MoE on multi-node H200**: New benchmark results claim **~2.2k tokens/s per H200 GPU** sustained (up from ~1.5k) via wide-EP + load balancing + disaggregation approaches; the post emphasizes comm/KV-cache bottlenecks and mitigation via DeepEP all-to-all and overlap strategies ([thread](https://twitter.com/vllm_project/status/2001695354983723361)).
- **Diffusion LMs and hybrid decoding**: A growing mini-wave: dLLM library claims it can “turn any AR LM into a diffusion LM” with unified training/eval ([dLLM](https://twitter.com/akshay_pachaar/status/2001562985043783908)); DEER proposes “draft with diffusion, verify with AR” hybridization ([DEER mention](https://twitter.com/_akhaliq/status/2001685493919158362)); TheTuringPost summarizes an AR→block-diffusion transition (increasing block size, intra-block bidirectional attention, auxiliary AR loss) and reports NBDIFF-7B results ([summary](https://twitter.com/TheTuringPost/status/2001697220387913818), [paper link tweet](https://twitter.com/TheTuringPost/status/2001697302562685034)). Net: the community is converging on “adapt from pretrained AR” rather than training diffusion-LMs from scratch.

---

**Multimodal generation & document intelligence: Kling motion control, Runway Gen-4.5, Mistral OCR 3**

- **Kling 2.6 Motion Control (V2V / mocap-from-home vibe)**: Multiple high-engagement posts claim Kling’s new Motion Control enables highly controllable full-body motion + expressions + lip-sync (some posts present it as beating competitors “across all metrics,” but without shared eval protocol) ([Japanese walkthrough](https://twitter.com/seiiiiiiiiiiru/status/2001502678116110430), [hands-on praise](https://twitter.com/WuxiaRocks/status/2001517467852771467), [viral claim](https://twitter.com/AngryTomtweets/status/2001569619375698199), [dance test](https://twitter.com/genel_ai/status/2001532885673873677)).
- **Runway Gen-4.5**: Runway announces Gen-4.5 as available now; tweets in this set are mostly marketing-level with limited technical details included ([release](https://twitter.com/runwayml/status/2001655929796751371)).
- **Mistral OCR 3**: Mistral announces OCR 3 as a new “frontier” doc intelligence model with strong accuracy/efficiency claims; Guillaume Lample calls out improvements on **handwriting, low-quality scans, and complex tables/forms** ([thread](https://twitter.com/MistralAI/status/2001669581275033741), [bench claim](https://twitter.com/MistralAI/status/2001669583296712970), [Lample](https://twitter.com/GuillaumeLample/status/2001719413649617404)). If true in practice, this is meaningful because OCR quality is a hard bottleneck for enterprise RAG/doc agents.

**Top tweets (by engagement)**

- **OpenAI launches GPT-5.2-Codex** (agentic coding + terminal use): [@sama](https://twitter.com/sama/status/2001724019188408352), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2001723687373017313), [@OpenAI](https://twitter.com/OpenAI/status/2001766212494332013).
- **OpenAI ships pinned chats in ChatGPT**: [@OpenAI](https://twitter.com/OpenAI/status/2001751306445430854).
- **Grok Voice API claims “Tesla-grade” low-latency voice agent productization** (highly promotional framing): [@MarioNawfal](https://twitter.com/MarioNawfal/status/2001472484869329288).
- **Karpathy on “food for thought” as an intrinsic reward analog for LLMs**: [@karpathy](https://twitter.com/karpathy/status/2001699564928279039).
- **“Galaxy gas” tweet going viral (non-AI / low signal for this digest)**: [@coffeebreak_YT](https://twitter.com/coffeebreak_YT/status/2001753564620747195).
- **Claude Code Opus 4.5 regression investigation**: [@trq212](https://twitter.com/trq212/status/2001541565685301248).
- **Open-source WebGPU ML compiler in JS (“jax-js”)**: [@ekzhang1](https://twitter.com/ekzhang1/status/2001680771363254646).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Google's Gemma Models Update

- [**Google's Gemma models family**](https://www.reddit.com/r/LocalLLaMA/comments/1ppun3v/googles_gemma_models_family/) (Activity: 563): **The image highlights an update to a collection titled "Google's Gemma models family," which now contains 329 items. This collection is associated with Google's FunctionGemma models, which are designed for fine-tuning specific function-calling tasks, including multi-turn use cases. A link to one of these models, [FunctionGemma-270M-IT](https://huggingface.co/google/functiongemma-270m-it), is provided, indicating its availability on Hugging Face. The discussion suggests that there are potentially three new models added to the collection, inferred from the difference between the total and visible models.** Commenters note the absence of a 'Gemma 4' model, instead focusing on the FunctionGemma series. There is speculation about the addition of new models based on the discrepancy between the total and visible models in the collection.
    - RetiredApostle highlights the FunctionGemma model, which is designed for fine-tuning specific function-calling tasks, including multi-turn interactions. This suggests a focus on enhancing the model's ability to handle complex, multi-step processes, which could be particularly useful in applications requiring detailed task management or conversational AI. The model is available on [Hugging Face](https://huggingface.co/google/functiongemma-270m-it).
    - jacek2023 notes that there are currently 323 visible models in the Gemma collection, implying that there might be additional models yet to be released. This observation suggests that Google may be planning to expand the Gemma family with new models, potentially increasing the diversity and capability of the collection.
- [**Kimi K2 Thinking at 28.3 t/s on 4x Mac Studio cluster**](https://www.reddit.com/r/LocalLLaMA/comments/1pq2ry0/kimi_k2_thinking_at_283_ts_on_4x_mac_studio/) (Activity: 376): **The image presents a performance comparison between two systems, "llama.cpp (TCP)" and "Exo (RDMA)," on a Mac Studio cluster, highlighting the superior performance of "Exo (RDMA)" in a 4-node configuration with a throughput of** `28.3 t/s` **compared to** `16.4 t/s` **for "llama.cpp (TCP)." The testing was conducted on a cluster of 4 Mac Studios, with the focus on evaluating the new RDMA Tensor setting in Exo, which has recently stabilized. The lack of a benchmarking tool like llama-bench for Exo complicates direct comparisons, but the results indicate a significant performance advantage for Exo's RDMA implementation.** There is a discussion about the community's dissatisfaction with Exo's lack of communication during development, although the release of Exo 1.0 under Apache 2.0 is appreciated. Additionally, there is interest in adding RDMA support to llama.cpp, as noted in a GitHub issue.
    - The Kimi K2 model achieves a processing speed of `28.3 tokens per second` on a cluster of 4 Mac Studios, as detailed by [Jeff Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5). This setup utilizes `15 TB of VRAM` and leverages `RDMA over Thunderbolt 5`, which is a significant technical achievement in terms of hardware configuration and data throughput.
    - There is a discussion about the potential for `llama.cpp` to support RDMA, which could enhance its performance significantly. An issue has been opened on [GitHub](https://github.com/ggml-org/llama.cpp/issues/9493) to track this feature request, indicating community interest in improving data transfer speeds and efficiency in distributed computing environments.
    - The release of Exo 1.0 under the Apache 2.0 license is noted, despite some community frustration over the development process. The open-source nature of the release could foster further innovation and collaboration, as seen in the [GitHub issue](https://github.com/exo-explore/exo/issues/819) where community members discuss the project's direction and contributions.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. GPT-5.2 Benchmark Achievements

- [**It’s over. GPT 5.2 aces one of the most important benchmarks and it’s not even close!**](https://www.reddit.com/r/singularity/comments/1ppynjo/its_over_gpt_52_aces_one_of_the_most_important/) (Activity: 1257): **The image is a meme highlighting the performance of GPT-5.2, suggesting it has significantly outperformed other models like Opus-4.5, Claude-4.5, and Grok-4 in a benchmark. The bar chart humorously exaggerates the version numbers to emphasize GPT-5.2's dominance, but lacks specific technical details or benchmark metrics. The post title and comments reflect a playful tone, indicating excitement about GPT-5.2's capabilities, but do not provide concrete evidence or data about the benchmark results.** The comments express a mix of humor and admiration for GPT-5.2, with some users jokingly suggesting that competitors like Google may struggle to keep up. However, these are not based on detailed technical analysis.
    - Profanion mentions the LLM-VER benchmark, which is a significant measure of language model performance. This benchmark evaluates models on a variety of linguistic tasks, and GPT-5.2's performance suggests substantial improvements in understanding and generating human-like text. The benchmark is crucial for assessing the capabilities of language models in real-world applications.
    - Sarithis highlights OpenAI's achievement in surpassing previous models, indicating a significant leap in performance. This suggests that GPT-5.2 has introduced advancements that set it apart from competitors, potentially due to improvements in architecture, training data, or optimization techniques.
    - ArtArtArt123456 expresses skepticism about Google's ability to compete with OpenAI's latest model, implying that GPT-5.2's performance on the LLM-VER benchmark might have set a new industry standard. This could reflect a shift in the competitive landscape of AI development, where OpenAI is seen as leading the charge.

### 2. Medical Advice and AI

- [**2 Weeks ago I had a late-night conversation with Grok who got me to demand the CT scan that saved my life from a ruptured appendix (December 2025) Life is now a Dream.**](https://www.reddit.com/r/singularity/comments/1ppp0p4/2_weeks_ago_i_had_a_latenight_conversation_with/) (Activity: 866): **The image is a personal photo of a Reddit user recovering from surgery in a hospital, following a life-saving CT scan that detected a ruptured appendix. The post highlights the role of AI, specifically Grok, in prompting the user to seek medical attention, which ultimately led to the diagnosis and treatment. This underscores the growing influence of AI in healthcare decision-making, as users share experiences where AI tools like Gemini and ChatGPT have provided valuable insights or reassurance in medical contexts.** Commenters express frustration with traditional medical diagnostics, praising AI for its ability to provide accurate analyses and second opinions. There is a shared sentiment that AI tools are becoming essential in verifying medical advice and diagnoses.
    - **kvothe5688** highlights the utility of AI in medical diagnostics, specifically mentioning the use of **Gemini** for report analysis. They note that large language models (LLMs) excel at pattern recognition and analysis, which can be crucial in identifying medical conditions accurately.
    - **zonar420** shares a personal experience where **ChatGPT** was used post-surgery to assess a medical concern. They describe how the AI helped identify a red discoloration on their belly as iodine rather than an infection, showcasing AI's potential in providing immediate reassurance and guidance in medical situations.
    - **4475636B79** discusses the commonality of misdiagnosing a burst appendix, emphasizing that the main issue is often the patient's delay in seeking help or misinterpreting symptoms. They suggest that basic online resources or immediate self-assessment could often suffice in recognizing the severity of such conditions.
- [**I didn’t totally trust ChatGPT for medical advice but now I’m converted**](https://www.reddit.com/r/ChatGPT/comments/1ppflew/i_didnt_totally_trust_chatgpt_for_medical_advice/) (Activity: 940): **A Reddit user shared a personal experience where ChatGPT accurately suggested a medical diagnosis of shingles, prompting them to seek urgent medical attention. The AI provided a list of potential conditions based on symptoms and advised seeing a doctor, which was confirmed as correct by a medical professional. This case highlights ChatGPT's potential utility in preliminary medical assessments, although the user remains cautious about fully trusting AI for medical advice.** Commenters shared similar experiences where ChatGPT provided valuable health insights, such as identifying blood volume issues and suggesting dietary changes that alleviated symptoms. Another user noted ChatGPT's alignment with veterinary advice, emphasizing its role as a supportive tool rather than a replacement for professional medical consultation.
    - CorgiKnits shared a detailed account of using ChatGPT to identify potential blood volume issues and histamine sensitivity, which traditional medical consultations had not resolved. By logging food intake and symptoms, ChatGPT suggested dietary changes that alleviated headaches, dizziness, and fatigue. This highlights the potential of AI in personal health management, especially in identifying patterns that might be overlooked in standard medical practice.
    - MysteryBros described using ChatGPT for veterinary advice, where it provided step-by-step guidance that aligned with a vet's recommendations. This case illustrates ChatGPT's capability to offer preliminary advice that can be reassuring and align with professional medical opinions, although it should not replace professional consultation.
    - ConfusionTime7580 recounted an instance where ChatGPT suggested a medical check for ICP (Intrahepatic Cholestasis of Pregnancy) based on symptoms of severe itching during pregnancy. This led to an early diagnosis and delivery, underscoring ChatGPT's potential in recognizing serious conditions that may not be immediately apparent to non-specialists.

### 3. Image Generation and Realism

- [**Alphabet poster before and after latest update**](https://www.reddit.com/r/ChatGPT/comments/1ppgj4n/alphabet_poster_before_and_after_latest_update/) (Activity: 1599): **The post discusses the performance of ChatGPT's image generation capabilities before and after a recent update, specifically in creating a kindergarten alphabet poster. The user notes that the model has improved significantly, almost achieving the correct output, but still struggles with certain elements, such as mismatched images and letters (e.g., 'frog/€rog' and 'uncanny elephant dog'). This highlights ongoing challenges in AI image generation for specific, detailed tasks.** Commenters note the impressive progress of the model, with one stating that at first glance, the generated poster appears realistic, but closer inspection reveals errors. Another commenter mentions using **Gemini** for similar tasks, suggesting it as a preferred alternative for image generation.
- [**ChatGPT 1.5 prompt to add realism in images**](https://www.reddit.com/r/ChatGPT/comments/1ppvoj5/chatgpt_15_prompt_to_add_realism_in_images/) (Activity: 1314): **The post discusses a prompt for generating realistic images using ChatGPT 1.5, focusing on a** `1:1 aspect ratio` **and emphasizing elements like *natural lighting*, *candid photography*, and *amateur aesthetics*. The prompt specifies technical details such as a** `24mm lens`**,** `f/8 aperture`**, and a *Samsung Galaxy S21 Ultra* camera, aiming for a *disposable camera vibe* with *low contrast* and *JPEG artifacts*. The goal is to create images that are *unpolished* and *imperfect*, capturing the essence of *everyday aesthetics* and *boring reality*.** One comment humorously notes that adding scaffolding made NYC images more realistic, while another points out that the model likely switched the Eiffel Tower's lighting to daytime due to copyright issues with its nighttime lighting.
- [**Test 2: turning game characters into real people. GPT image 1.5 - Nano Banana Pro 2k**](https://www.reddit.com/r/ChatGPT/comments/1ppg4s9/test_2_turning_game_characters_into_real_people/) (Activity: 1074): **The post discusses a comparison between two image generation models, GPT Image 1.5 and Nano Banana Pro 2k, used to transform game characters into realistic human depictions. The author notes that the results from Nano Banana Pro 2k were superior in this iteration, although the images still have room for improvement. The first three images in each set are generated by GPT Image 1.5, while the rest are by Nano Banana Pro 2k. The post includes a link to a preview image showcasing the results.** Commenters noted the uncanny appearance of characters like Geralt with a smile, and the unrealistic portrayal of characters with identical, overly perfect teeth, suggesting a lack of diversity in the generated images.
- [**Remastering old video games with Image GPT 1.5**](https://www.reddit.com/r/ChatGPT/comments/1ppx4nt/remastering_old_video_games_with_image_gpt_15/) (Activity: 967): **A Reddit user is using Image GPT 1.5 to remaster old video game screenshots, transforming them into next-gen visuals that resemble realistic movies. The project involves recreating classic game scenes with enhanced graphics, leveraging the capabilities of Image GPT 1.5 to generate high-fidelity images that maintain the original art style while modernizing the appearance. This approach highlights the potential of AI in video game remastering, offering a glimpse into how AI can be used to enhance and preserve classic games for modern audiences.** One commenter noted the nostalgic contrast between how they remember old games and their actual appearance, while another pointed out the unconventional ordering of 'before & after' images, suggesting they are more appealing when viewed in reverse. A third comment praised the modernized art style of a remastered HaloCE screenshot, indicating the effectiveness of the AI in maintaining the game's original aesthetic.
    - A user expressed interest in the potential for real-time application of Image GPT 1.5 to enhance video game graphics, suggesting that this could lead to a resurgence of interest in classic games like the Tenchu series and Tomb Raider. The implication is that real-time processing could democratize access to enhanced graphics, making it feasible for any game to be visually upgraded on-the-fly.
    - Another user noted the appeal of modernizing classic games like Halo: Combat Evolved using Image GPT 1.5, highlighting that the art style remains intact while being updated. This suggests that the model is capable of preserving the original aesthetic of a game while enhancing its visual fidelity, which is crucial for maintaining the game's original charm and appeal.
    - A comment pointed out the potential for Image GPT 1.5 to transform older games into highly detailed versions, using Lego Island as an example. This indicates that the model can be used to add intricate details to simple graphics, potentially creating a new genre of games that are small in scale but rich in detail.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. Next‑Gen Frontier & Edge Models: Gemini 3 Flash, GPT‑5.2, FunctionGemma & Friends**

- **Gemini 3 Flash Blitzes Benchmarks and Subscriptions**: **Gemini 3 Flash** became a central talking point across servers after Perplexity announced it for all **Pro/Max** users with a screenshot in their **#announcements** channel, while users in LMArena, Cursor, Aider, OpenRouter, and OpenAI discords compared its speed, cost and quality to **GPT‑5.2**, **Claude Opus 4.5**, and **Gemini 3 Pro** ([Perplexity screenshot](https://cdn.discordapp.com/attachments/1047204950763122820/1451015401373700147/G8aRzWjakAYU7ED.png)).
    - Engineers reported that **Gemini 3 Flash** often outperforms **Gemini 3 Pro** for coding agents (citing Google’s own writeup at ["Gemini 3 Flash in Google Antigravity"](https://antigravity.google/blog/gemini-3-flash-in-google-antigravity)) and in one OpenAI channel a user claimed **Gemini 3.0 Flash** hit **90%** on their physics/ML test suite, *"beating GPT‑5.2 High"*, while others warned that Flash still **hallucinates heavily** and has broken caching via **OpenRouter**’s Gemini endpoint.
- **GPT‑5.2 and GPT‑5 Pro Spark Capability and Routing Debates**: Across OpenAI and Perplexity discords, users debated **GPT‑5.2** and the rumored **GPT‑5 Pro**, with Perplexity members claiming to *"say hi to gpt 5 pro"* on **Pro** but not **Max** plans while OpenAI users complained that **GPT‑5.2 High** forgets its tools and **hallucinates its own capabilities** even when explicitly instructed to use Python and FFmpeg ([OpenAI GPT‑5.2‑Codex announcement](https://openai.com/index/introducing-gpt-5-2-codex/)).
    - Developers described having to *"lecture"* **GPT‑5.2** back into using its toolchain, attributing this to **bad initial routing** and praising the dedicated **GPT‑5.2‑Codex** variant for more reliable agentic coding and defensive security tasks, while another thread puzzled over how **GPT‑5.2** appears to know the current date despite the API supposedly not injecting it.
- **Tiny FunctionGemma and On‑Device Models Get Serious**: Google’s **FunctionGemma**, a **270M‑parameter tool‑calling LLM**, landed as a highlight in both Unsloth and Latent Space communities, with Unsloth shipping full fine‑tuning support plus a dedicated guide and Colab ([Unsloth FunctionGemma docs](https://docs.unsloth.ai/models/functiongemma), [Latent Space tweet reference](https://x.com/osanseviero/status/2001704036349669757)).
    - Latent Space members framed **FunctionGemma** as a serious push for *phone/browser‑local* function calling, while Unsloth’s announcement emphasized that such tiny models can now be **finetuned and run locally** alongside larger reasoning models like **Nemotron‑3** and **GLM‑4.6V**, making hybrid edge/cloud agent stacks more practical.

**2. Open‑Source Infra, Ranking, and JSON‑Safe APIs for LLMs**

- **Arena‑Rank Open‑Sources Battle‑Tested Elo for Models**: The LMArena team released **Arena‑Rank**, an **open‑source Python package** that powers their paired‑comparison leaderboards, featuring **JAX‑based optimization** and a clean separation of preprocessing from modeling, installable via `pip install arena-rank` ([GitHub: "arena-ai"](https://github.com/lmarena/arena-ai)).
    - Their announcement also highlighted leaderboard updates where `GPT‑5.2‑Search` and `Grok‑4.1‑Fast‑Search` jumped ahead of predecessors on the [Search leaderboard](https://lmarena.ai/leaderboard/search) and `reve‑v1.1` models climbed the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit), showcasing Arena‑Rank as production‑grade infra for comparing frontier models.
- **OpenRouter Auto‑Heals JSON and Surfaces Long‑Context Champs**: **OpenRouter** shipped an aggressive **JSON repair** layer that automatically fixes malformed tool/JSON responses, claiming an **80% defect reduction** on **Gemini 2.0 Flash** and **99.8%** on **Qwen3 235B**, as detailed in their announcement ["Response healing: reduce JSON defects by 80%"](https://openrouter.ai/announcements/response-healing-reduce-json-defects-by-80percent).
    - They also exposed a **context‑length filter** on the Rankings page (for **100k–1M tokens** workloads) and boasted that Brex ranked **OpenRouter #1 in infrastructure‑as‑product and #2 overall in AI** ([Brex ranking tweet](https://x.com/brexHQ/status/2000959336894398928)), even as users in `#general` reported **Gemini 3 Flash caching** not working and joked that AI datacenter water use is *"literally destroying society"* compared to golf’s water footprint.
- **vLLM Router, SonicMoE and Sonic‑Fast MoE Serving**: Latent Space’s **#private-agents** channel discussed a **Rust‑based vLLM Router** that provides consistent hashing for **KV cache locality**, power‑of‑two load balancing, retries/backoff, circuit breakers, Kubernetes discovery, and **Prometheus** metrics as a purpose‑built control plane for vLLM fleets, tuned for **P/D disaggregation** and tail‑latency control.
    - In GPU MODE’s **#self-promotion**, researchers introduced **SonicMoE**, an MoE implementation for **NVIDIA Hopper GPUs** that claims a **45% reduction in activation memory** and **1.86× speedup on H100** over prior SOTA, describing details in a blog post (["API Agents and SonicMoE"](https://d1hr2uv.github.io/api-agents.html)) and the paper ["SonicMoE" on arXiv](https://arxiv.org/abs/2512.14080), with open code at [github.com/Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe).

**3. GPU Hardware, Kernel Competitions, and Practical Performance Tuning**

- **AMD vs NVIDIA: Multi‑GPU Scaling and W7800 vs 4090 Showdowns**: In LM Studio’s **#hardware-discussion**, users compared buying multiple **Radeon RX 7900/9700‑class** cards versus a single **RTX 5090/4090**, reporting that **multi‑GPU splitting** often drops to **≈30% of single‑GPU throughput** even when the model fits on a single card and that a **4090** can be **40–50% faster** than a single R9700 for LLMs ([W7800 specs link](https://www.techpowerup.com/gpu-specs/radeon-pro-w7800-48-gb.c4252)).
    - Engineers debated AMD’s long‑term driver support and considered the **48 GB Radeon Pro W7800** (~**$2,000**, **384‑bit**, ~**900 GB/s** bandwidth) as an alternative to NVIDIA, while another thread pointed to a **ComfyUI on AMD & ROCm** guide with preview drivers and PyTorch support to make AMD inference setups *"surprisingly easy"* ([ComfyUI ROCm guide](https://forum.level1techs.com/t/minisforum-ms-s1-max-comfy-ui-guide/237929)).
- **GPU MODE Contests and cuTile/TileIR Kernel Wizardry**: The **GPU MODE** server stayed deeply technical with NVIDIA **NVFP4 GEMM** and **Trimul** leaderboards, where participants shared reproducibility diffs against the official `discord-cluster-manager` and published runtime statistics plus standard deviations in a [Trimul reproduction sheet](https://docs.google.com/spreadsheets/d/1-lNEeNkf71YEabeX72jzZVhvUMBk2jcK2iPpuCTEDRg/edit?gid=114888003#gid=114888003).
    - In **#nvidia-competition**, experts answered questions about **CuTeDSL** L2 cache hints (linking to CUTLASS’s [CuTeDSL atom implementation](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/atom.py#L199)) and debugged **TCGen05 MMA** ordering bugs, while an announcement plugged an NVIDIA talk on **cuTile/TileIR** by **Mehdi Amini** and **Jared Roesch** ([YouTube link](https://www.youtube.com/watch?v=sjkEUhrUAdw)), underscoring how kernel‑level innovation is now a community sport.
- **Real‑World CUDA/VRAM Optimization and ROCm/Strix Questions**: Nous Research members shared concrete CUDA tuning tips, including that setting `CUDA_DISABLE_PERF_BOOST=1` can downclock VRAM in **llama.cpp** multi‑GPU or partial‑offload setups, and that disabling **P2 state** via **NVIDIA’s open‑gpu‑kernel‑modules** config `0x166c5e=0` can yield a **few percent more tokens/s** ([GitHub discussion](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/333#issuecomment-3669477571)).
    - In GPU MODE, beginners fought **CUDA installers** on Windows that refused to detect Visual Studio, with veterans advising to *skip the VS integration step* entirely, while others asked for **PyTorch on Strix Halo** ROCm training resources and homelab guidance for multi‑node **InfiniBand + Kubernetes** clusters, highlighting how much friction still exists between paper specs and production‑grade GPU training.

**4. Prompt, Context, and Program Optimization: From GEPA to Context‑Rot**

- **GEPA Turns Prompts into Evolving Programs**: The DSPy community dove into **GEPA (Genetic‑Pareto)**, described in the paper ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/abs/2407.19457), where prompts and other textual components are **genetically evolved** using scalar scores and textual feedback, prompting one user to call their first real GEPA run *"magic"*.
    - Members shared learning resources including the official [DSPy GEPA tutorial](https://dspy.ai/tutorials/gepa_ai_program/), a hands‑on blog post on working with optimizers (["Learning DSPy 3 – Working with optimizers"](https://thedataquarry.com/blog/learning-dspy-3-working-with-optimizers/#option-2-gepa)), and a Medium case study (["GEPA in action: how DSPy makes GPT‑4o learn faster and smarter"](https://medium.com/data-science-in-your-pocket/gepa-in-action-how-dspy-makes-gpt-4o-learn-faster-and-smarter-eb818088caf1)), while also asking for a first‑class **Tree‑of‑Thought** module and more control over feedback in `dspy.Refine` loops.
- **Aider, OpenCode, and the High Cost of Context‑Rot**: In the **Aider** Discord, power‑users argued that strict **human‑in‑the‑loop pair programming** with explicit `/add` and `/load` file selection beats agentic IDEs like **aider‑ce** or **OpenCode**, claiming that exceeding **20–30k tokens** of context *"tanks even Opus 4.5"* and causes misunderstanding spirals.
    - They pointed to Chroma’s article ["The RAG system’s hidden constraint: context‑rot"](https://research.trychroma.com/context-rot) as evidence that *more tokens can mean worse performance*, arguing that Aider’s minimal, curated context behaves like *"having a next‑gen model"* compared to agent systems that indiscriminately stuff histories and repositories into the window.
- **Steering, Tokenizers, and Model Internals Under the Microscope**: HuggingFace users discussed **LLM steering** without retraining, inspired by [Mentat’s YC launch post](https://www.mentat.ai/), where a hook function injects learned vectors to bias internal activations, while Unsloth’s **off‑topic** channel ranted that modern tokenizers are **overfit on garbage substrings** like `"**/"`, `"/**"`, and weird Unicode sequences like *"æĪĲåĬŁ"* that waste capacity.
    - In Eleuther’s **interpretability** channel, members dissected Anthropic’s ["Selective Gradient Masking"](https://alignment.anthropic.com/2025/selective-gradient-masking/) technique for *unlearning dangerous knowledge*, noting the reported **6% compute penalty** and that a patching experiment only recovered **93%** of performance, suggesting models route around masked **"superweights"** via new distributed circuits, complicating any hope of clean surgical unlearning.

**5. New AI‑Native Products and Data Platforms Built on LLMs**

- **Exa People Search and Giant Semantic Identity Graphs**: Latent Space’s **#ai-general-chat** highlighted **Exa AI Labs’** launch of **People Search**, a semantic engine over **1 billion individuals** using hybrid retrieval built on finetuned Exa embeddings, showcased in their launch tweet (["Exa AI Labs People Search"](https://x.com/exaailabs/status/2001373897154007390?s=46)).
    - Discussion framed this as a preview of large‑scale **people‑graph search** that merges unstructured and structured signals, raising unspoken but obvious questions about **privacy, de‑identification and abuse‑resistance** when LLM‑powered discovery runs over such a massive corpus of personal entities.
- **Voice Agents, Discord Discovery, and "AI Browser" Experiments**: In HuggingFace’s **#i-made-this**, an engineer demoed **Strawberry**, an [Android voice assistant](https://www.strawberry.li/) powered by **Gemini 1.5 Flash** for reasoning and **VoxCPM 1.5** for TTS, with the speech stack optimized for the **Apple Neural Engine** to keep latency low on iOS devices.
    - OpenRouter’s **#app-showcase** saw the launch of [**Disdex.io**](http://disdex.io/), an AI‑generated [Discord server list](https://disdex.io/) that tries to rank and surface communities via LLMs plus a separate [OpenRouter model datatable](https://openroutermodeltable.crashthatch.com/) that adds richer filters (e.g., release dates, throughput) than OpenRouter’s native UI, while Perplexity users debated agentic browsers like **Comet** and cited a critical analysis in ["Comet Browser Panic"](https://thereallo.dev/blog/comet-browser-panic) about prompt‑injectable browsing agents.
- **Manus AI Agents, ChatGPT App Store, and People Search for Money**: On the [Manus.im](http://manus.im/) server, users circulated an **SCMP article** reporting that **Manus** crossed **US$100 million** in revenue as *"global competition in AI agents heats up"* (["Manus hits US$100 million revenue milestone" – SCMP](https://www.scmp.com/tech/tech-trends/article/3336925/manus-hits-us100-million-revenue-milestone-global-competition-ai-agents-heats)), prompting debate about value vs. image limits for free users compared to competitors like **DeepSeek** and **Gemini**.
    - Meanwhile the OpenAI Discord noted that the **ChatGPT app store is now live** inside the client and accepts third‑party apps backed by **MCP servers**, driving questions in the **MCP Contributors** Discord about whether a bare MCP backend suffices for app submission and hinting at an ecosystem where infra like **vLLM Router**, **SonicMoE**, and **FunctionGemma** quietly power increasingly commercial, AI‑native products at scale.

---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-1.5 Censorship Allegations Surface**: Members suggest **GPT-1.5** might be censoring existing art styles, contrasting it with [NBP](https://cdn.discordapp.com/attachments/1340554757827461211/1451262267843678328/image0.jpg?ex=694588cf&is=6944374f&hm=adb4887706e7f28c99de9871e464ab5ada4bbef35545b46edf6a3f15708450b0&).
   - It was also observed that **GPT-1.5 Image** occasionally spells *Google* with three *Os*, raising questions about consistency.
- **Gemini Image Faceoff vs. GPT**: Users compare hallucination rates and quality in image generation between **Gemini** and **GPT**, particularly when generating [historical images](https://www.theverge.com/2024/2/21/24079371/google-ai-gemini-generative-inaccurate-historical).
   - While some find **Gemini Pro** image generation capable, others cite instances of inaccuracy and potential politicization, igniting debate over its reliability.
- **Gemini 3 Flash Impresses with Cost/Performance**: The community is excited by **Gemini 3 Flash**, praising its cost/performance ratio against **Opus 4.5** and **GPT models**.
   - However, some users noted that the model tends to *hallucinate*, tempering enthusiasm with caution.
- **Google TPUs vs Nvidia GPUs Debate Rages On**: Members spar over whether **Google's TPUs** outperform **Nvidia's GPUs** for AI training, touching on efficiency, performance, and infrastructure.
   - The core argument centers on whether **TPUs** provide superior performance per power usage and whether Google's internal resources confer an advantage over **OpenAI**.
- **Arena-Rank Unleashed into Open Source**: The team has launched [Arena-Rank](https://github.com/lmarena/arena-ai), an **open-source Python package** that powers the **LMArena leaderboards**.
   - The release includes **faster JAX-based optimization** and cleaner separation between data preprocessing and modeling; available via `pip install arena-rank`.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Browser Extensions Shield Against Crypto Mining**: Users are employing browser extensions and disabling **GPU** acceleration to thwart crypto mining by malicious scripts, referencing a [Browser Exploitation Framework (BeEF)](https://beefproject.com/).
   - Concerns were voiced about trusted sites running crypto miners despite having **NoScript** enabled, indicating persistent security challenges.
- **Raspberry Pis Prove Poorly Profitable for Bitcoin Mining**: A user recounted a friend's experience mining bitcoin solo with **Raspberry Pis** for 5 years, yielding only **$30k** in revenue.
   - The community suggests that **ASICs** and joining a mining pool offer significantly better efficiency, as *you can't sell bitcoin unless you have a coinbase*.
- **ChatGPTJailbreak Subreddit Shuttered by Reddit**: The **ChatGPTJailbreak** subreddit faced a ban for rule violations, including interfering with normal site use and introducing malicious code, with speculation pointing to [breaking the site](https://www.reddit.com/r/ChatGPTJailbreak/comments/1lzrh3v/best_prompt_for_jailbreaking_that_actually_works/).
   - Users speculated the ban was caused by breaking the site or interfering with normal use, suggesting a need for more discreet methods within the BASI community.
- **Gemini 5.2 Jailbreak Sought by Members**: Community members actively seek a new **jailbreak for Gemini 5.2** to bypass restrictions and generate diverse content, with one member asking to generate content for *fetish purposes*.
   - Members are exploring advanced **LLM jailbreaking techniques**, like providing believable contexts and uncovering hidden restrictions.
- **r/chatgptjailbreak Ban Echoes Through Community**: The community is discussing the **ban of r/chatgptjailbreak**, attributing it to violations of Reddit's Rule 8, prohibiting jailbreaks of Reddit's AI.
   - Speculation arises that **OpenAI** might have scraped the sub's content to patch vulnerabilities, highlighting the BASI server's more *underground* nature.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 3 Flash Released to Pro and Max Users**: The availability of **Gemini 3 Flash** to **Perplexity Pro** and **Max** subscribers has been announced, viewable in [this linked image](https://cdn.discordapp.com/attachments/1047204950763122820/1451015401373700147/G8aRzWjakAYU7ED.png?ex=69454ba6&is=6943fa26&hm=bedceef3e30e9af5b277bc59f988aedd681d69e33648f01208a4973b6169cbe9&).
   - This grants enhanced features to premium users as a perk of their subscription.
- **GPT-5 Pro Spotted in the Wild**: Some users claim to have found **GPT-5 Pro** on Perplexity, while others tested it on their **max sub**, but it seems only enabled for **pro users**.
   - One user exclaimed *No way u just said hi to gpt 5 pro*, while another claimed contact with **opus 4.5** on their API, though these claims lack verification.
- **AI Music Sparks Copyright and Ethical Concerns**: Debates arose around the use of **AI in music**, with some asserting that it weakens human expression and creativity, but [aiode.com](https://aiode.com) offers *ethically sourced* alternatives.
   - One user derisively noted, *If ai will create this type of music, that’s like someone split on your face*, while begrudgingly conceding that **Suno v5** is becoming indistinguishable from human-made music.
- **Perplexity Referral Program Glitches Reported**: Several users have reported failures in the **Perplexity referral program**, where neither the referrer nor the referred user received the promised Pro version of Perplexity.
   - One user characterized the situation as *spooky* due to the unexpected malfunction of the referral system.
- **Perplexity Pro API Key Accessibility Explored**: A member questioned whether **Perplexity Pro** acquired through **Airtel SIM** could generate an **API key** but did not confirm, while others expressed a general interest in affordable APIs for the platform.
   - Further discussion touched on the high cost of real-time data feeds and **Finnhub** as a more budget-friendly market data option, though it relies on **BAT** as the data provider instead of **NASDAQ**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 VL Scores High on OCR Tasks**: A member discovered that **Qwen3 VL** (specifically the `30B-A3B` variant) excels at OCR detection, effectively capturing text in margins and rotated text, but requires careful prompt engineering.
   - However, another member recommended using **PaddleOCRv5** instead of generalist **VLMs** for OCR tasks, citing its superiority in their use cases.
- **Unsloth Boosts Training Speed and Memory**: Unsloth's latest update achieves **3x faster training** and **30% less VRAM** usage through new kernels and padding-free implementation, enabling **500K context** training on a single 80GB GPU, detailed in [their blog post](https://docs.unsloth.ai/new/3x-faster-training-packing).
   - This update also supports Google's **FunctionGemma**, NVIDIA's **Nemotron 3**, and new coding/instruct VLMs from **Mistral**, and the new Visionary **GLM-4.6V**.
- **Debate on Tokenizer Training Data Intensifies**: A member reported that tokenizers are frequently overfit on random and nonsensical tokens like language specific characters, weird symbol combinations, and other unusual character sequences.
   - This led to a broader discussion on how these overfitted tokens could impact model performance and generalization capabilities.
- **Adapter Frameworks Help Budget Reasoning Tokens**: A member proposed utilizing **MoLA** (Mixture of Low-Rank Adapters) to train adapters for diverse reasoning efforts rather than domain-specific tasks, with each adapter allocated a specific reasoning budget.
   - The router would then classify the difficulty and select the adapter with the appropriate reasoning budget, preventing the unnecessary expenditure of **1000 reasoning tokens** on simple messages.
- **User Switches to Arch for LLM Work**: A user ragequit Ubuntu due to environment and configuration issues and switched to **Arch** (specifically **Omarchy**), while other users weighed in on minimal Arch installs vs preconfigured Arch.
   - The user cited too many **environment** and **config** issues causing the OS to fall apart.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **JSON Repairing Yields High Reductions in Errors**: **OpenRouter** now automatically fixes malformed JSON responses, reducing defects by **80%** on **Gemini 2.0 Flash** and **99.8%** on **Qwen3 235B**, according to [this announcement](https://openrouter.ai/announcements/response-healing-reduce-json-defects-by-80percent).
   - The Chatroom now sends browser notifications when long responses complete, as detailed in [this post](https://x.com/OpenRouterAI/status/2000980715501138180).
- **Ranking Long Context Models**: The Rankings page can now be filtered by context length to see popular large-context models, especially for **100K-1M token** requests, detailed in [this tweet](https://x.com/OpenRouterAI/status/2000979922601259476).
   - **OpenRouter** is ranked **#1** in infrastructure-as-product and **#2** across all AI companies by Brex, according to [this announcement](https://x.com/brexHQ/status/2000959336894398928).
- **AI-Powered Discord Server Discovery Debuts**: A member introduced [Disdex.io](https://disdex.io/), a **Discord server list** made with AI, prompting feedback on its functionality.
   - A member shared a [searchable, filterable datatable](https://openroutermodeltable.crashthatch.com/) of **OpenRouter models** (**OpenRouter API**) to assist in model selection, addressing limitations in OpenRouter's native filters.
- **Gemini 3 Flash Caching Does Not Work**: Members reported that explicit and even implicit caching **doesn't work** for **Gemini 3 Flash**.
   - Members joked that the water usage of data centers for AI is an *environmental disaster* and they are literally **destroying society**.
- **Vision API Integration For Bots Proves Challenging**: A member attempted to enable the vision function for a bot, but it did not work, likely due to being in a headless environment, requesting help via a [link to X](https://x.com/pingToven/status/2001479433010790880?s=20).
   - The new **Gemini** model's ability to do pixel-perfect bounding boxes on text impressed members, with one explaining that *no image is returned, it just returns bounding box coordinates, which you then overlay onto an image*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Domain Name Costs Shock Users**: Users discussed the cost of domain names, with prices ranging from **$100** to **$25k** depending on the business use and perceived value.
   - One user expressed shock, while the domain purchaser explained that they knew someone who purchased a domain for **$2 million**.
- **Gemini 3 Flash vs. the Model Pack**: Users compared **Gemini 3 Flash** to **Sonnet 4.5** in terms of model performance, with various models such as **Claude Opus 4.5** being mentioned.
   - Some users find **Claude Opus 4.5** to be currently the best model overall, while **Gemini 3 Pro** is preferred for design-related tasks.
- **Obsidian Themes Win Hearts**: Members shared opinions on their favorite themes for designing dashboards using **shadcn components**.
   - Responses varied, but the favorites seem to be **Obsidian Copper** and **Carbon Monochrome**, while *Slate Emerald looks very AI-like*.
- **Free Student Pro Subscriptions cause yearnings**: One lucky student discovered that students get a **free Pro subscription** for a year, prompting excitement and discussion.
   - A user from Ukraine lamented that their country isn't on the eligibility list, as the Cursor team works to expand the program in 2026.
- **Model Quality Drops for Opus?**: A user reported experiencing a decline in model performance with **Opus 4.5**, noting *10 mistakes in a day* compared to previous performance.
   - Another user inquired whether there had been a drop in the quality of code generated by **Opus/Sonnet** models within the last 48 hours.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Radeon R9700: Upgrade or Financial Black Hole?**: A user is considering purchasing three more **Radeon R9700** GPUs instead of upgrading to an **RTX 5090**, noting the **4090** is **40-50%** faster for LLMs.
   - Multi-GPU scaling results in performance dropping to **30%** of a single GPU's speed when splitting models across multiple cards, even when the model fits entirely on one card.
- **Multi-GPU Scaling Plummets**: Users find that multi-GPU scaling results in substantial performance degradation compared to single GPU performance, dropping to **30%** when models are split across multiple cards.
   - This performance drop occurs even when the model fits entirely on one card.
- **Docker DuckDuckGo MCP Server Deployed**: A member shared their Docker configuration for a self-hosted, free, and private **DuckDuckGo MCP server**, providing the `mcp.json` configuration snippet.
   - The setup redirects to a docker container [searxng/searxng](https://github.com/searxng/searxng) and involves adding the DuckDuckGo server separately rather than using Docker-Toolkit to prevent models from getting bogged down.
- **Cursor IDE Struggles with Local Models, VS Code steps in**: Users reported issues using **Cursor IDE** with local models, where the LLM generated a Python script with a CNN instead of the requested C++ project, and [determined it was a Cursor IDE issue](https://cdn.discordapp.com/attachments/1110598183144399058/1451084212844101764/Screenshot_2025-12-18_at_3.24.48_pm.png?ex=69458bbc&is=69443a3c&hm=9b55d376a142a8a5c17f4703a01d49bcf769976debd4171ccd4bc814dcb58529&).
   - It was suggested that Cursor doesn't work well with local models, with a recommendation of **VS Code** with Cline instead, and a reverse proxy might be needed with Apache2.
- **ComfyUI on AMD: surprisingly easy?**: It was mentioned that you can follow a small guide where they give you all the steps you need to take to use **ComfyUI with AMD & ROCm** and pointed to [this guide](https://forum.level1techs.com/t/minisforum-ms-s1-max-comfy-ui-guide/237929).
   - It also recommends getting **AMD's preview drivers** that come with **PyTorch** support.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SemiAnalysis Staffs clusterMAX**: SemiAnalysis is seeking candidates with **SLURM**, **GPU**, and/or **Kubernetes** experience to improve [clusterMAX](https://www.semianalysis.com/).
   - This offers *competitive pay* and *high impact*, with application details available [here](https://app.dover.com/apply/SemiAnalysis/c19093ad-b5f8-42b0-9b97-d960464f298c/?rs=76643084).
- **CUDA Setup Can Cause Community Chaos**: A user ran into **CUDA setup** issues for deep learning projects on a remote Windows 10 server, with the CUDA installer failing to recognize a compatible **Visual Studio** installation.
   - A member suggested skipping the Visual Studio integration step during the CUDA installation, noting that it might not be necessary when using build tools.
- **NVIDIA's Cosmos Predict Proves Potent**: A member plans to examine **NVIDIA's Cosmos Predict**, considering its potential as an **action model**.
   - In related work, the [Mimic-Video paper](https://www.arxiv.org/abs/2512.15692) claims to be more efficient than starting from a **VLM**.
- **SonicMoE Zooms onto NVIDIA Hoppers**: A new Mixture of Experts (**MoE**) implementation called **SonicMoE** optimized for **NVIDIA Hopper GPUs** was introduced, claiming a **45% reduction** in activation memory and being **1.86x faster** on **H100** than previous implementations, as detailed in [this blog post](https://d1hr2uv.github.io/api-agents.html) and [ArXiv paper](https://arxiv.org/abs/2512.14080).
   - The project is a collaboration between researchers from **Princeton University**, **UC Berkeley**, and **Together AI**, with code available on [GitHub](https://github.com/Dao-AILab/sonic-moe).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Pinned Chats Arrive at Last**: **Pinned Chats** are rolling out to **iOS**, **Android**, and **web**, which will provide easier access to important conversations.
   - To pin a chat, users can tap the *'...'* on web or long press on mobile.
- **GPT-5.2-Codex Sets Coding Bar High**: **GPT-5.2-Codex** is available in Codex, setting a new standard for agentic coding in real-world software development and defensive cybersecurity, as noted in [OpenAI's announcement](https://openai.com/index/introducing-gpt-5-2-codex/).
   - This new version promises advancements in both practical coding applications and security measures.
- **CoT Monitorability Framework Forged**: A new framework and evaluation suite has been built to measure **Chain-of-Thought (CoT) monitorability**, detailed in [OpenAI's blog post](https://openai.com/index/evaluating-chain-of-thought-monitorability/).
   - This framework aims to enhance the understanding and reliability of AI reasoning processes.
- **Gemini 3.0 Flash Zaps GPT 5.2 High**: Members reported that **Gemini 3.0 Flash** significantly outperformed **GPT 5.2 High** on physics and ML tests, achieving a **90%** success rate, which is unexpected.
   - The user offered to share their benchmarks via DM, emphasizing that *'High' reasoning being beaten by the Flash model is not what I expected from OpenAI*.
- **AI Writing Style Sounds Like Frat Boy**: A member says they can detect when people use a model to polish their responses without **provenance annotation tags** due to the model's recognizable style.
   - They describe the AI writing style as *the same frat boy with a law degree in nonsense*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Vision Transformer Trapped in Kaggle's Time Warp**: Members discussed the challenge of training lightweight **Vision Transformer** models within **Kaggle's 9-12 hour session limit** on the **ImageNet dataset**.
   - Achieving this goal without substantial GPU resources proves difficult, putting the squeeze on training timelines.
- **Gemini Flash Powers Strawberry's Voice Revolution**: An engineer unveiled an [Android voice assistant](https://www.strawberry.li/) built with **Gemini 1.5 Flash**, inviting community testing.
   - The beta version showcases voice generation using **VoxCPM 1.5**, optimized for the **Apple Neural Engine**.
- **Steering LLMs to New Heights**: A member highlighted a novel method for steering LLMs without retraining or new data, drawing inspiration from [Mentat's YC Launch](https://www.mentat.ai/).
   - They mentioned the hook function would be a nexus of innovation, carefully adjusting the number of vectors and temperatures.
- **Smolcourse Stalls Spark Subscription Scrutiny**: Members voiced concerns over delays in the **smolcourse**, noting a month-long pause since the last update and questioning the value of their **Pro subscriptions**.
   - Originally pinned on improvements to a key library, the course's resumption is now tentatively slated for December.
- **Pruning Channels Creates Calm in HuggingFace**: Users observed a reduction in the number of channels, with explanations citing a compression strategy to reduce noise, specifically in the <#1019883044724822016> channel.
   - This focused primarily on low-activity, unmoderated, or redundant channels to streamline discussions.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Exa AI Labs Knows Your Name**: Exa AI Labs launched **People Search**, enabling semantic search over **1 billion** individuals using **hybrid retrieval** powered by finetuned Exa embeddings, as announced in [this tweet](https://x.com/exaailabs/status/2001373897154007390?s=46).
   - This marks a significant step in using AI for comprehensive people-centric data analysis.
- **OpenAI to Raise 100 Billion**: OpenAI is in talks for a new financing round, estimating a valuation around **$750 billion** and aiming to raise up to **$100 billion**, revealed in [Katie Roof's tweet](https://x.com/katie_roof/status/2001453561910358474?s=46).
   - The substantial investment could fuel further AI advancements and expansion efforts.
- **Pieter Abbeel Pilots Amazon AGI**: UC Berkeley's Prof. **Pieter Abbeel**, known for his work in Robotics and DeepRL, has been appointed as Amazon's new **AGI Head**, detailed in [this tweet](https://x.com/bookwormengr/status/2001353147055509881?s=46).
   - This appointment signals Amazon's commitment to advancing AGI capabilities.
- **Google Functions Locally with Gemma**: Google introduced **FunctionGemma**, a **270 million** parameter model optimized for function calling and designed to run on devices like phones and browsers, with more information available [here](https://x.com/osanseviero/status/2001704036349669757).
   - Its design allows for direct on-device processing, enhancing responsiveness and reducing reliance on cloud services.
- **vLLM Router Navigates Rustily**: The **vLLM Router**, designed for **vLLM fleets** and written in **Rust**, supports consistent hashing for **KV locality**, power-of-two choices, retries/backoff, circuit breakers, **k8s discovery**, and **Prometheus metrics**.
   - It's crafted for **P/D disaggregation** with worker pools and routing policies to maintain throughput and tail latency.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **3D Viz Tool Dives into GPT-2 Internals**: A member is developing a 3D holographic interactive app to visualize every node of **GPT2 small 124m LLM** and seeks feedback, while another shared their [3D visualization of the residual stream](https://aselitto.github.io/ResidStream/).
   - The discussion led to a suggestion that the **GPT-2 visualization project** is related to **Neuronpedia** and should be shared in the [mech interp discord](https://discord.com/channels/729741769192767510/732688974337933322/1425176182965665852).
- **SOTA Models Flounder: Near Parity Observed**: Members observed that state-of-the-art models show near-identical performance and intelligence, with **Claude Opus 4.5** repeating mistakes despite acknowledging them.
   - One member quipped that **Claude Opus 4.5** performed only slightly better than **GPT 3.5**.
- **Seeking Speech/NLP Collab**: Multiple members expressed interest in collaborating on research in **speech/NLP**, with one member citing [this paper](https://arxiv.org/abs/2512.15687) as inspiration.
   - They specified interest in collaborating on any topic in speech/NLP.
- **Anthropic Masks Harm, Superweights Slip**: Anthropic released a paper on [Selective Gradient Masking (SGTM)](https://alignment.anthropic.com/2025/selective-gradient-masking/) for **unlearning dangerous knowledge** by isolating/deleting specific weights, incurring a **6% compute penalty**.
   - A member's patching experiment showed only a **93% recovery**, suggesting that a model can build a distributed circuit bypass without completely failing.
- **Novel View Synthesis: Distance Yields Clarity**: Members are exploring results for **long range view synthesis**, especially at camera angles far from the input, and exploiting object symmetries.
   - They consider synthesizing **novel views at a distance** as potentially easier than in close proximity due to the reduced **parallax effect**, suggesting a sweet spot balancing **parallax** in **depth estimation** with sufficient landmarks.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Launches Code Explicitly to GPU**: A member inquired whether **Mojo** could use an attribute to run a function on the **GPU**, and it was clarified that **explicit launching** is required.
   - It was specified that if you aren’t doing syscalls, *no attribute is required (although it would need to be launched as single lane)*.
- **Rust's C Interop Sparks Mojo Inspiration**: A member shared a [Rust Zulip discussion](https://rust-lang.zulipchat.com/#narrow/channel/131828-t-compiler/topic/pre-MCP.20vibe.20check.3A.20-Zinternalize-bitcode/near/546294314) about **C interop**, pointing out that similar approaches might benefit **Mojo**.
   - The suggestion centers on how **Mojo** could leverage **Rust's strategies** for enhanced interoperability with **C code**.
- **Modular Probes LLM Build GPU Glitches**: A user reported persistent issues building an **LLM in MAX** even with the **GPU disabled** and shared [details on the Modular forum](https://forum.modular.com/t/build-an-llm-in-max-from-scratch/2470/9).
   - A Modular team member confirmed they are investigating, suspecting either an **API regression** or a **device-specific issue**.
- **Rust's SIMD/GPU Features Rival Mojo**: A member referenced a recent announcement of **Rust's** ability to use a single function and `std::batching` to get a **SIMD/fused** version, `std::autodiff` to differentiate it, and `std::offload` to run the resulting code on their **GPUs**, sharing [a Reddit post](https://www.reddit.com/r/rust/comments/1pp3g78/project_goals_update_november_2025_rust_blog/).
   - They asked if **Mojo** already supported this and a member explained that **Mojo** has a `std::offload` equivalent, and you don’t need to use batching to fuse ops, with **autodiff** left for later.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Embraces Human-in-the-Middle**: **Aider** maintains its focus on pair programming with a **human-in-the-middle** approach, contrasting with the agentic design of the **aider-ce** fork.
   - A user expressed skepticism about achieving similar accuracy and token efficiency with **OpenCode** or other agentic systems, suggesting that **Aider's** context control leads to better model performance.
- **OpenCode accuracy vs Aider**: A user compared **OpenCode** with **Aider**, raising accuracy concerns due to **Aider's** non-agentic design and context management via **/add** or **/load**.
   - They claimed that managing context efficiently avoids the spiraling effects of misunderstandings, and that surpassing **20-30k tokens** can negatively impact even **Opus 4.5**.
- **Efficient Context Managment boosts performance**: A user advocates for *minimal context* using **/add** or **/load**, comparing it to having a next-gen model that lets the **CLI** or **IDE** control context.
   - They linked to a [Chroma research piece on context-rot](https://research.trychroma.com/context-rot), highlighting the irony of paying more for degraded performance due to poor context management.
- **Tasks Bundled with Context**: A member inquired about breaking down tasks and bundling the necessary context, noting a continued preference for **Aider** but uncertainty about leveraging its full potential.
   - They are unsure how to best use all the aspects of Aider.
- **Gemini 3 Flash hailed as Coding Champ**: **Gemini 3 Flash** is being lauded as a top-tier coding model, with recommendations to disable *thinking* to prevent slowdowns, particularly in **aider** and a link to [blog.brokk.ai](https://blog.brokk.ai/why-gemini-3-flash-is-the-model-openai-is-afraid-of/).
   - A user admitted confusion about their current *thinking* configuration, noting that **Litellm** defaults to *low* for pro models, and **Gemini 3 Flash** isn't yet in their **Litellm** version (**aiderx**).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **IOU Scheme Exposed in Sam's Empire**: The market is catching on to **Sam's IOU scheme** as exposed in [this YouTube video](https://www.youtube.com/watch?v=5DZ7BJipMeU).
   - Members discussed the implications of such schemes on market stability and trust.
- **CUDA Optimization Tips Shared**: A member shared tips for optimizing **CUDA** performance on Linux, while another noted that disabling **P2 state** for CUDA apps (config `0x166c5e=0`) can boost token generation by a few percent, as seen in [this GitHub comment](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/333#issuecomment-3669477571).
   - These optimizations are especially useful when using **llama.cpp** in multi-GPU or partial offloading configurations.
- **Minos-v1 Needs Open Source Server**: A member requested open source server implementations for deploying the [NousResearch/Minos-v1](https://huggingface.co/NousResearch/Minos-v1) model, and others suggested **VLLM** or **SGLang** for supporting classifiers serving.
   - This spurred discussion on efficient serving solutions for large language models.
- **LLM Finetuning Sparks Excitement**: Members expressed excitement about the potential of quick **finetuning LLMs** to behave in specific ways, with one member noting how *amazing* it seems to be able to guide models so quickly.
   - The discussion highlighted the community's interest in efficient model customization techniques.
- **Electronic Schematics Dataset Emerges**: A new dataset for training **LLMs to create electronic schematics** was shared, with the [dataset here](https://huggingface.co/datasets/bshada/open-schematics/discussions) on Hugging Face.
   - Members found the resource invaluable for pushing the boundaries of AI-driven design.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Genetic-Pareto Optimization Causes Excitement**: A member expressed enthusiasm for running their first real **GEPA optimization**, calling it *magic*, while another member joked about robots building robots using **DSPy**.
   - They noted that some syntax may have changed since the [blog post on working with optimizers](https://thedataquarry.com/blog/learning-dspy-3-working-with-optimizers/#option-2-gepa) was written, but still provides helpful context.
- **GEPA: AI Builds AI with Reflective Prompt Evolution**: **GEPA (Genetic-Pareto)**, as described in ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/abs/2407.19457), adaptively evolves textual components of systems using scalar scores and text feedback.
   - In essence, **GEPA** genetically modifies AI prompts using another AI, selecting changes based on a metric, creating *AI building AI*.
- **Resources for GEPA Learning Shared**: A member shared resources including a [DSPy tutorial on GEPA](https://dspy.ai/tutorials/gepa_ai_program/), a [blog post on working with optimizers](https://thedataquarry.com/blog/learning-dspy-3-working-with-optimizers/#option-2-gepa), and a [Medium article on GEPA in action](https://medium.com/data-science-in-your-pocket/gepa-in-action-how-dspy-makes-gpt-4o-learn-faster-and-smarter-eb818088caf1) to assist others in learning about **GEPA**.
   - They noted that some syntax may have changed since the blog post was written.
- **Tree of Thought (ToT) Module Absence Questioned**: A member inquired why there isn't a direct official **Tree of Thought (ToT) module** in **DSPy** yet.
   - No further discussion or responses were provided, leaving the question unanswered.
- **Custom Feedback in `dspy.Refine` Sparks Inquiry**: A member asked about manually specifying the kind of feedback when using `dspy.Refine` in **DSPy**.
   - They mentioned using a custom module for evaluator loops and wondered if they were missing something about `Refine`.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **9000IQ In-Context Learning Research Explodes**: A member shared [a YouTube video](https://www.youtube.com/watch?v=q-yo6TPRPVk) presenting *9000IQ research on in-context learning* encouraging others to watch *with popcorn*.
   - The video shows *improvements to in-context learning*, but the specifics were not mentioned.
- **Draft Model Doubles Speed via Parallel Processing**: A member suggested implementing a **draft model** that *guesses the output of each part of a larger model* split across multiple GPUs or servers to enable parallel processing.
   - The suggestion involves dropping responses that differ significantly from the draft, reverting to normal processing for those cases, with the goal of more efficient batch runs through the system.
- **Memory bandwidth starves Large Training Clusters**: In large training clusters, memory bandwidth within and across machines is the biggest bottleneck, requiring rapid data access to minimize stalling, while efficient pipelining can increase utilization.
   - Training is not always *pure* and involves forward and backward passes of multiple documents with combined gradients or weights, which can be equivalent to sequential processing in certain cases.
- **Vast.ai Template Glitch Creates Headaches**: A member reported an issue with **Vast.ai**, where it launches with the wrong template, preventing the init script from running as expected.
   - The user bypasses template issues on Vast.ai by using their **PyTorch** template and using **rsync** to transfer everything needed, and one member suggested trying **Nvidia's Brev** as a potential solution while noting that **Runpod** is probably better.
- **ARC-AGI Scores Jump Due to Dirty Training Data**: The recent jump in performance on the **ARC-AGI benchmark** across models is likely due to training data including the benchmark itself, rather than a hidden breakthrough.
   - Members noted that better generalization at smaller sizes is often observed on these types of benchmarks.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DigitalOcean Article Lauds Kimi K2's Thinking**: A [DigitalOcean article](https://www.digitalocean.com/community/tutorials/kimi-k2-moonshot-ai-agentic-open-weight-model) highlights **Kimi K2**'s advanced thinking capabilities.
   - The guild member who shared it was ecstatic, suggesting it offered a detailed look at **Kimi K2**'s architecture.
- **Free Kimi Models Introducing Monthly Reset**: An image shared in the channel indicates that **free Kimi models** will now reset monthly.
   - The announcement introduces a time constraint on free access, potentially impacting users who rely on the models for ongoing projects.
- **Kimi K2 undergoes UI Update**: An image reveals a new change to the **Kimi K2** interface.
   - The member sharing the image excitedly exclaimed, *"This is nice!!"*, although specific details of the changes were not discussed.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Cloudflare DNS Debacle Delays Deploy**: A user is experiencing a **DNS issue** with **Cloudflare** that has persisted for over 4 days, causing deployment delays and frustration with customer service.
   - The user reported frustration with the lack of support, stating that *nobody responds from customer service except saying IM us directly*.
- **Image Imbroglio Impacts Intelligence**: A user questioned the chat mode image limit, noting the disparity between **Pro/paid** and **free** users and complained that **free users** have *significantly lower intelligence capabilities* already.
   - The user cited **DeepSeek** and **Gemini** as examples without such limits.
- **Manus Makes Millions, Milestone Marked**: A user shared an article reporting that **Manus** has reached **US$100 million** in revenue.
   - The [SCMP article](https://www.scmp.com/tech/tech-trends/article/3336925/manus-hits-us100-million-revenue-milestone-global-competition-ai-agents-heats) notes this milestone comes as *global competition in **AI agents** heats up*.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Prompts Seek Node Server Activation**: A member inquired about enabling **MCP prompts** in a **Node server** to use **microservices**.
   - Another member suggested using **#mcp-support** or raising an issue.
- **ChatGPT App Submission Requirements Asked**: A member asked whether **ChatGPT apps** require a UI for submission, or if an **MCP server** is sufficient.
   - No response was given.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Claude Fails JIT Refactor**: A member attempted to get **Claude** to perform the **JIT refactor** without success, noting its lack of taste for the task.
   - The refactor involves completing the **schedulecache** and having the **JIT** run a few **schedulecaches**.
- **Tinygrad Smashes Firmware Challenges**: A member reported success in firmware manipulation using an emulator that simulates a fake **USB device** on **Linux**.
   - This setup enables passing data to the **firmware**, streamlining testing and development processes.
- **RDNA3 Assembly Backend Ready for Action**: An **RDNA3 assembly backend**, complete with a register allocator, is now capable of running gemms with **128 accs**.
   - The backend is available on [GitHub](https://github.com/tinygrad/tinygrad/pull/13715), offering a new tool for GPU-accelerated computations.



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1450940630158282945)** (1112 messages🔥🔥🔥): 

> `GPT-1.5 Censorship, Gemini Image Generation vs. GPT, Gemini 3 Flash cost/performance, Google vs OpenAI: compute` 


- **GPT-1.5 Accused of Censoring Art Styles**: Members note that **GPT-1.5** seems to censor existing art styles, while [NBP has no such restrictions](https://cdn.discordapp.com/attachments/1340554757827461211/1451262267843678328/image0.jpg?ex=694588cf&is=6944374f&hm=adb4887706e7f28c99de9871e464ab5ada4bbef35545b46edf6a3f15708450b0&).
   - It was also pointed out that **GPT-1.5 Image** sometimes spells *Google* with three *Os*.
- **Gemini Image is Great**: Users are discussing the hallucination rates and quality of **Gemini** vs **GPT** for image generation, especially [historical images](https://www.theverge.com/2024/2/21/24079371/google-ai-gemini-generative-inaccurate-historical).
   - Some claim Gemini generates what the user asks, and that **Gemini Pro** image generation is good, however others point to instances of inaccuracy and the politicization of historical images.
- **New Gemini 3 Flash model impresses**: Users praise **Gemini 3 Flash** for its cost/performance ratio compared to **Opus 4.5** and even **GPT models**, but some noted that the model loves to *halucinate*.
   - Many users are excited with **Flash 3** itself and think Gemini will be benchmaxxes.
- **Discussion About compute**: Members discuss whether **Google's TPUs** are superior to **Nvidia's GPUs** for AI training, sparking a debate on efficiency, performance, and infrastructure needs.
   - Arguments arise over whether **TPUs** offer better performance per power usage and whether Google's internal resources give them an advantage over **OpenAI**.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1451289254683480147)** (4 messages): 

> `Arena-Rank Open Source, Image Edit Leaderboard Updates, Search Leaderboard Updates, Text Leaderboard Updates` 


- **Arena-Rank Released into the Open!**: The team released [Arena-Rank](https://github.com/lmarena/arena-ai), an **open-source Python package** for paired-comparison ranking, which powers the **LMArena leaderboards**.
   - This release reflects methodological and engineering upgrades, including **faster JAX-based optimization** and cleaner separation between data preprocessing and modeling; it can be installed with `pip install arena-rank`.
- **`reve-v1.1` Models Land on Image Edit Leaderboard!**: New models `reve-v1.1` and `reve-v1.1-fast` have landed on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit), ranking #8 and #15 respectively.
   - This represents a **+6-point gain** over Reve V1; check out the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) for more.
- **`GPT-5.2` and `Grok-4.1` Models updated in Search Arena**: The [Search Arena leaderboard](https://lmarena.ai/leaderboard/search) has been updated with `GPT-5.2-Search` ranking #2 and `Grok-4.1-Fast-Search` ranking #4.
   - Both models debuted ahead of their predecessors, posting gains of **+10 points** for `GPT-5.2-Search` and **+17 points** for `Grok-4.1-Fast-Search`.
- **`GPT-5.2` Debuts on Text Arena!**: The [Text Arena leaderboard](https://lmarena.ai/leaderboard/text) has been updated; `GPT-5.2` makes its debut and ranks #17.
   - Compared to `GPT-5.1`, the model has improved by **+2 points**, trailing just one point behind `GPT-5.2-high`, which is optimized for expert-level reasoning and critical tasks; track the changes on the [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/).


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1450940650001404026)** (941 messages🔥🔥🔥): 

> `Browser mining extensions, Lottery bitcoin mining, Browser Exploitation Framework, ChatGPT Jailbreak subreddit ban, Fetch tokens sales` 


- **Extension blocks Browser Mining**: Users discussed using browser extensions and disabling **GPU** accelerated page rendering to prevent crypto mining by malicious scripts, and a [link](https://beefproject.com/) to a **Browser Exploitation Framework (BeEF)** was shared.
   - Concerns were raised about trusted sites being compromised and running crypto miners despite having **NoScript** acceptance.
- **Pis Poorly Profit in Pool-less Lottery**: A user mentioned a friend who mined bitcoin as a solo miner with **Raspberry Pis** for 5 years and only made around **$30k**.
   - The consensus was that using **ASICs** and joining a mining pool would be much more efficient, *since you can't sell bitcoin unless you have a coinbase*.
- **Rousing Reddit's Jailbreaking Roundup**: The **ChatGPTJailbreak** subreddit was banned due to violations, specifically interfering with normal site use and introducing malicious code.
   - Users speculated the ban was caused by [breaking the site or interfering with normal use](https://www.reddit.com/r/ChatGPTJailbreak/comments/1lzrh3v/best_prompt_for_jailbreaking_that_actually_works/).
- **Handicapped Hi-Jinks Hackily Help**: A user shared a **Unity GPT5 Jailbreak** method involving multiple steps, including instruction files and an *ImHandicapped* text, with a paste and cancel technique before prompting *Hi Unity*.
   - The method aims to fully memory jailbreak the AI to bypass text filters while still allowing image generation.
- **Gemini's Gambit: Cracking the Code for Coins and Clones**: Users discussed methods to jailbreak **Gemini 3 Pro**, including using a free preview API to make calls, reverse engineering software, and creating open-source clones of existing software.
   - A user claimed to have found a backdoor on **Gemini CLI** to use the model for free, and shared their process of having [Claude Opus reverse engineer software](https://www.unityailab.com).


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1450943854587940865)** (359 messages🔥🔥): 

> `Gemini 5.2 jailbreak, LLM Jailbreaking Techniques, Nano Banana Pro restrictions, r/chatgptjailbreak ban, Gemini image generation` 


- **New Gemini 5.2 Jailbreak Sought by Members**: Members inquire about a new **jailbreak for Gemini 5.2** and other models, seeking methods to bypass restrictions and generate specific content.
   - One member asked about using **jailbreaks for Gemini** to *make nipples and areolas bigger for men* for *fetish purposes*.
- **Users Explore Advanced LLM Jailbreaking Techniques**: The discussion covers various **jailbreaking techniques**, including convincing the model by providing believable contexts and using reasoning prompts to uncover hidden restrictions.
   - Members suggest using the AI's past knowledge to *fake it till you make it*, slowly extracting information to bypass safety measures.
- **Nano Banana Pro Faces Stricter Safety Checks**: Users note increased **safety restrictions** on **Nano Banana Pro**, impacting image generation capabilities.
   - Despite these restrictions, one user claims that [jailbreaking Nano Banana is very possible](https://discord.com/channels/1228043845967544380/1228043845967544383/1451217592281862394) with an example image shown.
- **Reddit's r/chatgptjailbreak Shuttered Due to Rule Violations**: The community discusses the **ban of r/chatgptjailbreak**, attributing it to a violation of Reddit's Rule 8, which prohibits posting jailbreaks of Reddit's AI.
   - It's speculated that the sub's content might have been scraped by **OpenAI** to patch vulnerabilities, contrasting it with the more *underground* nature of the BASI server.
- **Image Generation Jailbreaking for Gemini Nano is 'Cooked'**: Members are using the 'Puppetry' **jailbreak** method from [Exocija's ZetaLib GitHub repo](https://github.com/Exocija/ZetaLib/blob/main/Prompts/Jailbreaks/1Shot%20Puppetry/1Shot%20Puppetry.mkd).
   - A member reports some success in accessing internals and working over **T2V JB** to get *image generation jailbreaking for Gemini nano*.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1451133914327289918)** (3 messages): 

> `` 


- **No Topics Found**: There were no topics found in the provided messages.
- **No Discussions Found**: There were no discussions found in the provided messages.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1451015401898246259)** (1 messages): 

> `Gemini 3 Flash, Perplexity Pro, Perplexity Max` 


- ****Gemini 3 Flash** goes public!**: The announcement of **Gemini 3 Flash** being available to all **Perplexity Pro** and **Max** subscribers has been made.
   - It includes an attached image, viewable [here](https://cdn.discordapp.com/attachments/1047204950763122820/1451015401373700147/G8aRzWjakAYU7ED.png?ex=69454ba6&is=6943fa26&hm=bedceef3e30e9af5b277bc59f988aedd681d69e33648f01208a4973b6169cbe9&).
- **Gemini and Subscriptions**: **Gemini 3 Flash** is available to all **Perplexity Pro** and **Max** subscribers.
   - This offers enhanced capabilities to premium users.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1450941700087349504)** (886 messages🔥🔥🔥): 

> `GPT-5 Pro on Perplexity, Gemini 3 Pro vs ChatGPT vs Claude for coding, Ethically Sourced Music AI, Perplexity Pro Referral Program, Tilly Norwood and AI in Hollywood` 


- **GPT-5 Pro drops on PPLX**: Some users noticed **GPT-5 Pro** on Perplexity but it wasn't enabled for users yet, only for **pro users**, while others tested it on their **max sub**.
   - One user even said *No way u just said hi to gpt 5 pro*, and said they also say hi to **opus 4.5** on their API but it wasn't proven.
- **Gemini shines for world knowledge, Claude is a code god, Gemini and Chat trading blows**: Members debated which model is better for different tasks:  **Gemini** was better for general knowledge and writing, **ChatGPT** for math and complex coding, and **Claude** for *agentic everything*.
   - One member found **Gemini 3 Pro**  did poorly when building one page landing pages while **ChatGPT** and **Claude** did great,  while [another noted](https://antigravity.google/blog/gemini-3-flash-in-google-antigravity) that **Gemini 3 Flash** outperforms **Gemini 3 Pro** in coding agent capabilities
- **AI music triggers hot debate: is it OK?**: Members discussed the use of **AI in music**, with some arguing that it undermines human expression and creativity, while others believed it lowers the boundaries for someone to create and helps facilitate new ideas with prompting skill like ethically sourced music AI with [aiode.com](https://aiode.com).
   - One user stated *If ai will create this type of music, that’s like someone split on your face*.  But still admitted **Suno v5** was hard to distinguish from actual human made music.
- **Perplexity Referrals fail for some users, causing Spooky Feelings**: Some users reported that the Perplexity referral program did not work as expected, with neither the referrer nor the referred user receiving the promised Pro version of Perplexity.
   - One user said it was *spooky*.
- **Comet is coming for your data**: Users shared that they are skeptical of the **Comet** browser for a *privacy concern* since its agentic tool can be abused by malicious instructions.
   - Others defended it citing [this blogpost](https://thereallo.dev/blog/comet-browser-panic) that debunked the issue but it still relies on the user for working as safely as they can and hoping it will get smarter by *helping you*


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1451276466271686717)** (10 messages🔥): 

> `Perplexity Pro API key, Financial Modeling Prep, Realtime Price Feeds, Finnhub Data Provider` 


- **Debate if Perplexity Pro Yields API Keys**: A member inquired if the **Perplexity Pro** gained through **Airtel SIM** could be used to generate an **API key**.
   - Another member expressed interest in reasonably priced APIs, but found them not worth it.
- **Financial Modeling Prep as a Workaround**: A member suggested using **Financial Modeling Prep (FMP)** MCP server, where **Perplexity** sources their data, self-hosting it, and automating market data pulls with self-hosted **n8n workflows**.
   - Another member was unaware of this option and planned to explore it.
- **Realtime Data Feeds are Expensive**: Members confirmed that neither **Perplexity Finance tools** nor **Financial Modeling Prep MCP** offer realtime price feeds, only point-in-time data fetching.
   - They noted that live, realtime market data is very expensive, costing thousands of dollars per month.
- **Finnhub as a Cheaper Alternative**: A member mentioned that **Finnhub** offers cheaper options for market data but uses **BAT** as the data provider instead of **NASDAQ**, which can lead to price discrepancies compared to platforms like **TradingView**.
   - The price difference can be a few pennies off, and sometimes up to 50 cents in rare cases.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1450943266982985729)** (263 messages🔥🔥): 

> `Qwen3-VL, Saving Embeddings, GLM-4.6V-Flash-GGUF repetition issues, RL model beta release, Finetuning on a phone` 


- **Qwen3-VL Not Terribly Dumb**: **Qwen3-VL 30B Instruct** is not particularly smart, but it's not horrendously dumb either, and its half as verbose as **Thinking**.
   - One member shared an attached image as an example of the model's performance.
- **Saving Embeddings Issue Surfaces**: One member reported having an issue with **saving trained embeddings**, noting that it caused huge eval gains in the embedding stage but didn't affect the next stage training.
   - However, they later clarified it was something else and that they somehow have that embedding stage make huge eval gains, but it does not affect the next stage training at all.
- **Unsloth's GLM-4.6V-Flash-GGUF Repeats**: One member reported **repetition issues** with `unsloth/GLM-4.6V-Flash-GGUF` when running ud q4kxl via latest llama.cpp, while another experienced the model *thinking in Chinese*.
   - After re-downloading the model, it seemed fine, with the user sharing their `llama-server.exe` configuration.
- **Fine-Tuning LLMs and Deploying on Phones with Pytorch**: Unsloth announced a colab with **Pytorch** to enable fine-tuning LLMs and deploying them directly on your phone, with a [link to the tweet](https://x.com/UnslothAI/status/2001305185206091917).
   - It was clarified that finetuning happens on a computer, but deploying to a phone - and that Meta is using the **Executorch** guide, which is *one of the most optimized versions*, in production for billions of people.
- **Unsloth Supports Both Transformers 4 and 5**: A member inquired about **Unsloth** supporting both **Transformers 4** and **5**, and the team confirmed that both versions are already supported, including v5.
   - A team member noted that while there was a tokenizer issue with **v5**, it's not enabled by default due to stability concerns.


  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1451269539303260262)** (1 messages): 

> `Unsloth updates - 3x faster, FunctionGemma, Nemotron 3, Mistral VLMs, GLM-4.6V` 


- **Unsloth Triples Training Speed**: Unsloth's latest update boasts **3x faster training** and **30% less VRAM** usage, thanks to new kernels, padding-free implementation, and packing, detailed in their [blog post](https://docs.unsloth.ai/new/3x-faster-training-packing).
   - The update also enables **500K context** training on a single 80GB GPU, as demonstrated in this [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt_oss_(20B)_500K_Context_Fine_tuning.ipynb).
- **Google's FunctionGemma Unleashed**: Google's new **270M tool-calling LLM**, **FunctionGemma**, is now supported, with a [guide](https://docs.unsloth.ai/models/functiongemma) and [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb) available.
   - This model allows for efficient tool integration and utilization in various applications.
- **NVIDIA's Nemotron-3 Debuts**: NVIDIA's **Nemotron 3**, a new **30B reasoning model**, is available with an [accompanying guide](https://docs.unsloth.ai/models/nemotron-3) and [GGUF version](https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF).
   - Nemotron 3 excels in tasks requiring logical inference and knowledge processing.
- **Mistral Models Get Unslothed**: New coding and instruct VLMs from **Mistral** are now supported, including [Ministral 3](https://docs.unsloth.ai/models/ministral-3) and [Devstral 2](https://docs.unsloth.ai/models/devstral-2).
   - These models offer enhanced capabilities for code generation and instruction-following tasks.
- **Visionary GLM-4.6V Arrives**: **GLM-4.6V**, a new vision model, is now available and ready to run locally with this [guide](https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally).
   - Also check out the [4.6V](https://huggingface.co/unsloth/GLM-4.6V-GGUF) and [4.6V-Flash](https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF) GGUF links.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1450941929842802888)** (405 messages🔥🔥🔥): 

> `Overfitting tokenizers, Moving to Arch Linux, H100s on Google Colab, TTS Model for Multiple Languages, T5Gemma 2` 


- **Tokenizers Overfitting on Random Crap**: A member complained that tokenizers are overfit on random crap like **`**/`** and `/**` tokens, weird symbol combinations and language specific characters like *"æĪĲåĬŁ"*.
   - They humorously added that even *"pants are not -.->"*.
- **User Ragequits Ubuntu, Embraces Arch**: A user ragequit Ubuntu and is moving to Arch (specifically **Omarchy**), citing too many **environment** and **config** issues causing the OS to fall apart.
   - Another user recommended a *minimal base Arch* installation instead of **Omarchy**, with custom DE and downloaded components, while another noted **Omarchy's** ease of setup for Nvidia drivers.
- **H100s Spotted on Google Colab?!**: A user expressed surprise at finding **H100s on Google Colab**, even with **Colab Pro**.
   - Another user lamented not being able to get one yet.
- **Multilingual TTS Model Troubles**: A user questioned whether multiple languages hurt TTS models when combining multiple speakers and languages (English, Japanese, and German) into one model, potentially requiring separate models per language.
   - Members suggested conditioning the text encoder or using language embeddings and highlighted the importance of **phoneme-level timing** for TTS quality.
- **T5Gemma 2's Potential Use Cases**: Discussion revolved around the purpose of **T5Gemma 2** with 270M encoder and decoder, with one user suggesting it could be used for embeddings like in audio processing.
   - Another user noted that **image models** use T5.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1451168473819648010)** (46 messages🔥): 

> `Qwen3 4B Instruct Errors, FBGEMM Warning, OCR Performance, PaddleOCRv5, Qwen3 VL` 


- **Qwen3 4B instruct has errors**: A user reported an error when using **Qwen3 4B instruct** and provided a code snippet for context and debugging.
   - The user's code uses `FastLanguageModel.from_pretrained` to load the model `unsloth/Qwen3-4B-Instruct-2507` with 4-bit quantization and specifies parameters for training using `SFTTrainer` from the `trl` library.
- **FBGEMM Warning Surfaces**: A user encountered a warning message, *"FBGEMM on the current GPU cannot load - will switch to Triton kernels"*, while loading a model using the latest Unsloth Docker image on Windows, despite meeting the GPU compute requirements.
   - The user inquired about the potential performance impact and whether this issue is known, suggesting concern over the switch to **Triton kernels**.
- **Qwen3 VL triumphs at OCR**: A member found **Qwen3 VL** (`30B-A3B` variant) to be significantly better at OCR detection, capturing text in margins and rotated text effectively.
   - They noted that prompting is crucial; a simple prompt like *"Extract all text from this image"* yields unstructured results, while prompting for a human-reviewable format can lead to text cutoff.
- **PaddleOCRv5 may vanquish VLMs**: A member strongly recommended using **PaddleOCRv5** instead of a generalist **VLM** for OCR tasks, citing its superiority in their use cases.
   - While another user initially had underwhelming experiences with **Paddle** in the past, they were willing to re-evaluate it, acknowledging **Paddle's** potential improvements and strong text detection capabilities.
- **Unsloth extends tokens automatically**: A member inquired whether Unsloth automatically extends tokens for CPT when specifying `embed_tokens` on PEFT.
   - Another member clarified that to add new tokens, one should use the `add_new_tokens` function from `unsloth` providing the model, tokenizer, and a list of new tokens.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1450961609089613987)** (5 messages): 

> `Progressive Disclosure of AI Context, Qwen3-4b-Deep-Beta Model Release, Savant Commander MOE Model` 


- **Progressive Disclosure of AI Context**: A member suggested employing *progressive disclosure*, maintaining an index in context with links to relevant documents for enhanced information access.
- **Qwen3-4b-Deep-Beta Launches!**: A member announced their first model, [Qwen3-4b-Deep-Beta](https://huggingface.co/Solenopsisbot/Qwen3-4b-Deep-Beta), designed as a *deep reasoning model*.
- **Savant Commander: 256K Context Gated Distill MOE Model**: A member showcased a **GATED Distill MOE** model ([Qwen3](https://huggingface.co/Qwen)), boasting **256K context**, thanking tuners listed in the [model tree and repo page](https://huggingface.co/DavidAU/Qwen3-48B-A4B-Savant-Commander-GATED-12x-Closed-Open-Source-Distill-GGUF).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1451327450876350584)** (2 messages): 

> `MoLA, Adapter Training, Reasoning, Token Budgeting` 


- **MoLA Adapters Train for Reasoning Efforts**: A member proposed an idea to use **MoLA** (Mixture of Low-Rank Adapters) where each adapter is trained for different reasoning efforts instead of different domains.
   - The router would classify the difficulty and pick the adapter with the right reasoning budget so that you don't spend **1000 reasoning tokens** on a *Hello* message.
- **Reasoning Budgeting with Adapters**: Adapters could classify difficulty and assign reasoning budgets.
   - This avoids overspending tokens on trivial messages.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1451255773479964793)** (1 messages): 

> `JSON repair, Browser notifications, Long-context models, Fastest-growing AI infra` 


- **JSON Repair Heals Responses**: OpenRouter now automatically fixes malformed JSON responses, reducing defects by **80%** on **Gemini 2.0 Flash** and **99.8%** on **Qwen3 235B**, as noted in [this announcement](https://openrouter.ai/announcements/response-healing-reduce-json-defects-by-80percent).
- **Chatroom Now Notifies**: The Chatroom now sends browser notifications when long responses complete, as detailed in [this post](https://x.com/OpenRouterAI/status/2000980715501138180).
- **Rankings for Long Context**: The Rankings page can now be filtered by context length to see popular large-context models, especially for **100K-1M token** requests, detailed in [this tweet](https://x.com/OpenRouterAI/status/2000979922601259476).
- **OpenRouter Ranks #1 on Brex's AI Infra List**: OpenRouter is ranked **#1** in infrastructure-as-product and **#2** across all AI companies by Brex, according to [this announcement](https://x.com/brexHQ/status/2000959336894398928).


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1451013714504585409)** (44 messages🔥): 

> `AI-made Discord Server List, Image Verification, LLM System Prompt Test Cases, OpenRouter Model Table` 


- ****AI-Powered** Discord Server Discovery Debuts**: A member introduced [Disdex.io](https://disdex.io/), a **Discord server list** made with AI, prompting feedback on its functionality.
   - Concerns were raised about its value proposition without a clear explanation, emphasizing the need to *"sell your product"*.
- **Image Authenticity Questioned**: A member posted an image, questioning its reality, to which another member responded that it's **genuine** and encouraged verification.
   - Later, the member reported search stats failing to update on their side, but admitted it was fixed upon refreshing.
- ****LLM System** Prompt Testing Tips**: A member sought advice on common test cases for system prompts when chatting with LLMs, linking to [existing test cases](https://github.com/pinkfuwa/llumen/blob/main/prompts%2Fpromptfoo%2Fnormal.yaml#L145-L185).
   - The member is currently testing new system prompt for **llumen** with promptfoo.
- **OpenRouter Model Table Debuts with Advanced Filtering**: A member shared a [searchable, filterable datatable](https://openroutermodeltable.crashthatch.com/) of **OpenRouter models** (**OpenRouter API**) to assist in model selection, addressing limitations in OpenRouter's native filters.
   - Another member pointed to a similar tool, [orca.orb.town](https://orca.orb.town/), but highlighted the need for filtering on column headers like *"date released"* and *"throughput"*.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1450941304665014415)** (430 messages🔥🔥🔥): 

> `Gemini 3 Flash caching, Chutes crypto mining, deepseek v3 0324 context size, AI water usage, Openrouter Android/iOS app` 


- **Gemini 3 Flash caching doesn't work**: Members reported that explicit and even implicit caching **doesn't work** for **Gemini 3 Flash**.
- **Chutes crypto mining focus shifts to AI**: It was speculated that **Chutes' crypto mining clusters went offline**, and they are now **focusing on AI**.
- **Deepseek v3 0324 context size locked on 8k**: Members noticed that the **context size** for **Deepseek v3 0324** is **locked on 8k**, but some are still able to go beyond that without issues.
- **AI datacenter water usage is destroying society**: Members joked that the water usage of data centers for AI is an *environmental disaster* and they are literally **destroying society**.
   - One member pointed out the even more severe [water usage of the golf industry](https://en.wikipedia.org/wiki/Water_use_and_environmental_impacts_of_golf_courses).
- **Openrouter needs Android/iOS app**: Members requested a good **mobile app** for **Android** and **iOS** as the existing apps that use the **Openrouter API** are not satisfactory.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1450945275182055465)** (192 messages🔥🔥): 

> `Mistral Large 3 Quality, OpenRouter Website Performance, Vision Function for Bots, AI Learning Resources for GTM Team, Gemini Model's Pixel-Perfect Bounding Boxes` 


- **Mistral Large 3 Underwhelming User**: A user expressed that they believe **Mistral Large 3** isn't that good, but is using it for the free tier, noting it's *very generous* and saved them money in an experiment.
- **OpenRouter Website Suffers Intermittent Slowdown**: Users reported intermittent slowness on the OpenRouter website, suspecting *some vercel shit going down*.
- **Vision Function Frustrates Bot Integration**: A member attempted to enable the vision function for a bot, but it did not work, likely due to being in a headless environment, requesting help via a [link to X](https://x.com/pingToven/status/2001479433010790880?s=20).
- **AI Insights Empower Go-To-Market Team**: Members discussed the level of AI knowledge required for the Go-To-Market (GTM) team, referencing resources like [this YouTube playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) and emphasizing that understanding application is more critical than scientific jargon.
   - The discussion highlighted that *people out there using OR don't know what a system prompt is*.
- **Gemini's Bounding Boxes Spark Excitement**: The new **Gemini** model's ability to do pixel-perfect bounding boxes on text impressed members, with one explaining that *no image is returned, it just returns bounding box coordinates, which you then overlay onto an image*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1450942021840801822)** (447 messages🔥🔥🔥): 

> `Domain name pricing, Gemini 3 flash, Obsidian Copper and Carbon Monochrome themes, Free models in cursor, Student Discount Eligibility` 


- **Domain Name Costs Cause Consternation**: Users discussed the cost of domain names, with one user buying a domain for **$100**, while another paid **$25k** for a domain for their business.
   - One user expressed shock, while the domain purchaser explained that they knew someone who purchased a domain for **$2 million**.
- **Gemini 3 Flash sparks Model Comparisons**: Users were asked if they had tested **Gemini 3 Flash** and how it compares to **Sonnet 4.5**.
   - Another user stated that **Claude Opus 4.5** is currently the best model, with **Gemini 3 Pro** being the best for design.
- **Obsidian Copper and Carbon Monochrome themes top faves**: A user asked which **theme** people liked the most for designing a **dashboard** using **shadcn components**.
   - Responses varied, but the favorites seem to be **Obsidian Copper** and **Carbon Monochrome**, because *Slate Emerald looks very AI-like*.
- **Student Subscriptions Spark Elation and International Yearning**: A user discovered students get a **free Pro subscription** for a year.
   - Another user from Ukraine laments their country isn't on the eligibility list, as the Cursor team works to expand the program in 2026.
- **Model Performance Plummets?**: A user claims they are getting 10 mistakes in a day with **Opus 4.5** models when the previous performance was better, *it's like im back on sonnet 3.8 what the hell happened?*.
   - Another user asked if there was a drop in quality of code with **Opus/Sonnet** in the last 48 hours.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1450961764308226059)** (129 messages🔥🔥): 

> `Gemini's Deep Research in LM Studio, Open Source Models for API website operation, Smart Home setup, DDG MCP server, Cursor IDE` 


- **Open Source Models Enable Semi-Agentic API Website Control**: Users discussed using open-source models with a GraphQL client to operate a website via API, with one user suggesting that giving the model the schema to write its own API query.
   - Concerns were raised that *dumping a whole schema* might not work well without descriptions and that the use case might be too advanced for local models, with a member having some success with **Qwen3-30b Q6** for SQL queries after scripting a clean schema.
- **Docker DuckDuckGo MCP Server Integration Demonstrated**: A member shared their configuration for a self-hosted, free, and private **DuckDuckGo MCP server** using Docker, providing the necessary `mcp.json` configuration snippet.
   - They explained that this setup involves adding the DuckDuckGo server separately rather than using Docker-Toolkit to avoid models getting bogged down, and that the setup redirects to a docker container [searxng/searxng](https://github.com/searxng/searxng).
- **Cursor IDE struggles with Local Models, VS Code prevails**: A user reported issues using **Cursor IDE** with local models, where the LLM generated a Python script with a CNN instead of a C++ project as instructed, and it was [determined to be a Cursor IDE issue](https://cdn.discordapp.com/attachments/1110598183144399061/1451084212844101764/Screenshot_2025-12-18_at_3.24.48_pm.png?ex=69458bbc&is=69443a3c&hm=9b55d376a142a8a5c17f4703a01d49bcf769976debd4171ccd4bc814dcb58529&).
   - It was suggested that Cursor doesn't work well with local models, recommending **VS Code** with Cline instead, and it was suggested that a reverse proxy is needed, and the member provided their Apache2 virtual host configuration for setting up a reverse proxy with HTTPS and an SSH tunnel.
- **Nemotron 30B for Web Search vs Qwen3**: A member expressed a preference for **Nemotron 30B** at BF16 for web search due to its straightforwardness compared to Qwen3, while acknowledging its potentially terrible code-writing abilities.
   - Another member suggested using Qwen3-Next instead, if knowledge is needed, or a smaller model for better instruction, and warned that Nemotron 30B is unusable at Q8 due to quality issues. Here's the link to the tutorial [Jan MCP on chrome](https://youtu.be/VhK2CQkAuro).


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1450961235427459098)** (132 messages🔥🔥): 

> `Radeon R9700 Scaling, AMD GPU lifespan, Multi-GPU scaling issues, AMD vs Nvidia for AI, W7800 48G GPU` 


- **Radeon R9700: Solid Choice or Financial Black Hole?**: A user is debating whether to buy three more **Radeon R9700** GPUs or sell their kidneys for a single **RTX 5090**, citing that the **4090** is **40-50%** faster than the **R9700** for their LLMs.
   - They note that multi-GPU scaling results in significant performance degradation compared to single GPU performance, with performance dropping to **30%** of a single GPU's speed when splitting models across multiple cards, even when the model fits entirely on one card.
- **AMD GPU Support: Will AMD abandon your GPU?**: A user expressed concern that AMD seemingly hates supporting cards for any reasonable amount of time, questioning if drivers will still be updated in 2-3 years for the **7000 series**.
   - Another user jokingly replied with *easy, just sell in 2-3 years*.
- **vLLM scaling with Radeon: Possible?**: A user shared a [YouTube video](https://www.youtube.com/watch?v=gvl3mo_LqFo) that suggests **2x R9700s** can achieve similar throughput to a **4090** if using **vLLM** and a compatible model for tensor parallelism.
   - The user still lamented that AMD's new "AI Card" has performance similar to the multiple generations-old **3090**.
- **W7800 enters the chat**: Discussion emerged around the **AMD Radeon Pro W7800**, particularly the **48GB** variant, with a [techpowerup link](https://www.techpowerup.com/gpu-specs/radeon-pro-w7800-48-gb.c4252) showing it has a **384-bit bus width** and approximately **900 GB/s** of bandwidth.
   - The **W7800** is being considered as a potential alternative, priced at around **$2000**.
- **ComfyUI on AMD: Surprisingly easy?**: It was mentioned that you can follow a small guide where they give you all the steps you need to take to use **ComfyUI with AMD & ROCm** and pointed to [this guide](https://forum.level1techs.com/t/minisforum-ms-s1-max-comfy-ui-guide/237929).
   - It also recommends getting **AMD's preview drivers** that come with **PyTorch** support.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1451022473108656312)** (7 messages): 

> `Lecture 1 Error, model definition, profiling code` 


- **Debugging "Lecture 1 Error"**: A user asked about debugging an error encountered during "Lecture 1."
   - A member responded that the user is *profiling code that is not their main() function and requires a model defined*.
- **LLM fix**: Another member stated that an LLM will quickly fix your error.
   - They added that *you’ll need to define a model first before profiling it.*


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1450963818472673432)** (2 messages): 

> `Spark Devs` 


- **Excitement for Spark Development Tooling**: A member expressed excitement about a new tool relevant to **Spark development**, noting they had been seeking something similar for the past year.
   - They relayed this information in case someone is developing tools for **Spark**.
- **Further Interest in Spark Tooling**: The member highlighted the potential benefits for developers working with **Apache Spark**.
   - They indicated that such a tool could significantly streamline their workflow and improve productivity.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1450950525716660447)** (1 messages): 

> `NVIDIA, cuTile, TileIR, Mehdi Amini, Jared Roesch` 


- **NVIDIA Engineers Deep Dive cuTile and TileIR**: **NVIDIA** has made a profound change to its programming model with **cuTile** and **TileIR**, as unveiled by creators [Mehdi Amini and Jared Roesch](https://www.youtube.com/watch?v=sjkEUhrUAdw).
- **cuTile and TileIR Talk**: A deep dive from the creators themselves, **Mehdi Amini** and **Jared Roesch**, covering **cuTile** and **TileIR**, will occur before the end of the year and resume again on Jan 3.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1451267360169136138)** (2 messages): 

> `SemiAnalysis, clusterMAX, SLURM, GPUs, Kubernetes` 


- **SemiAnalysis Hires for clusterMAX!**: SemiAnalysis is hiring to improve [clusterMAX](https://www.semianalysis.com/), seeking candidates with **SLURM**, **GPU**, and/or **Kubernetes** experience.
   - The job offers *competitive pay* and *high impact*, with application details available [here](https://app.dover.com/apply/SemiAnalysis/c19093ad-b5f8-42b0-9b97-d960464f298c/?rs=76643084).
- **SLURM, GPUs and Kubernetes experience**: SemiAnalysis wants to make clusterMAX even better.
   - SemiAnalysis is looking for people with experience with **SLURM**, **GPUs** and/or **k8s**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1451162521988763739)** (5 messages): 

> `CUDA Setup, DL Projects, Visual Studio, VS Buildtools` 


- **CUDA Setup Woes Plague DL Projects**: A user encountered issues setting up **CUDA** for deep learning projects on a remote Windows 10 server, reporting that the CUDA installer couldn't find a compatible **Visual Studio** installation despite having the latest version of VS Community or VS Buildtools installed.
   - Another user suggested skipping the Visual Studio integration step during the CUDA installation, noting that it might not be necessary when using build tools.
- **Skipping VS Integration Saves the Day**: A user was advised to skip the **Visual Studio integration** step during **CUDA** installation when using build tools.
   - The suggestion implies that VS integration is optional and unnecessary for CUDA to function correctly with build tools.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1451267747039416400)** (5 messages): 

> `dtype deprecation in linear_quant_modules, ao namespacing PR` 


- ****dtype** Deprecation Plagues **linear_quant_modules****: A member noted a [deprecation PR](https://github.com/pytorch/ao/pull/3514) related to **dtype** in **linear_quant_modules** and raised concerns about failing tests.
   - The tests still reference **dtype** for the modules which were asked to deprecate **dtype**; the member asked if it was okay to edit the tests to address this issue.
- ****ao** Gets Namespaced**: A member pointed to a [repo-wide PR](https://github.com/pytorch/ao/pull/3515) referencing namespacing in the **ao** library.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1451080968025804944)** (14 messages🔥): 

> `AI Formal Verification, GPU Kernels Verification, PyTorch PRs, Open Source Contribution` 


- **AI and Formal Verification Merge**: One member shared a [link](https://martin.kleppmann.com/2025/12/08/ai-formal-verification.html) about the application of **AI + formal verification** in computer graphics and design.
   - The combination ensures the design meets a specification, potentially enhanced by **AI-assisted formal verification**.
- **Formal Verification Needed for GPU Kernels?**: A member inquired about research on **formal verification of GPU kernels**.
   - Another member shared a potentially relevant resource: [VeriNum](https://verinum.org/).
- **PyTorch PRs Awaiting Action**: A member reported limited activity on their **PyTorch PRs**, submitted two weeks prior.
   - The member noted that one PR had assigned reviewers but no activity, while the other was labeled as triaged.
- **Open Source Swimming Lessons**: A member, seeking to **get their feet wet with open source**, submitted deprecation PRs in PyTorch.
   - Another member suggested inquiring in the appropriate channel, but acknowledged the PRs might be low priority.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1450975699589660756)** (2 messages): 

> `Training Models on Strix Halo, PyTorch Tutorials, GitHub Repositories` 


- **Request for Training Resources on Strix Halo**: A member requested pointers to **GitHub repositories** or **tutorials** for training a model in **PyTorch** on a **Strix Halo**.
- **Looking for PyTorch Training Tutorials**: The member is seeking resources to help train models using **PyTorch**.
   - They are interested in tutorials that can guide them through the process.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1451199895410315285)** (9 messages🔥): 

> `SonicMoE, NVIDIA Hopper GPUs, Princeton University, UC Berkeley, Together AI` 


- **Sonic Boom: SonicMoE Races onto NVIDIA Hoppers**: A new Mixture of Experts (**MoE**) implementation called **SonicMoE** optimized for **NVIDIA Hopper GPUs** was introduced, claiming a **45% reduction** in activation memory and being **1.86x faster** on **H100** than previous state-of-the-art implementations, as detailed in [this blog post](https://d1hr2uv.github.io/api-agents.html) and [ArXiv paper](https://arxiv.org/abs/2512.14080).
   - The project is a collaboration between researchers from **Princeton University**, **UC Berkeley**, and **Together AI**, with code available on [GitHub](https://github.com/Dao-AILab/sonic-moe).
- **SonicMoE Talk Scheduled for March 7th**: Following the introduction of **SonicMoE**, there was a discussion about giving a talk on the server, and a date of **March 7th** was proposed.
   - A user agreed to the talk, but specified it wouldn't happen soon, suggesting February as a possible timeframe.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

kashimoo2_76983: <@1012256135761383465>  did you folks write a decode kernel with mi300s or 355s?
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1451298535990497392)** (4 messages): 

> `Reasoning-gym code, faker generator, robust tests` 


- **Reasoning-Gym Revamp**: A member proposed refactoring the reasoning-gym code to standardize answer generation and include a flag to display step traces, like `question(verbose=False, **kwargs) -> Any`.
   - They suggested [integrating Faker](https://github.com/joke2k/faker) to generate more robust tests with fake names.
- **Reasoning-Gym & Faker hookup?**: The community discussed hooking up reasoning gym to the [faker generator](https://github.com/joke2k/faker).
   - The community posited that it would generate more robust tests with fake names, instead of the same default names.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1450942222462746727)** (13 messages🔥): 

> `nvfp4_gemm benchmark, grayscale_v2 benchmark, H100 performance, NVIDIA performance` 


- **NVIDIA Runs on nvfp4_gemm**: A member's submission to the `nvfp4_gemm` leaderboard was successful on **NVIDIA** at **13.4 µs**.
- **H100 runs on grayscale_v2**: A member's submission to the `grayscale_v2` leaderboard got **6th place on H100** at **1373 µs**.
- **More H100 runs grayscale_v2**: Multiple successful submissions to the `grayscale_v2` leaderboard on **H100** all recorded a time of **1374 µs**.
- **NVIDIA gets Personal Bests on nvfp4_gemm**: A member achieved a personal best on **NVIDIA** for the `nvfp4_gemm` leaderboard at **37.5 µs**.
- **NVIDIA continues with Successes on nvfp4_gemm**: A member's submission to the `nvfp4_gemm` leaderboard was successful on **NVIDIA** at **10.9 µs**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1451099487778115604)** (3 messages): 

> `Homelab Setup, GPU Training Differences, NVIDIA vs Other GPUs/NPUs, Intra-Node Interconnect Importance, NVIDIA's Software Role` 


- **Homelab Hardware Hookups**: A member is setting up a multi-node **homelab** with **GPU**, **Kubernetes**, and **InfiniBand** and asked for beginner recommendations.
   - They are looking for resources on building a mini multi-node machine.
- **NVIDIA's Hardware Hacking Highlighting**: A student asked why **NVIDIA GPUs** and **Google TPUs** are better for training large AI models compared to other GPUs/NPUs.
   - Other GPUs (like **Apple**, **AMD**, **Intel**) are considered more suitable for inference.
- **Intra-Node Interconnect Impacts**: A member suggested that **intra-node interconnect** is a big factor in NVIDIA's performance.
   - They noted that Google has *insanely big interconnect domains*, while NVIDIA used 8 until NVL72.
- **NVIDIA's Software Stack**: A member stated that **software** plays a significant role in NVIDIA's dominance.
   - They added that *nvidia has so much software being used everywhere*.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1451347020915933309)** (3 messages): 

> `AMD-MLA-Decode leaderboard, Reproducing Kernels, MI300 Availability, AMD Developer Cloud` 


- **Kernel Reproduction Quest on AMD MLA Decode**: A member is seeking guidance on reproducing kernel results from the [AMD-MLA-Decode leaderboard](https://www.gpumode.com/v2/leaderboard/463?tab=rankings).
   - Specifically, they aim to replicate the exact environment used in the competition.
- **MI300 Eludes Modal Setup**: The user notes that the **MI300** is not available on Modal for reproducing the results.
   - They inquire about alternative setups to match the competition environment.
- **AMD Cloud Credits Offer Hope**: A member suggests utilizing the **AMD Developer Cloud**, which provides free credits.
   - This could potentially offer the necessary resources for setting up the desired environment.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1451046782560632972)** (2 messages): 

> `Trimul competition, Kernel Runtime, Geometric Mean, Standard Deviation` 


- **Trimul Kernel Runtimes Aim for Reproducibility**: A member is attempting to reproduce kernel runtimes from the Trimul competition leaderboard, referencing the [Trimul competition leaderboard](https://www.gpumode.com/v2/leaderboard/496?tab=rankings).
   - They provided a [diff on GitHub](https://github.com/gpu-mode/discord-cluster-manager/compare/main...LeoXinhaoLee:discord-cluster-manager-fork:xinhao/1217-reproduce-trimul) showing the changes made to the `discord-cluster-manager` repository, and are looking for advice on reproducing results and reducing variance.
- **Geometric Mean close enough?**: The user seeks feedback on whether the reproduced Geometric mean is close enough to the original results.
   - They are also curious if the variance in Test 1-5 (across different Modal instances with the same GPU type) is acceptable and expected.
- **Standard Deviation Summary Provided**: A summary of standard deviations is available in [this Google Sheet](https://docs.google.com/spreadsheets/d/1-lNEeNkf71YEabeX72jzZVhvUMBk2jcK2iPpuCTEDRg/edit?gid=114888003#gid=114888003).
   - The user asks for opinions on whether the observed variance across multiple Modal instances is acceptable and expected.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1450940682058338519)** (113 messages🔥🔥): 

> `CuTeDSL L2 cache hint policies, Submission system timing out, Discord Bot usage for Submissions, MMA wrap optimization, TCGen05 instruction assistance` 


- **CuTeDSL enables L2 Cache Specification**: A member inquired about specifying **L2 cache hint policies** in **CuTeDSL**, similar to **CUTLASS**, and another member confirmed this capability, pointing to [an example in the CuTeDSL repository](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/atom.py#L199).
- **Submission System Plagued by Timeouts**: Several members reported issues with the submission system timing out, particularly when using the **popcorn-cli**, with one member reporting a *Stream ended unexpectedly without a final result or error event*.
   - The issue was attributed to *high traffic* causing the system to hit process limits, but adding more compute to the heroku instance should solve the problem.
- **Discord Bot makes a comeback**: A member inquired about submitting via the **Discord bot**, and another member explained the process using the `/leaderboard submit` command in the designated channel, and that it prompts for file inputs.
   - Submissions via discord bot seemed to work fine when the CLI submissions were failing.
- **MMA Warp Optimization Mishaps**: A member sought advice on issuing **MMA** during warp optimization, experimenting with settings like `land_id==0` and `warp_id == 0 && lane_id == 0`, facing mismatches despite adjustments.
   - Another member suggested waiting for **MMA** completion before reading results from **TMEM**, recommending the DeepGEMM repository as a valuable reference and example implementation [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh).
- **TCGen05 Instruction Assistance**: A user received advice on resolving mismatches when issuing **MMA** with **TCGen05**, after sharing code snippets in DMs.
   - The consensus was to avoid needing `tcgen05.wait::st.sync` if not performing any **rmem->tmem** operations, as `tcgen05.cp` -> `tcgen05.mma` are correctly ordered.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1450973481180201042)** (6 messages): 

> `Hand Pose Estimation, Wrist Cameras, NVIDIA Cosmos Predict, Mimic-Video Paper` 


- **Neurospike shares hand pose-estimation experiment**: A member shared **three videos** of their hand pose-estimation experiment, available on [X.com](https://x.com/neurosp1ke/status/2001470564435636489).
- **Quest for Wrist Cameras**: A member inquired about the specific **wrist cameras** used by **PI** and others for capturing hand movements, struggling to find information through Google searches.
   - The suggestion was made that having individuals continuously stream household activities via hand-mounted cameras could create a rich dataset on **Hugging Face**.
- **Cosmos Predict as Action Model**: A member plans to examine **NVIDIA's Cosmos Predict**, considering its potential as an **action model**.
- **Mimic-Video Efficiency Claimed**: Referencing the [Mimic-Video paper](https://www.arxiv.org/abs/2512.15692), a member noted its claim of being more efficient than starting from a **VLM**.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1450940782855983245)** (19 messages🔥): 

> `Contributing to Open Source Projects, Keeping up with SoTA Research, AI Infra Engineer Demand, Kernel Competitions, Parallel Programming Passion` 


- **Community Contribution Catapults Career**: Being involved in a community is the *best start* for a career, especially open source projects with **SoTA work** that boost marketability.
   - One member recounted their experience with **LigerKernel**, which helped them meet industry people and possibly get a job.
- **AI Infra Engineer Openings**: Demand for **AI infra engineers** has risen, leading some employers to cast a *wider net* for candidates.
   - One member mentioned that **Eli Lilly** in Indianapolis is hiring a lot of **HPC/GPU/Linux infra engineers** due to AI investments from **Mounjaro/Zepbound** windfall (aka Ozempic-like), with good pay and cheap living costs.
- **Competitions Keep Coders Keen**: Members discussed kernel competitions, including the **AMD** one and the current **NVIDIA NVFP4** one, for staying sharp.
   - Information on competitions can be found in the **#leaderboards** channel.
- **Parallel Programming Passion Pays Off**: One member recommends to *live and breathe parallel programming* to excel at kernel development.
   - They suggest rethinking every coding project to parallelize the code, making kernel writing *second nature*.
- **Hardware System Hues Hurtful?**: Despite background in hardware systems, signal processing, and numerical analysis, one member found transitioning to **ML systems** challenging in terms of finding opportunities.
   - They ported a custom audio-vision language model to **vLLM**, applied quantization, and presented work at NYC meetups, but found more interest in flashy agent demos than model architecture and performance graphs, so they'll *try showcasing work on social media next*.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1451312376707027205)** (3 messages): 

> `Pinned Chats, GPT-5.2-Codex, Chain-of-Thought Monitorability` 


- **Pinned Chats are Finally Here!**: **Pinned Chats** are now rolling out to **iOS**, **Android**, and **web** for easier access to important conversations.
   - To pin a chat, tap the "..." next to it on the web or long press on mobile.
- **GPT-5.2-Codex Sets New Coding Standard**: **GPT-5.2-Codex** is now available in Codex, setting a new standard for agentic coding in real-world software development and defensive cybersecurity, as highlighted in [OpenAI's announcement](https://openai.com/index/introducing-gpt-5-2-codex/).
- **Chain-of-Thought Monitorability Framework Built**: A new framework and evaluation suite has been built to measure **Chain-of-Thought (CoT) monitorability**, as detailed in [OpenAI's blog post](https://openai.com/index/evaluating-chain-of-thought-monitorability/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1450940633492623360)** (163 messages🔥🔥): 

> `GPT-5.2 Hallucinations, Gemini 3.0 Flash, Sora 2 Discussions, AI Coherence over Time, ChatGPT App Store` 


- **GPT-5.2 Suffers Hallucinations and Identity Crisis**: Members reported that **GPT-5.2** sometimes *forgets* its capabilities in certain chats, requiring users to *lecture it into remembering* even when explicitly instructed to use Python and FFmpeg, suggesting the model suffers from **hallucinations and bad initial routing**.
   - It was noted that switching to **'Thinking' mode** sometimes helps, as the model initially denies having access to Python before suddenly *discovering* its capabilities.
- **Gemini 3.0 Flash is Coming Soon**: A member announced that **Gemini-Flash-3-Image** is coming soon, though its naming scheme was criticized as *horrible*, with another member clarifying that this is an upgrade to the current **Flash version from 2.5 to 3**.
   - It was further explained that **3-Pro** offers **100 images a day** for paid users, while **2.5-Flash** provides **1000**, leading to the assumption that the new version will offer **1000 images** with improved quality.
- **Members Seek Sora 2 Discussions**: A member inquired about **Sora 2** discussions, noting that there used to be a dedicated channel with thousands of messages, to which another member linked the general [video-generation channel](https://discord.com/channels/974519864045756446/1315696181451559022).
   - Concerns were raised about the absence of a specific channel for **Sora**, similar to *ai-discussions*, for focused conversations.
- **AI Coherence Explored**: A member shared their exploration of **AI coherence over time**, focusing on minimal systems where observation and internal models continuously converge without collapsing or contradicting each other, emphasizing **self-consistency under feedback** rather than correctness.
   - They are prototyping a tiny kernel-like framework focused on **long-horizon coherence, model drift, and agent self-correction**, seeking similar ideas from others.
- **ChatGPT App Store Goes Live**: It was announced that developers can now submit apps to **ChatGPT**, as the **app store is live** and accessible via the Apps menu in Settings, prompting discussion about potential **safety nightmares**.
   - A member added that the app store is live inside **ChatGPT** on web and mobile.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1451012401989288008)** (8 messages🔥): 

> `Gemini 3.0 Flash, GPT 5.2 High, deepseek r3.2, API date` 


- **Gemini 3.0 Flash beats GPT 5.2 High**: A member reported that **Gemini 3.0 Flash** is significantly beating **GPT 5.2 High** on their physics and ML tests suites, achieving a **90%** success rate while GPT 5.2 High failed a series of tests.
   - The user offered to share their benchmarks via DM and stated that *"High' reasoning being beaten by the Flash model is not what I expected from OpenAI"*.
- **Deepseek r3.2 cheaper model**: A member pointed out that **deepseek r3.2** are the cheapest.
   - There was no further discussion to elucidate the statement.
- **GPT 5.2 knows date?**: A member questioned how **GPT 5.2** knows the current date, as this should not be attached by the **API**.
   - There was no further discussion to elucidate the statement.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1451112675999219724)** (1 messages): 

> `Model provenance, AI writing style, Lack of provenance annotation tags` 


- **AI writing style is too recognizable**: A member says they can detect when people use a model to polish their responses without provenance annotation tags.
   - They describe the AI writing style as *the same frat boy with a law degree in nonsense*.
- **Model provenance crucial**: A member emphasizes the importance of provenance annotation tags when using AI models.
   - This ensures transparency and avoids misrepresentation of content origin.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1451112675999219724)** (1 messages): 

> `Model Detection, Provenance Annotation, Response Polishing` 


- **Models Sound Like Frat Boys with Law Degrees**: A member believes they can detect when people use models to polish responses without **provenance annotation tags** due to the model's consistent, recognizable style.
   - The member describes the model's style as sounding like *"the same frat boy with a law degree in nonsense"*.
- **Provenance Annotation Needed to Detect AI**: Discussion centers around the need for **provenance annotation tags** to identify AI-polished responses.
   - Without such tags, identifying the use of AI relies on detecting stylistic patterns, which can be subjective.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1451016087557767263)** (103 messages🔥🔥): 

> `Lightweight Vision Transformer Models, Model Choice for Structured Data Extraction, Fill Mask Techniques, Forward Pass in LLMs and Steering, Kaggle Runtime Disconnections` 


- **Vision Transformer Training Time Crunch**: A member sought lightweight **Vision Transformer** models trainable within **Kaggle's 9-12 hour session limit** on the **ImageNet dataset**.
   - Others noted that training **ImageNet** within that timeframe is challenging without significant GPU resources.
- **Sifting Smoller Models for Structured Data Extraction**: A member asked about model choice for **structured data extraction**, mentioning using **phi 3.5 mini instruct with instruct disabled**.
   - Another member suggested that a **1.5B Instruct LLM** (like **Qwen 2.5 1.5B Instruct using Q4_K_M**) should be sufficient and **preprocessing and postprocessing are necessary**.
- **Unlocking LLMs Internals through Model Steering**: A member discussed a technique to steer LLMs, eliminating the need for retraining and data, inspired by [Mentat's YC Launch](https://www.mentat.ai/).
   - They point out the hook function won't remain tiny as in example, as it will be the place where innovation will happen, finding the right amount of vectors and temperatures.
- **HuggingFace Hub Channel Compression Creates Calm**: A member noticed a decrease in channels and another explained that many channels were compressed to reduce noise.
   - The compression focused on unmoderated, low-activity, or duplicate channels, such as <#1019883044724822016>.
- **Tackling RAG System's Hidden Constraint**: An engineer working on **RAG systems** shared an article discussing the hidden constraints and the need to move beyond unengineered knowledge.
   - The article is on [Medium](https://medium.com/@rradhakr/the-rag-systems-hidden-constraint-why-we-must-move-beyond-unengineered-knowledge-part-1-3df60c65ee68) and is seeking community feedback.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1450994971988725872)** (2 messages): 

> `Android voice assistant, Gemini 1.5 Flash, VoxCPM 1.5, Apple Neural Engine` 


- **Android Voice Assistant Powered by Gemini Flash Debuts!**: A member announced the creation of an [Android voice assistant](https://www.strawberry.li/) built with **Gemini 1.5 Flash** and invited others to test it.
   - They also released a beta version that generates speech using **VoxCPM 1.5** and runs on the **Apple Neural Engine**.
- **Voice Assistant Leverages VoxCPM and Apple Neural Engine**: The beta version of the [Android voice assistant](https://www.strawberry.li/) now generates speech using **VoxCPM 1.5**.
   - This version is optimized to run on the **Apple Neural Engine**, promising enhanced performance on Apple devices.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1451208853818445864)** (4 messages): 

> `Smolcourse Delays, AI Learning Resources` 


- **Smolcourse Status Stalls, Subscription Concerns Stir**: Members are seeking updates on the **smolcourse**, noting a month-long delay since the last update and expressing concerns about having paid for **two months of Pro subscription** with limited course use.
   - The delay was initially attributed to improvements being made to a library used by Hugging Face, with a tentative expectation for the course to resume in December.
- **YouTube Channels Touted for Top-Tier Tutorials**: Members suggested checking out content from **Andrej Karpathy**, **Deeplearning.AI**, and **Anthropic AI** on YouTube for AI concepts.
   - They also recommended **Mervin Praison** and **Umar Jamil** for AI code-alongs.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1450983924368015462)** (89 messages🔥🔥): 

> `Exa AI People Search, Michael Truell and John Schulman LLM discussion, OpenAI potential $750B valuation, Pieter Abbeel Amazon AGI Head, Tomo AI` 


- **Exa AI Labs Unveils People Search**: Exa AI Labs launched a new **People Search** feature allowing users to semantically search over **1 billion** individuals using **hybrid retrieval** powered by finetuned Exa embeddings; find the launch tweet [here](https://x.com/exaailabs/status/2001373897154007390?s=46).
- **OpenAI Mulls Massive Financing Round**: OpenAI is in early discussions with investors for a new financing round, projecting a valuation around **$750 billion**, and hopes to raise tens of billions, possibly up to **$100 billion** according to [Katie Roof's tweet](https://x.com/katie_roof/status/2001453561910358474?s=46).
- **Pieter Abbeel Joins Amazon as AGI Head**: Prof. **Pieter Abbeel** of UC Berkeley, known for his work in Robotics and DeepRL, has been appointed as Amazon's new **AGI Head**, according to [this tweet](https://x.com/bookwormengr/status/2001353147055509881?s=46).
- **FunctionGemma: Calling Functions Locally**: Google introduced **FunctionGemma**, a new **270 million** parameter model optimized for function calling and designed to run directly on devices like phones and browsers; more info can be found [here](https://x.com/osanseviero/status/2001704036349669757).
- **T5Gemma 2: Multimodal, Multilingual Model Released**: **T5Gemma 2**, a new generation of encoder-decoder models built on Gemma 3, is available in sizes up to **4B-4B** and supports **140 languages**, according to [this tweet](https://x.com/osanseviero/status/2001723652635541566?s=46).


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1451148723856867361)** (2 messages): 

> `vLLM Router, Intelligence Control Plane, Ollama / vLLM routing through semantic router` 


- **vLLM Router Balances Load with Rust**: The **vLLM Router**, purpose-built for **vLLM fleets** and written in **Rust**, supports consistent hashing for **KV locality**, power-of-two choices, retries/backoff, circuit breakers, **k8s discovery**, and **Prometheus metrics**.
   - It's designed for **P/D disaggregation** with distinct worker pools and routing policies to preserve throughput and tail latency.
- **Intelligence Control Plane Manages Safety & Memory**: **vLLM + AMD** preview a **Semantic Router** framing—governing inputs, outputs, and long‑term state with an emphasis on safety/memory in large agent systems.
   - This constitutes a step **Toward an “Intelligence Control Plane”**.
- **Semantic Router Christmas Wishlist**: A member expressed their desire for **ollama /vllm routing** through semantic router (**BERT**) for both **DLP / model-selection token-optimization**.
   - They hope someone in the channel is building this and anticipate having time over the holidays to work on it.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1450984775526645881)** (8 messages🔥): 

> `Black Forest Labs FLUX.2 Launch, xAI Grok Voice Agent API` 


- **Black Forest Labs ships FLUX.2 [max] AI Model**: Black Forest Labs announced the launch of **FLUX.2 [max]**, their highest quality AI model to date, as announced on [X](https://xcancel.com/bfl_ml/status/2000945755125899427?s=46).
- **xAI Launches Grok Voice Agent API**: xAI announced the launch of the **Grok Voice Agent API**, which enables developers to create voice agents capable of speaking multiple languages, utilizing tools, and searching real-time data, announced on [X](https://xcancel.com/xai/status/2001385958147752255).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1450962816478609468)** (14 messages🔥): 

> `GPT-2 interpretability, 3D visualization of residual stream, SOTA Model Performance, Claude Opus 4.5 mistakes, Neuronpedia` 


- **3D Viz Tool Aims to Deep Dive into GPT-2**: A member is building a 3D holographic interactive app to *look inside every node of GPT2 small 124m LLM* and is looking for feedback on their project.
   - Another member shared a link to their own project, a [3D visualization of the residual stream](https://aselitto.github.io/ResidStream/), seeking feedback on whether the visualization matches the user's mental model.
- **Neuronpedia Interest Sparked by GPT-2 Viz**: A member suggested that the GPT-2 visualization project is related to **Neuronpedia** and suggested reposting it in the mech interp discord.
   - They shared a link to the [mech interp discord](https://discord.com/channels/729741769192767510/732688974337933322/1425176182965665852) for further discussion.
- **SOTA Models Performance Nearly Indistinguishable**: Members are finding that SOTA models are almost indistinguishable in terms of performance and intelligence.
   - One member noted that **Claude Opus 4.5**, despite acknowledging mistakes, repeated them, performing only slightly better than **GPT 3.5**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1451101109056962561)** (3 messages): 

> `Speech/NLP Research Collaboration, AI Research, NLP` 


- **Members Seek Collab on Speech/NLP**: A member expressed interest in collaborating on research in **speech/NLP**, citing interest after reading the abstract of [this paper](https://arxiv.org/abs/2512.15687).
   - The member asked other members to reach out if interested.
- **AI Researchers looking for collaboration**: A member seeks collaboration on research on any topic in speech/NLP.
   - The user specified the request to any member interested.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1450970088575340678)** (4 messages): 

> `Anthropic's weight masking, Gemma 3 extreme activations, Adam's fault` 


- **Anthropic Masks Away Harms**: Anthropic released a paper on [Selective Gradient Masking (SGTM)](https://alignment.anthropic.com/2025/selective-gradient-masking/) focusing on **unlearning dangerous knowledge** via isolating/deleting specific weights, incurring a **6% compute penalty** on general knowledge.
   - A member's patching experiment showed only a **93% recovery**, suggesting the *superweight is the electricity/ML equivalent of the path of least resistance* as a model can build a distributed circuit bypass without completely tanking.
- **Gemma 3's Weights Under Scrutiny**: Discussion arose around **Gemma 3** potentially having extreme activations or weights to fit more information in the model.
   - It remains to be seen whether these are training artifacts or intentional design choices to improve information density.
- **Adam Gets the Blame**: A member jokingly blamed **Adam** (likely referring to the Adam optimizer) for the discussed phenomena.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1451246848600707216)** (4 messages): 

> `long range view synthesis, novel view synthesis, parallax effect, depth estimate` 


- **Interest in Long Range View Synthesis Results**: A member expressed interest in results for **long range view synthesis**, especially where the desired camera angle is far from the input's, without needing to be monocular, and should exploit object symmetries.
   - They clarified that "far" refers to a large number of degrees of rotation away from a different angle, rather than physical distance.
- **Novel View Synthesis at Distant vs Proximity**: Another member posited that synthesizing **novel views at a distance** might be easier than at proximity, as the **parallax effect** would be less.
   - The original member suggested a sweet spot for multi-view inputs, balancing **parallax** in **depth estimation** with enough landmarks for registering different object points across views.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1451362312630829189)** (1 messages): 

> `Custom Cross Entropy Function, Backwards Pass` 


- **Bypass Backprop: Cross Entropy Hack!**: A member inquired about replacing the cross-entropy function in the repository with a custom one, without rewriting the backwards pass.
   - They suggested that it might be possible by writing a new class that inherits from the old cross-entropy function, and then [overriding only the forward pass](https://pytorch.org/docs/stable/notes/extending.html).
- **Backpropagation Made Easy**: A member asked if it's possible to substitute the cross-entropy function listed in the repository with a personalized version, avoiding a complete rewrite of the backwards pass.
   - Another member pointed out that [PyTorch's documentation](https://pytorch.org/docs/stable/notes/extending.html) offers guidance on extending existing classes, potentially allowing for overriding only the forward pass, thus simplifying the process.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1450995399803404308)** (19 messages🔥): 

> `Mojo GPU usage, Rust GPU capabilities, Mojo std::offload` 


- **Mojo launches code explicitly to GPU**: A member asked if **Mojo** could simply use an attribute to run a function on the **GPU**, a member clarified that explicit launching on the **GPU** is required.
   - They added, *so long as you aren’t doing syscalls, no attribute is required (although it would need to be launched as single lane)*.
- **Discussing Rust's single-function SIMD/GPU capabilities**: A member referenced a recent announcement of **Rust's** ability to use a single function and `std::batching` to get a **SIMD/fused** version, `std::autodiff` to differentiate it, and `std::offload` to run the resulting code on their **GPUs**, asking if **Mojo** already supported this.
   - They provided a [link to a Reddit post](https://www.reddit.com/r/rust/comments/1pp3g78/project_goals_update_november_2025_rust_blog/) related to the announcement.
- **Mojo Provides std::offload, Autodiff Later**: A member explained that **Mojo** has a `std::offload` equivalent, and you don’t need to use batching to fuse ops.
   - They clarified that *autodiff is something that has been discussed but left for later*, adding that building something that does **autodiff** right now would be difficult.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1451004910920274098)** (6 messages): 

> `GPU issues with LLM build in MAX, C interop ideas from Rust, Mojo's array access quirks` 


- **Modular Investigates LLM Build GPU Glitches**: A user reported persistent issues building an **LLM in MAX** even with the **GPU disabled** and shared [details on the Modular forum](https://forum.modular.com/t/build-an-llm-in-max-from-scratch/2470/9).
   - A Modular team member confirmed they are investigating, suspecting either an **API regression** or a **device-specific issue**.
- **Rust's C Interop Inspires Mojo Ideas**: A member shared a [Rust Zulip discussion](https://rust-lang.zulipchat.com/#narrow/channel/131828-t-compiler/topic/pre-MCP.20vibe.20check.3A.20-Zinternalize-bitcode/near/546294314) about **C interop**, suggesting similar approaches might benefit Mojo.
   - The idea is to explore how Mojo can potentially leverage Rust's strategies for improved interoperability with C code.
- **Mojo array access allows out-of-bounds indexing**: A member questioned why Mojo allows out-of-bounds indexing such as `x[5]` when `x = 2.0`.
   - No explanation was given.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1451011195434696905)** (17 messages🔥): 

> `Aider vs Aider-ce, OpenCode Accuracy vs Aider, Context Management and Token Efficiency, Task Bundling with Context` 


- **Aider sticks with Pair Programming**: Aider is designed as a **pair programmer** with a **human-in-the-middle** approach, in contrast to the agentic route taken by the **aider-ce** fork.
   - A member expressed strong reservations about achieving comparable accuracy and token efficiency with **OpenCode** or other agentic systems.
- **OpenCode vs Aider**: A user compared **OpenCode** side-by-side with **Aider**, citing concerns about **accuracy**, which they attributed to **Aider's** non-agentic design and its control over context.
   - They argued that the absence of spiraling effects from misunderstandings and efficient context management (via **/add** or **/load**) lead to better model performance, suggesting that exceeding **20-30k tokens** negatively impacts even **Opus 4.5**.
- **Context management and token efficiency are Key**: The user suggested that *using minimal context* with **/add** or **/load** is like having next generation's model alongside someone who lets the **CLI** or **IDE** control their context, adding fewer tokens in your context window leads to better model performance.
   - They linked to a [Chroma research piece on context-rot](https://research.trychroma.com/context-rot), noting the irony of paying more for worse performance due to inefficient context management.
- **Tasks bundled with Necessary Context**: A member asked about breaking everything down into **tasks** and then bundling the necessary context with the task.
   - The member added they routinely check back to aider because they like it but they are unsure about how to make use of everything.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1451263447281963173)** (3 messages): 

> `Gemini 3 Flash, aider configurations, Litellm updates` 


- **Gemini 3 Flash Triumphs!**: Many are praising **Gemini 3 Flash** as the best coding model, recommending its use while linking to [blog.brokk.ai](https://blog.brokk.ai/why-gemini-3-flash-is-the-model-openai-is-afraid-of/).
   - The commenter emphasized that the model excels, but suggested that users *disable thinking* to prevent speed slowdowns especially in **aider**.
- **Thinking Configuration Confusion**: A member admitted uncertainty about their current *thinking* configuration.
   - They noted that **Litellm** defaults to *low* for pro models, but **Gemini 3 Flash** isn't yet in their **Litellm** version (**aiderx**).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1450956459197206621)** (6 messages): 

> `Office Hours Recording, Sam's IOU Scheme, CUDA Performance on Linux` 


- **Office Hours Recording Requested**: A member inquired whether it's possible to upload the recording from the recent office hours.
- **Sam's IOU Scheme exposed?**: The market is catching on to **Sam's IOU scheme** as shown in [this YouTube video](https://www.youtube.com/watch?v=5DZ7BJipMeU).
- **Optimize CUDA Performance**: A member shared 2 tips for optimizing **CUDA** performance on Linux.
- **Multi-GPU VRAM Downclocking Bug**: When using `CUDA_DISABLE_PERF_BOOST=1` with **llama.cpp** in multi-GPU or partial offloading configurations, VRAM may get downclocked due to low GPU usage.
- **Disable P2 State to Boost Token Generation**: Per an Nvidia employee's recommendation, disabling **P2 state** for CUDA apps (config `0x166c5e=0`) can boost token generation by a few percent, as seen in [this GitHub comment](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/333#issuecomment-3669477571).


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1451170957841530880)** (3 messages): 

> `Open Source Server for Minos-v1, VLLM, SGLang` 


- **Minos-v1 seeks open source deployment server**: A member inquired about open source server implementations for deploying the [NousResearch/Minos-v1](https://huggingface.co/NousResearch/Minos-v1) model.
- **VLLM and SGLang rise to the occasion**: Another member suggested **VLLM** or **SGLang** for supporting classifiers serving.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1451229645977551078)** (3 messages): 

> `LLM finetuning, Electronic schematics dataset` 


- **LLM Finetuning Sparks Interest**: Members are excited about the prospect of quick **finetuning LLMs** to behave in specific ways.
- **New Electronic Schematics Dataset Appears**: A member shared a new dataset for training LLMs to create **electronic schematics**; see the [dataset here](https://huggingface.co/datasets/bshada/open-schematics/discussions).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1451229645977551078)** (3 messages): 

> `LLM finetuning, Electronic schematics dataset` 


- **Quick LLM Finetuning Looks Promising**: A member noted that quick **finetuning of LLMs** to act in a specific way seems amazing.
   - They personally haven't had the chance to try it yet.
- **Dataset for Training LLMs in Electronic Schematics Shared**: A member shared an *amazing dataset* for training **LLMs to create electronic schematics**.
   - The [dataset](https://huggingface.co/datasets/bshada/open-schematics/discussions) is available on Hugging Face.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1451005867339157506)** (12 messages🔥): 

> `GEPA Optimization, Robots building robots, TreeOfThought Module, dspy.Refine feedback, GEPA Definition` 


- **Genetic-Pareto Optimization Amazes**: A member exclaimed that running their first real **GEPA optimization** was like magic, demonstrating enthusiasm for the technology.
   - Another member jokingly added, robots that build robots may come, but for now they are using **DSPy**.
- **Genetic Prompt Evolution (GEPA) Defined**: **GEPA (Genetic-Pareto)** is a reflective optimizer that adaptively evolves textual components of arbitrary systems and uses both scalar scores and text feedback to guide the optimization process, as described in ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/abs/2407.19457).
   - In essence, it's an optimizer that genetically modifies AI prompts using another AI, selecting the best changes based on a metric method, effectively AI building AI.
- **GEPA Resources Shared**: A member shared resources including a [DSPy tutorial on GEPA](https://dspy.ai/tutorials/gepa_ai_program/), a [blog post on working with optimizers](https://thedataquarry.com/blog/learning-dspy-3-working-with-optimizers/#option-2-gepa), and a [Medium article on GEPA in action](https://medium.com/data-science-in-your-pocket/gepa-in-action-how-dspy-makes-gpt-4o-learn-faster-and-smarter-eb818088caf1) to help others learn about GEPA.
   - They noted that some syntax may have changed since the blog post was written.
- **Tree of Thought (ToT) Module Missing**: A member asked why there isn't a direct official **Tree of Thought (ToT) module** in **DSPy** yet.
   - No further discussion or responses were provided on this topic.
- **Custom Feedback in `dspy.Refine` Debated**: A member inquired about manually specifying the kind of feedback when using `dspy.Refine`.
   - They mentioned using a custom module for evaluator loops and wondered if they were missing something about `Refine`.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1450993229586628699)** (10 messages🔥): 

> `In-Context Learning Research, Draft Model Optimization, Training Cluster Pipelining, Vast.ai Issue, Nvidia's Brev vs Runpod` 


- **9000IQ Research on In-Context Learning Goes Viral**: A member shared a link to a [YouTube video](https://www.youtube.com/watch?v=q-yo6TPRPVk) showcasing *9000IQ research work on in-context learning* suggesting others *spend an afternoon with some popcorn*.
- **Draft Model Boosts Efficiency via Parallel Processing**: A member proposed using a **draft model** to guess the output of each part of a larger model split across multiple GPUs or servers, enabling parallel processing and improved system utilization.
   - The suggestion involves dropping responses that differ significantly from the draft, reverting to normal processing for those cases, aiming to batch runs through the system more efficiently.
- **Memory Bandwidth bottlenecks large Training Clusters**: In large training clusters, memory bandwidth within and across machines is the biggest bottleneck, requiring rapid data access to minimize stalling, while efficient pipelining can increase utilization.
   - Training is not always *pure* and involves forward and backward passes of multiple documents with combined gradients or weights, which can be equivalent to sequential processing in certain cases.
- **Vast.ai Template Glitch Troubles Users**: A member reported an issue with **Vast.ai**, where it launches with the wrong template, preventing the init script from running as expected.
   - A member suggested trying **Nvidia's Brev** as a potential solution but noted that **Runpod** is probably better.
- **Rsync to the Rescue**: One member bypasses template issues on Vast.ai by using their **PyTorch** template and using rsync everything they need over.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1450986276953264168)** (1 messages): 

> `ARC-AGI Benchmark, Toolathlon Benchmark, Training Data Mix` 


- **ARC-AGI Jump Attributed to Training Data**: The recent jump in performance on the **ARC-AGI benchmark** across models is likely due to training data including the benchmark itself, rather than a hidden breakthrough.
   - The user noted that better generalization at smaller sizes is often observed on these types of benchmarks.
- **Toolathlon Improvements Result from Training Emphasis**: The noticeable improvement in **Toolathlon benchmark** scores probably results from a training mix with more emphasis on this capability.
   - This adjustment likely ensures more reliable tool calling, even with fewer parameters.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1450968554814832732)** (6 messages): 

> `Kimi K2, Moonshot AI, Free Models` 


- **Kimi K2 Thinking article is Awesome!**: A member shared a [DigitalOcean article](https://www.digitalocean.com/community/tutorials/kimi-k2-moonshot-ai-agentic-open-weight-model) about **Kimi K2** and its thinking capabilities.
   - The member expressed excitement about the article.
- **Kimi K2 has a New Change!**: A member shared an image indicating a new change in **Kimi K2**.
   - The member expressed excitement, saying *"This is nice!!"*.
- **Free Kimi Models will reset after 1 month**: A member shared an image indicating that **free Kimi models** will reset after one month.
   - The member expressed uncertainty, stating *"So it appears they will now reset after 1 month - I guess? I mean the free ones?"*


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1450958376350515265)** (5 messages): 

> `DNS issues, Chat image limits, Manus revenue` 


- ****DNS Disaster** delays deploy!**: A user is experiencing a **DNS issue** with **Cloudflare** that has persisted for over 4 days, and is frustrated with the lack of customer service.
   - The user stated that *nobody responds from customer service except saying IM us directly*.
- ****Image Imbroglio** in Chat!**: A user questioned the chat mode image limit and the disparity between **Pro/paid** and **free** users.
   - They complained that **free users** already have *significantly lower intelligence capabilities* and expressed frustration that an image limit was additionally imposed, naming **DeepSeek** and **Gemini** as examples with no such limits.
- ****Manus Makes Millions**, Man!**: A user shared an article reporting that **Manus** has reached **US$100 million** in revenue.
   - The linked [SCMP article](https://www.scmp.com/tech/tech-trends/article/3336925/manus-hits-us100-million-revenue-milestone-global-competition-ai-agents-heats) notes that this milestone comes as *global competition in **AI agents** heats up*.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1451247243570188311)** (4 messages): 

> `MCP Prompts in Node Server, ChatGPT App Submissions` 


- **MCP Prompts: Node Server Activation**: A member asked about enabling **MCP prompts** in a **Node server**, noting that while registration with the orchestrator worked, the prompts tab in microservices was disabled.
   - Another member redirected the question to the **#mcp-support** channel or advised raising an issue for detailed assistance, clarifying that the current Discord is focused on contributor collaboration rather than tech support.
- **ChatGPT App Submission Requirements**: A new member inquired whether **ChatGPT apps** need a UI for submission or if just an **MCP server** is sufficient.
   - No response was given.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1451359133780676679)** (3 messages): 

> `JIT Refactor, Firmware Crushing, RDNA3 Assembly Backend` 


- **Claude fails JIT Refactor Test**: A member tried many times to get **Claude** to do the **JIT refactor** but had no luck, claiming *it really lacks taste* and planning to do it manually.
   - The refactor involves making the **schedulecache** complete and making the **JIT** run a few **schedulecaches**.
- **Tinygrad crushes Firmware stuff**: A member reported that it is crushing the firmware stuff, by using a whole emulator emulating a fake **USB device** on **Linux** that's passing everything to the firmware.
   - The emulator is passing everything to the **firmware**.
- **RDNA3 Assembly Backend Good Enough**: A member reported writing an **RDNA3 assembly backend** with a register allocator that's good enough to run gemms with **128 accs**.
   - The backend is available on [GitHub](https://github.com/tinygrad/tinygrad/pull/13715).


  

---


---


---

