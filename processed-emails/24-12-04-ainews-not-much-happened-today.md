---
id: 276161f6-797f-4b69-898f-ddb7654dba8e
title: not much happened today
date: '2024-12-05T02:41:39.435306Z'
original_slug: ainews-not-much-happened-today-1872
description: >-
  AI news for December 3-4, 2024, covering product launches, research releases,
  talent moves, model quality debates, and community discussions.
companies:
  - openai
  - google
  - deepmind
models:
  - gpt-4.5
  - claude-3-sonnet
  - genie-2
  - o1-full
topics:
  - product-launches
  - research
  - talent-moves
  - model-performance
  - multimodal
  - weather-forecasting
  - virtual-worlds
---


<!-- buttondown-editor-mode: plaintext -->**another quiet day is all we need.**

> AI News for 12/3/2024-12/4/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**198** channels, and **2915** messages) for you. Estimated reading time saved (at 200wpm): **317 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

*Smol.ai update: [Smol Talk now has vision!](https://www.loom.com/share/34b37822c6784989bafd6fcc5fee6420?sid=75bf3b4c-61b5-46fd-a2b1-7c7fe911df89) Where previously if it encounters an image, it would hallucinate, now we do the necessary prompting. See today's Reddit Recaps for an example, and now your personalized recaps also get them.*

**If you are interested in NeurIPS next week, there are 50 more tickets left for [our end of year recap event](https://lu.ma/LSLive) (livestream available, NeurIPS ticket not required). [Most speakers have been announced](https://x.com/swyx/status/1864423257266639166).**

[Genie 2](https://news.ycombinator.com/item?id=42317903) has topped HN all day, and we [previously covered SIMA](https://buttondown.com/ainews/archive/ainews-deepmind-sima-one-ai-9-games-600-tasks/), but given that this continues to be (impressive) cherrypickware, we aren't giving it title story status.

o1-full is expected [during their new advent calendar](https://x.com/OpenAI/status/1864328928267259941), just as they [poach a bunch of DeepMind researchers](https://x.com/iScienceLuvr/status/1864217903232385348). Perhaps it is true that [openai is so back](https://x.com/tszzl/status/1863882905422106851).

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

Here are the key themes and discussions from the Twitter data, organized by topic:

**OpenAI's "12 Days of Christmas" Launch Announcement**

- **Major product announcements**: [@sama](https://twitter.com/sama/status/1864335461268754712) and [@OpenAI](https://twitter.com/OpenAI/status/1864328928267259941) announced "12 Days of OpenAI" starting tomorrow, with daily livestreams featuring launches and demos. The community is speculating about potential releases like **O1 full model**, **Sora video model**, and **GPT-4.5**.
- **Launch logistics**: [@joannejang](https://twitter.com/joannejang/status/1864344210327130357) noted the challenge of shipping 12 consecutive announcements, suggesting backup plans like having executives juggle if needed.

**DeepMind's Major Research Releases**

- **GenCast Weather Model**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1864340994965098513) released an AI weather forecasting system in Nature that can make **15-day predictions in 8 minutes** using TPU chips, with state-of-the-art accuracy.
- **Genie 2 World Model**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1864367798132039836) launched a model that can create playable 3D worlds from single images, aimed at training future AI agents in virtual environments.

**High-Profile Talent Moves**

- **Vision Research Team to OpenAI**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1864217903232385348) reported that leading computer vision researchers **Lucas Beyer**, **Alexander Kolesnikov**, and **Xiaohua Zhai** moved from Google DeepMind to OpenAI. [@giffmana](https://twitter.com/giffmana/status/1864419226649546883) confirmed they'll be opening an office in Zürich.

**Criticism of AI Model Quality**

- **OpenAI Strategy Concerns**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1864367068314730778) criticized OpenAI's strategy of competing with customers while falling behind on model quality, suggesting they should focus on building great models like Anthropic.
- **Model Performance**: Multiple users noted that **Claude/Sonnet** outperforms other models despite being cheaper, with debate around the relative merits of different API pricing strategies.

**Memes & Humor**

- [@scaling01](https://twitter.com/scaling01/status/1864330169898684622) joked about wanting "computer use agents sora o1 GPT-5 fully multimodal 4o cheaper o1 models"

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Nemotron-51B Released: Nvidia's NAS Optimized Model Matches 70B Performance**

- **Modified llama.cpp to support Llama-3_1-Nemotron-51B** ([Score: 79, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1h6724m/modified_llamacpp_to_support_llama3_1nemotron51b/)): A developer successfully modified **llama.cpp** to support **Nvidia's Llama-3_1-Nemotron-51B** model, which performs similarly to the larger **70B** variant through **Neural Architecture Search (NAS)** optimization. The modified model is available on [HuggingFace](https://huggingface.co/ymcki/Llama-3_1-Nemotron-51B-Instruct-GGUF) with **Q3_K_S**, **Q4_0**, **Q4_0_4_8**, and **Q4_K_M** quantization options, with potential for integration into the main **llama.cpp** repository.
  - **Q3_K_S** quantization of the **51B model** shows better performance than **IQ2_XS** of the **70B model**, with users confirming improved results in practical testing. The **51B Q3_K_S** version requires **22.7GB** of VRAM.
  - Technical discussion reveals that **IQ4_XS** quantization for the **51B model** would require approximately **27.84GB** VRAM, exceeding **3090** GPU capacity, while the same quantization for the **70B** model needs **37.9GB**.
  - Performance degradation occurs with lower quantization levels without **imatrix**, as evidenced in the **Q2_K_S** implementation. The official performance claims can be found in [NVIDIA's blog post](https://developer.nvidia.com/blog/advancing-the-accuracy-efficiency-frontier-with-llama-3-1-nemotron-51b/).


**Theme 2. Dynamic 4-bit Quantization: Selective Layer Precision for Better Performance**

- **Quantizing to 4bits can break models - Dynamic quantization 10% FP16 90% 4bit** ([Score: 119, Comments: 50](https://reddit.com/r/LocalLLaMA/comments/1h6ojwr/quantizing_to_4bits_can_break_models_dynamic/)): **Unsloth** researchers discovered that quantizing all layers to **4-bit** precision can degrade model performance, demonstrating this with **Qwen2-VL-2B Instruct** where full 4-bit quantization produced incorrect image descriptions while using **10% FP16** and **90% 4-bit** precision maintained accuracy while reducing model size from **4.11GB** to **1.81GB**. Analysis of **Llama 3.2 11B Vision Instruct** revealed significant activation errors in **MLP layers** and weight quantization errors in **Cross Attention layers**, leading to the release of new dynamic quantization models on **HuggingFace** that achieve **2x faster** inference and use **50% less VRAM**.
  - **Unsloth** developers confirmed that **QwQ dynamic quantization** works for both vision and text models, with their first text-based model [QwQ-32B-Preview](https://huggingface.co/unsloth/QwQ-32B-Preview-unsloth-bnb-4bit) now available on HuggingFace. They noted that **vision encoders** generally shouldn't use **4-bit** quantization, particularly in **Llava-based models**.
  - Users expressed interest in implementing these hybrid quantization techniques, with discussions focusing on **GGUF quantization** similarities and requests for **OpenAI-compatible API servers** for local VLM deployment. The developers indicated plans to integrate this functionality into the broader **Unsloth** framework.
  - The research team shared additional analysis plots showing **activation spikes** in 4-bit quantization, with model configuration files indicating problematic layers. Community response was overwhelmingly positive, particularly regarding the detailed model debugging approach.


**Theme 3. FishSpeech v1.5: Multilingual Zero-Shot Voice Cloning Breakthrough**

- **FishSpeech v1.5 - multilingual, zero-shot instant voice cloning, low-latency Only 500M params - #2 ranked on TTS-Arena** ([Score: 91, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1h6p335/fishspeech_v15_multilingual_zeroshot_instant/)): **FishSpeech v1.5**, a multilingual voice cloning model trained on **1M hours** of data across **13 languages**, achieves **#2 rank** on **TTS-Arena** while maintaining **<150ms latency** with only **500M parameters**. The model is now open-source and accessible through multiple platforms including [fish.audio](http://fish.audio/), [GitHub](http://github.com/fishaudio/fish-speech), and [Hugging Face](http://huggingface.co/spaces/fishaudio/fish-speech-1), offering both self-hosting and cloud deployment options.
  - Users inquired about **voice cloning capabilities** and adding **emotional range** similar to **Bark**, highlighting key areas for potential future development in TTS technology.
  - The model comes with **non-commercial licensing restrictions** as specified on its [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5) repository.


**Theme 4. ByteDance Intern Drama: ¥8M Lawsuit Winner Gets NeurIPS Best Paper**

- **Former Intern Sabotages ByteDance’s AI Training, Faces ¥8 Million Lawsuit, Yet Wins NeurIPS 2024 Best Paper** ([Score: 79, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1h6i1m9/former_intern_sabotages_bytedances_ai_training/)): **Keyu Tian**, a former **ByteDance** intern, faces an **¥8 million** lawsuit for allegedly sabotaging the company's AI model training involving **over 8,000 GPUs** in **August 2024**, resulting in claimed losses of tens of millions of dollars. Despite the legal controversy, Tian went on to win the **NeurIPS 2024 Best Paper Award** for research conducted during his ByteDance internship, with his paper "[VAR](https://arxiv.org/abs/2404.02905)" developed in collaboration with the company's Commercialization Technology Department.
  - According to **ByteDance's official statement**, the intern maliciously interfered with model training in the **Commercialization Technology Team** only, not affecting other business operations. The company clarified that claims of "**8,000 GPUs**" and "**tens of millions**" in losses were **grossly exaggerated**.
  - **Keyu Tian** was dismissed in **August** and the matter was reported to both his university and industry alliance. The incident specifically impacted research projects within his team, with no involvement in **ByteDance's AI Lab** or large models.
  - Technical experts note that modern AI training includes extensive **logging**, **real-time analytics**, and **checkpoint testing**, making it unlikely that entire model training efforts were lost. The damages likely stem from opportunity costs of GPU cluster downtime.


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. OpenAI '12 Days of Shipmas' to Include Sora and O1 Model Releases**

- **[OpenAI’s 12 days of ‘shipmas’ include Sora and new reasoning model](https://www.theverge.com/2024/12/4/24312352/openai-sora-o1-reasoning-12-days-shipmas)** ([Score: 203, Comments: 60](https://reddit.com/r/OpenAI/comments/1h6ib0o/openais_12_days_of_shipmas_include_sora_and_new/)): **OpenAI** announced a **12-day product release schedule** that includes their new **Sora video generation model** and **O1 reasoning model**. No additional details were provided about specific release dates or technical capabilities of these models.
  - **Sam Altman's tweet** confirms daily livestreams with product launches and demos, but community members express skepticism about actual releases, noting OpenAI's history of announcing features as *"coming in weeks"* without immediate deployment.
  - Discussion around **compute resources** suggests the **O1 transition** from preview to stable won't significantly increase system load, while the community speculates about OpenAI's GPU capacity for handling multiple major releases like **Sora** simultaneously.
  - The announced **Santa Voice** feature for **Advanced Voice Mode** generated excitement for potential parent-child interactions, though some users jokingly referenced the standard AI model disclaimer *"I'm sorry, as a language model, I can't bring you toys"*.


- **[What's coming next? What's your guess?](https://i.redd.it/tplh9liduu4e1.jpeg)** ([Score: 392, Comments: 126](https://reddit.com/r/OpenAI/comments/1h6jjrt/whats_coming_next_whats_your_guess/)): **OpenAI** announced "**12 Days of OpenAI**," a series of **12 livestreams** starting tomorrow that will feature various announcements. The community speculates about the content of these announcements, which OpenAI describes as ranging from *"big and small"* developments.
  - Community expectations center around releases of **O1**, **Sora**, and **Operator**, with many users citing **Anthropic's MCP release** as pressure for OpenAI to deliver. The most upvoted comments express skepticism about timely access to announced features.
  - Users predict a mix of immediate releases and future promises, with specific interest in **GPT-4 Mini updates**, **cheaper real-time API pricing**, and **advanced voice mode** features. Several comments suggest these announcements may be timed to compete with **Google/Gemini**.
  - Technical speculation focuses on potential **agent models**, **unlimited memory** features, and **full browser control capabilities**. Most developers express desire for practical improvements like better API pricing over flashier announcements.


**Theme 2. New Open Source AI Video Models: Tencent Hunyuan vs LTX Comparison**

- **[Tencent's new open source AI text-to-video model Hunyuan can do bounce physics. It's over.](https://v.redd.it/mmjvx1xbjs4e1)** ([Score: 771, Comments: 120](https://reddit.com/r/ChatGPT/comments/1h6b9h8/tencents_new_open_source_ai_texttovideo_model/)): **Tencent** released their **Hunyuan** text-to-video model on **HuggingFace**, accessible at [Tencent-Hunyuan-Large](https://huggingface.co/tencent/Tencent-Hunyuan-Large). Without access to the referenced video content, no specific claims about physics capabilities or model performance can be verified.
  - Users noted the model's impressive **physics simulation capabilities**, particularly for **hair movement** and other dynamic elements, with comparisons drawn to games like **GTA VI** and **Stellar Blade**.
  - The community discussed **open-source motivations** behind Chinese companies releasing models, with Tencent's official statement citing goals to *"inspire more researchers with innovative ideas and collectively advance the progress of AI technology"*. The correct model link was shared at [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo).
  - Multiple comments expressed concerns about **AI-generated content** potentially disrupting various industries, with predictions that a significant portion of certain online content will be AI-generated within years.


- **[LTX Video vs. HunyuanVideo on 20x prompts](https://v.redd.it/y6comqv9lw4e1)** ([Score: 60, Comments: 57](https://reddit.com/r/StableDiffusion/comments/1h6sdsp/ltx_video_vs_hunyuanvideo_on_20x_prompts/)): Unable to provide a meaningful summary as the post body is empty and video content cannot be analyzed. A proper summary would require the actual content, discussion points, or comparative analysis mentioned in the title about **LTX** and **HunyuanVideo** models.
  - **Hunyuan** requires significant computational resources, needing a minimum of **60GB GPU memory** for **720x1280** resolution and taking **2 hours** per 6-second video generation. Users note that performance varies between **15 minutes** on **544x960** resolution when fitting in VRAM versus **2 hours** when overflowing to RAM.
  - The comparison methodology is questioned due to **LTX** benefiting from **100+ step counts** versus the apparent **10 steps** used in the test. Critics point out that **LTX** requires detailed prompts and is still in **version 0.9** training.
  - A full comparison is available at [checkbin.dev](https://app.checkbin.dev/snapshots/70ddac47-4a0d-42f2-ac1a-2a4fe572c346), with users noting that while **Hunyuan** shows promise for open-source video models, future **quantized versions** may improve accessibility beyond current **A100** GPU requirements.


**Theme 3. OpenAI Reaches 300M Weekly Users, Signs Defense Contract**

- **[ChatGPT now has over 300 million weekly users](https://www.theverge.com/2024/12/4/24313097/chatgpt-300-million-weekly-users)** ([Score: 200, Comments: 19](https://reddit.com/r/OpenAI/comments/1h6m4so/chatgpt_now_has_over_300_million_weekly_users/)): **ChatGPT** has achieved **300 million weekly active users**, marking a significant user base milestone for the **OpenAI** chatbot.
  - **300M weekly users** demonstrates significant mainstream adoption, with users comparing **ChatGPT** to **Google's** search dominance and noting its potential to disrupt traditional search business models.
  - Users highlight that **ChatGPT** represents a genuine technological revolution, with many comparing it to being the *"smartest person in the world"* who can help with endless tasks, though some still mistake it for a gimmick like **NFTs** or **cryptocurrency**.
  - Discussion focuses on monetization strategies, with users debating between subscription models and data-based revenue, while expressing hope that **OpenAI** won't resort to ad-based monetization like traditional search engines.


- **[OpenAI’s new defense contract completes its military pivot](https://www.technologyreview.com/2024/12/04/1107897/openais-new-defense-contract-completes-its-military-pivot/?utm_medium=tr_social&utm_source=reddit&utm_campaign=site_visitor.unpaid.engagement)** ([Score: 31, Comments: 22](https://reddit.com/r/OpenAI/comments/1h6odpi/openais_new_defense_contract_completes_its/)): **OpenAI** has not officially announced any defense contracts or military applications, and this appears to be misinformation without a credible source or post body to analyze. No factual summary can be provided without verifiable content to reference.
  - **OpenAI** announced a partnership with defense-tech company **Anduril** to deploy AI models for defending against drone attacks, focusing on data synthesis and situational awareness for **US and allied forces**.
  - The partnership specifically targets **unmanned aerial threats** and aims to protect **US personnel and facilities**, with spokesperson **Liz Bourgeois** emphasizing this aligns with company policies and won't develop harmful systems.
  - Community responses express skepticism about **AI safety** claims, noting the partnership between **Sam Altman** and **Palmer Luckey** with a tone of cynicism about the company's stated safety priorities.


**Theme 4. Claude 3.5 vs ChatGPT: User Migration and Comparison Trends**

- **How Claude 3.5 helped me fight off a $10,000 rental car damage claim - and won** ([Score: 99, Comments: 21](https://reddit.com/r/ClaudeAI/comments/1h6pxdn/how_claude_35_helped_me_fight_off_a_10000_rental/)): **Enterprise Rental Car** attempted to charge a user **$10,000** in damage fees by claiming their **Loss Damage Waiver (LDW)** only applied to business trips, despite the waiver being automatically included and unremovable during booking through an alma mater's rental program. Using **Claude 3.5** to analyze rental documentation and correspondence, the user identified that no business-use restrictions existed in the coverage terms, and with support from their school's **Risk Management office**, successfully disputed the claim, resulting in **Enterprise** dropping the **$10,000** charge entirely.
  - A user is currently leveraging **Claude** to contest a **$30,000 USD insurance claim** in first instance proceedings, demonstrating the AI's utility in legal documentation analysis. The case shows potential for resolution without legal escalation.
  - Users highlight the effectiveness of **human-AI collaboration** in legal disputes, with **Claude** demonstrating exceptional accuracy in document analysis and discovery when provided complete context and documentation.
  - Multiple users report declining service quality at **Enterprise**, with one detailing receiving a heavily damaged **Ram 1500** and a high-mileage **Chrysler 300c** as rental options, while another confirms losing their business after the **$10,000** damage claim incident.


- **[Have you noticed this pattern too?](https://i.redd.it/y1tbmo0l2w4e1.png)** ([Score: 50, Comments: 20](https://reddit.com/r/ClaudeAI/comments/1h6pt4s/have_you_noticed_this_pattern_too/)): A tweet by **@Aella_Girl** observes a growing trend of people switching from **ChatGPT** to **Claude** for personal advice and decision-making. The tweet gained significant traction with **284,600 views**, **2,100 likes**, **171 retweets**, and **98 comments** on **December 4, 2024**.
  - Users highlight **Claude's** ability to provide **nuanced responses** and **push back** on poor ideas, though it may be harder for new users to navigate compared to **ChatGPT**. The default **Claude** personality is more conversational while **ChatGPT** gives more bland responses.
  - A user shared their success with a **"Style >Intellectual Inquisitor"** prompt for **Claude**, which creates an analytical mindset focused on deconstructing arguments and identifying logical fallacies. They maintain just **3 different styles** for different purposes.
  - Despite individual preferences, **ChatGPT** remains the **market leader**, though **Claude's** popularity on **X** (Twitter) is seen as a significant signal. Users emphasize choosing tools based on effectiveness rather than brand loyalty.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1: Amazon Unveils Nova AI Models, Shakes Up AI Landscape**

- [**Amazon Drops Six New Nova Models to Rival GPT-4**](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws): Amazon announced six new foundation models in the **Nova** family at re:Invent, aiming to compete with GPT-4, offering support for up to **300K tokens** and **200+ languages**.

- [**Users Buzz Over Nova's Speed and Pricing**](https://x.com/_philschmid/status/1864016010464080260): Early users are excited about Nova's impressive **speed** and competitive **pricing**, eagerly anticipating integration into platforms like **Perplexity Pro**.

- [**AWS Bedrock Gets Supercharged with Nova's Launch**](https://aws.amazon.com/bedrock/pricing/): Amazon's Nova models are exclusively available via **Amazon Bedrock**, bolstering AWS's AI offerings and influencing developer choices.


**Theme 2: OpenAI's 12 Days of Announcements Ignite Anticipation**

- [**OpenAI Teases '12 Days of OpenAI'; Community Goes Wild**](https://x.com/OpenAI/status/1864328928267259941): OpenAI announced **12 days** of livestreams featuring launches and demos starting tomorrow, fueling excitement and speculation in the AI community.

- [**Rumors Swirl About OpenAI's Upcoming Surprises**](https://www.theverge.com/2024/12/4/24312352/openai-sora-o1-reasoning-12-days-shipmas): Users speculate on potential releases, including interface updates, new features for **ChatGPT**, and even a **text-to-video AI tool**.

- [**Developers Brace for OpenAI's Big Reveals**](https://x.com/sama/status/1864335461268754712): The community prepares for significant announcements, hoping for tools and improvements that could transform their projects and workflows.


**Theme 3: Cursor IDE Outages Push Users Toward Alternatives**

- [**Cursor Crashes; Developers Jump Ship to Windsurf**](https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221): **Cursor IDE** faces outages and performance issues, prompting frustrated users to revert to **ChatGPT** or switch to **Windsurf** for code assistance.

- [**Removal of Long Context Mode Sparks User Revolt**](https://forum.cursor.com/t/long-context-mode-gone-in-newest-update/29449): Cursor's elimination of key features like **long context mode** and interface changes leads to widespread dissatisfaction and backlash.

- [**Windsurf Rides the Wave as Cursor Sinks**](https://discord.com/channels/1074847526655643750): With Cursor's troubles, **Windsurf** emerges as a reliable alternative, gaining praise for better handling coding tasks without unnecessary code alterations.


**Theme 4: NVIDIA's SANA Model Slammed for Draconian License**

- [**Fast but Furious: NVIDIA's SANA License Sparks Outrage**](https://x.com/cloneofsimo/status/1864309440356470894): The **SANA** model impresses with speed but infuriates users with its restrictive **non-commercial license** and NVIDIA-only GPU usage requirement.

- [**Developers Fume Over SANA's GPU Lock-In**](https://x.com/cloneofsimo/status/1864312857674043599): The community criticizes NVIDIA for limitations preventing SANA's use on **AMD machines** and for retaining rights to generated outputs.

- [**SANA's License Blunder Sends Users Searching Elsewhere**](https://nvlabs.github.io/Sana/): Frustrated by SANA's restrictive terms, developers are turning to alternative models and openly accessible options for their AI projects.


**Theme 5: Pydantic AI Supercharges Development with New Integrations**

- [**Pydantic AI Teams Up with DSLModel and DSPy; Developers Rejoice**](https://ai.pydantic.dev/): The integration of **Pydantic AI** with **DSLModel** and **DSPy** provides an enhanced agent framework that simplifies AI development.

- [**Live Demo Promises to Master AI Development Magic**](https://youtube.com/live/mBQFKo8bPBI "Master AI Development: PydanticAI + DSPy + DSLModel Deep Dive"): An upcoming live demo titled *"Master AI Development"* will dive deep into combining **PydanticAI**, **DSPy**, and **DSLModel**.

- [**Coding the Future: Pydantic AI Makes LLMs a Breeze**](https://ai.pydantic.dev/): Developers praise Pydantic AI for making large language model integration seamless, especially when used with familiar tools like **FastAPI**.


---

# PART 1: High level Discord summaries




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor experiences outage**: Many users reported that **Cursor is experiencing outages**, leading to significant delays and an inability to generate responses.
   - Users expressed frustration over the lack of updates on the status and quality of responses, with some reverting to **ChatGPT** or switching to **Windsurf**.
- **Changes to Cursor features spark concerns**: The removal of **long context mode** and recent interface changes in **Cursor** have caused widespread dissatisfaction among users.
   - Many users noted a decline in the effectiveness of model responses, suggesting possible downgrades in model quality or performance issues.
- **Windsurf emerges as a reliable alternative**: **Windsurf** has been reported by some users as a dependable alternative, claiming it handles coding tasks better without significantly altering code.
   - This has led to discussions on whether **Cursor's** recent updates are a direct response to **Windsurf's** features and increasing popularity.
- **OpenAI announces 12 days of updates**: **OpenAI** is set to announce new updates daily for the next **12 days**, starting tomorrow, which has generated excitement among users.
   - Users are hopeful these announcements will bring improvements to existing tools, potentially addressing **Cursor's** recent challenges.
- **Issues with Cursor's performance persist**: Developers noted that **Cursor's** recent updates have not only slowed down responses but have also increased errors in code editing.
   - Users are questioning the effectiveness of these changes and are seeking potential solutions or workarounds.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX Dominates TPU Performance Over PyTorch**: Debate surged over whether **JAX** outperforms **PyTorch** in large AI labs, especially regarding TPU utilization versus PyTorch's GPU strengths.
   - Opinions varied as some members highlighted [Hacker News discussion](https://news.ycombinator.com/item?id=39876444) emphasizing JAX's efficiency on TPUs while others noted PyTorch's widespread adoption for GPU tasks.
- **Apple Leverages AWS Custom AI Chips**: At an [AWS event](https://www.macrumors.com/2024/12/03/apple-amazon-ai-chips-search/), **Apple** announced their use of AWS's custom **Inferentia** and **Graviton** AI chips for search services.
   - Despite this partnership, discussions pointed out that **Apple** continues to prefer GPU solutions for their extensive machine learning workloads.
- **Skepticism Surrounds Second Order Optimizers**: Members questioned the effectiveness of **second-order optimizers** in non-convex optimization, citing mixed empirical results compared to **AdamW**.
   - While some believe second-order optimizers could excel with tiny eigenvalues, the consensus leans towards no significant performance gains, as highlighted in recent community studies.
- **Mira Virtual AI Empowers Multimodal Tasks on 2GB VRAM**: **Mira Virtual AI** was introduced as a [GitHub project](https://github.com/Mirror-Prismals/Mira-Virtual-Ai) offering tools for multimodal conversions that run on consumer hardware with just **2GB of VRAM**.
   - Designed for users with limited coding experience, these self-contained scripts aim to make AI experimentation accessible and inject **fun and automation** into multimodal workflows.
- **Enhancing lm-eval-harness with External Loadable Evals**: Proposals were made to enable external loadable evaluations in **lm-eval-harness** via [Hugging Face](https://huggingface.co), allowing seamless dataset and eval configuration integrations.
   - Concerns about reproducibility and dataset versioning were raised, with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/f49b0377bf559f5558e8cd9ebd1190218c7df2a4/lm_eval/__main__.py#L197) currently supporting some external eval capabilities, though challenges remain.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Translation Tools Showdown**: Members debated various **AI translation tools**, favoring **DeepL** for its higher accuracy compared to **Google Translate** and **Microsoft** alternatives. Suggestions included leveraging **Cohere's API** and using **open-webui filters** to enhance chatbot multilingual capabilities.
   - The community emphasized the importance of precise translations in AI applications and discussed potential integrations to optimize language support for diverse user bases.
- **GPT Halts Image Processing**: A member reported that **GPT** is no longer capable of processing images, raising concerns about the repercussions of this capability change. This adjustment marks a significant shift in **GPT's functionalities**.
   - The limitation sparked curiosity among members about the underlying reasons and how it might affect future **AI workflows**.
- **Quantum Computing in Voting Systems**: Discussions explored the application of **quantum computing** in enhancing voting systems through advanced algorithms. Members debated the practicality of quantum algorithms in real-world voting scenarios.
   - One perspective highlighted that *voters are not in superposition*, questioning the immediate benefits of quantum technologies in electoral processes.
- **Cohere AI Excels in Hungarian Translations**: **Cohere AI**'s platform was recognized for supporting over **100 languages**, including **Hungarian**, with notably high translation accuracy. Members shared their positive experiences with **Cohere AI's multilingual capabilities**.
   - Resources such as [Mark Johns's YouTube video](https://www.youtube.com/watch?v=nUa_r9GKjtI) and the [OpenEmpathic project](https://dct.openempathic.ai/) were cited as valuable tools for leveraging **Cohere AI** in multilingual projects.
- **Innovative Prompt Engineering Techniques**: Members exchanged strategies for enhancing prompt engineering, including the use of **YAML structures** and **markdown formatting** to improve prompt clarity and context. Emphasis was placed on the significance of **contextual attention** in crafting effective prompts.
   - Discussions also covered the challenges of evaluating prompt effectiveness and the potential of **API automation** as a testing ground for various prompt strategies.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Amazon Bedrock Nova Model Introduced**: Amazon announced the new **Nova** series foundation models, available exclusively through [Amazon Bedrock](https://aws.amazon.com/blogs/aws/reduce-costs-and-latency-with-amazon-bedrock-intelligent-prompt-routing-and-prompt-caching-preview/), featuring context lengths up to **300K tokens**.
   - Performance is comparable to **Llama 3**, with competitive pricing tailored for different model capabilities.
- **Aider's New watch-files Feature**: The newly introduced `--watch-files` feature in [Aider](https://aider.chat/docs/usage/browser.html) enables seamless interaction with code through AI comments, triggering actions based on specified markers.
   - Early feedback praises the functionality as a significant advancement, although documentation is still being finalized.
- **Underperformance of QwQ Model**: The **QwQ 32B Preview** model achieved a score of **54%** for whole edit formats and **50%** for diffs, falling short of expectations.
   - Users are encouraged to consider **Qwen** or **Sonnet** models for better results, reflecting concerns about QwQ's practical utility.
- **Aider Docker Setup and Timeout Challenges**: Members discussed setting up [Aider in Docker](https://aider.chat/docs/install/docker.html) with shared volumes, encountering 'Permission denied' errors when aligning user settings in CentOS containers.
   - Additionally, timeout issues persist when running Aider with a local server using `--timeout 5000`, possibly due to a litellm bug.
- **MCP Adoption and OpenAI's Development Strategy**: The **MCP** is viewed as a future cornerstone by members, with strong community interest in its adoption.
   - There are concerns that **OpenAI** might choose to *reinvent the wheel* instead of integrating MCP into their development strategy.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Networking Features Awaiting Updates**: A discussion highlighted ongoing developments in **Mojo's networking capabilities**, targeting **25-40 Gbps of TCP throughput** per core with advancements in [io_uring](https://github.com/marti).
   - Members emphasized the need for efficient **API design** post-update to meet modern requirements.
- **Exploring SIMD Operations in Mojo**: Members explored the usage of [SIMD](https://github.com/simdjson/simdjson) operations in **Mojo**, noting its user-friendly implementation compared to C/C++ intrinsics.
   - **Darkmatter** suggested embedding most SIMD intrinsics into the standard library to reduce reliance on direct intrinsic calls.
- **Developing a High-Performance File Server**: A member shared plans to develop a **high-performance file server** for a game, aiming for a **30% increase in packets/s** over Nginx's 200-byte HTTP header parsing.
   - Strategies discussed included achieving efficiency and the necessity for robust network API support.
- **Inline References Concept Proposed**: The introduction of an `InlineReference` type was proposed, facilitating memory-efficient access patterns without storing addresses, potentially enhancing performance by enabling **contiguous memory reads**.
   - The discussion touched on balancing **reference usability** and **compiler visibility**, with concerns about integrating this feature.
- **Memory Optimization Strategies in Mojo**: Focused on **small string and vector optimizations**, members emphasized that these could boost **performance** by enabling **zero-copy scenarios** during large array scans.
   - Interest was expressed in practical use cases and effective implementation methods for these optimizations.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Dynamic 4-bit Quantization**: Unsloth introduced [Dynamic 4-bit Quantization](https://x.com/UnslothAI/status/1864380913922265300), enhancing **model accuracy** while reducing VRAM usage compared to traditional 4-bit methods.
   - The method dynamically opts out of quantizing certain parameters to prevent accuracy loss, requiring users to rename their model to 'unsloth-bnb-4bit' to activate the mode.
- **Llama 3 Fine-tuning Challenges**: Users are experiencing **fine-tuning errors** with **Llama 3**, encountering runtime issues when saving models to GGUF format due to missing files in `llama.cpp`.
   - Attempts to resolve these issues by switching notebook versions have failed, and the only current workaround involves using the **Unsloth framework** for GGUF conversions.
- **GGUF Conversion Techniques**: Amid **GGUF conversion challenges**, community members are exploring alternative methods and **Colab setups** to properly convert models, primarily utilizing the **Unsloth framework**.
   - Participants have shared [Colab resources](https://colab.research.google.com/drive/12hkbsOMJfYfmqJLA93cV5tGoPIeZ5gDK#scrollTo=oAC_WYSUX7k_) and potential solutions to navigate the limitations in current conversion processes.
- **Role of Continued Pretraining**: The community highlights the **importance of Continued Pretraining (CPT)** for models such as **Llama 3**, enabling them to adapt to new domains and acquire new tokens effectively.
   - While base models undergo extensive pretraining on large datasets, **CPT** remains crucial for specialized applications in fields like law and medicine to maintain relevance and accuracy.
- **Claude vs CodeLlama: Model Performance**: Debate arose comparing **Claude** and **CodeLlama**, with members deeming **CodeLlama** outdated and advocating for models like **Qwen2.5-coder** as superior alternatives.
   - **Qwen2.5-coder** has been noted to deliver performance akin to **Claude**, reinforcing its position in current model discussions and applications.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Amazon Nova Models Launch**: The [Amazon Nova launch](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws) impressed users with its **speed** and **accuracy**, generating eager anticipation for integration into **Perplexity Pro**.
   - Early experimentation showed positive feedback, highlighting Nova's potential for high-performance AI-driven tasks among the engineering community.
- **Perplexity Pro Subscription Issues**: Users expressed frustration over **Perplexity Pro** subscription costs, particularly the transition from the **$4.99 first month** pricing to higher charges without clear communication.
   - This led to broader discussions about the financial model supporting free access for students and the implications for **API access** and pro features.
- **Perplexity API Quality Concerns**: Members raised significant issues regarding the **quality of the Perplexity API**, noting it has become **unusable** for certain use cases.
   - With multiple users expressing dissatisfaction, there's speculation about potential provider changes and ongoing challenges with API performance.
- **User Interface Problems on Mac**: **Perplexity AI's Mac application** has been criticized for **slow performance** and an awkward interface compared to the web version.
   - Users also reported **battery drain** issues, prompting conversations about upcoming fixes and improvements.
- **Heisenberg Heat Inquiry**: A discussion was initiated around the **Heisenberg Heat** concept, inviting exploration into its principles and implications for AI engineering.
   - Members are encouraged to dive into the associated **theoretical inquiries** and **practical applications** presented in the shared link.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.5 Haiku Price Reduction**: OpenRouter announced a **20% price reduction** for **Claude 3.5 Haiku**, aiming to make the model more accessible.
- **Hermes 405B Service Termination**: The free service for **Hermes 405B** has been discontinued, likely due to provider decisions, leading to disappointment among users.
   - Despite the termination, the **base 405B model** remains available for free, prompting some users to explore alternative options.
- **Gemini Ultra Access Restrictions**: **Gemini 1.0 Ultra** is currently subject to allowlists, with rumors of availability amid concerns over potential discontinuation.
   - Users are confused by the rollout and versioning of Google's models, speculating that Ultra might be discontinued soon.
- **Amazon Nova for Creative Writing**: There is curiosity about the effectiveness of the **Amazon Nova** model for creative writing tasks, with users seeking personal experiences.
   - Specs on Nova's capabilities compared to alternatives like Runway remain uncertain as its evaluation continues.
- **Custom Provider Keys Beta Access**: **Custom Provider Keys** feature is in beta, with users requesting early access and anticipating possible future fees.
   - One member pleaded, *'I would like the custom key beta access as well!'*, while another shared gratitude for the team's efforts regardless of the timeline.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Distributed Training Run Nears Completion**: A **distributed training run** is currently underway and is set to complete in just over a day, involving pre-arranged compute partners from the onset.
   - Further details about the training run's completion are expected soon, with discussions about potential **public involvement** acknowledged within the community.
- **Forge Reasoning API Beta Officially Launched**: Nous Research has launched the **Forge Reasoning API Beta**, aiming to enhance inference times for various models and potentially boost the capabilities of **Hermes 70B**.
   - This development responds to community interest in **large-scale foundation models** and their practical applications, as noted in the [official announcement](https://x.com/NousResearch/status/1856417883934601246).
- **Debate on Implementing Live Memory in LLMs**: Members discussed strategies for **implementing live memory** within LLM architectures, weighing the use of function calls against **RAG methods** for improved consistency and performance.
   - There was a consensus favoring **classical approaches** to better ground neural networks reliably while maintaining style consistency.
- **Linux from Scratch Proposed as AI Benchmark**: A query was raised about the feasibility of utilizing the **Linux from Scratch** book as a benchmark for evaluating AI agents.
   - This indicates a move towards establishing **concrete metrics** for assessing agent performance in real-world scenarios.
- **Integrating Momentum into Residual Stream Architecture**: A member proposed incorporating the concept of **momentum** into the **residual stream** architecture, questioning its mathematical underpinnings.
   - This sparked a discussion on whether **addition and skip connections** are sufficient for achieving similar performance enhancements.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Teams Up with Spotify for AI Podcasts**: On [December 4, 2024](https://blog.google/technology/google-labs/notebooklm-spotify-wrapped/), **NotebookLM** partnered with **Spotify** to launch the **Spotify Wrapped AI Podcast**, offering a personalized audio recap of users' yearly music preferences.
   - The podcast utilizes **NotebookLM** to analyze users' favorite tracks and artists, featuring **AI hosts** that dissect defining moments in their musical year.
- **AI Audio Generation Enhancements in NotebookLM**: Members showcased **AI-generated multilingual audio** clips, highlighting **NotebookLM**'s capability to produce content in multiple languages, despite occasional focus loss.
   - Discussions included inquiries about **Polish language support**, indicating ongoing improvements in language processing settings.
- **Revolutionizing Sports Journalism with NotebookLM**: **NotebookLM** is being leveraged to create nightly pregame and postgame feature stories for professional sports teams, enabling scalable content generation.
   - Users emphasized the ease of generating branded avatars and enhancing fan engagement through automated storytelling.
- **Legal Content Simplification via NotebookLM**: Users praised **NotebookLM** for effectively parsing complex legal jargon, making information on data laws across states more accessible.
   - It is cited as a daily tool for simplifying legal documents, enhancing understanding for non-experts.
- **Language Settings Challenges in NotebookLM**: Users reported difficulties in changing language settings within **NotebookLM**, particularly for podcast content despite adjusting their Google account to languages like Indonesian.
   - There were expressions of confusion and disappointment when attempts to generate audio in languages such as Portuguese failed after script uploads.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Amazon Launches 6 New Foundation Models**: During [re:Invent](https://link.to.amazonbedrock), **Amazon** announced **6 new foundation models** including **Nova Micro** and **Reel**, supporting **up to 300K tokens** and **200+ languages**.
   - These models, available exclusively through [Amazon Bedrock](https://link.to.amazonbedrock), aim to provide text-to-video generation capabilities, with pricing starting at **$0.035** for Micro models.
- **NVIDIA's SANA License Faces Backlash**: **NVIDIA** introduced the **SANA model**, praised for speed but criticized for licensing that restricts usage to non-commercial applications and **NVIDIA GPUs only**.
   - Users voiced concerns over limitations like incompatible use on AMD machines and **NVIDIA** retaining rights to generated outputs, as discussed in [this tweet](https://x.com/cloneofsimo/status/1864309440356470894).
- **IFEval Benchmark Saturation Questioned**: Members debated the relevance of the **IFEval benchmark**, noting that **90% benchmarking** is now commonplace with many achieving high scores.
   - This has led to discussions on the potential need for new meta benchmarks to better assess AI models' performance.
- **Anduril Partners with OpenAI for US AI Leadership**: **Anduril Industries** and **OpenAI** formed a partnership to advance **U.S. artificial intelligence** leadership, integrating **Lattice** systems for security across domains.
   - The collaboration focuses on supporting armed forces missions with innovative AI technologies, as detailed in [Anduril's announcement](https://www.anduril.com/article/anduril-partners-with-openai-to-advance-u-s-artificial-intelligence-leadership-and-protect-u-s/).
- **Mistral Large 2 Outperforms GPT-4 in Bash Scripts**: **Mistral Large 2** was praised for outperforming **GPT-4** and **3.5 Sonnet** in handling bash scripts and queries, as shown in [Xeophon's tweet](https://x.com/TheXeophon/status/1833921199170355480).
   - Users humorously noted that with AI and an online bash interpreter, recalling **ffmpeg flags** is no longer necessary.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gram Matrix Gains Efficiency**: A user discussed methods for efficiently computing the upper triangle of a Gram matrix (**A@A^T**) without performing a standard matrix multiplication followed by a triplet upper function, suggesting the use of [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) to compute only relevant tiles and alternatives like **cuBLAS's syrk** and **cutlass**.
   - Resources such as [Triton's matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) were shared to assist in mastering matmul kernel optimizations, although some noted the materials may not be beginner-friendly.
- **Triton's MLIR Documentation Deep Dive**: Discussions centered on the availability of documentation for Triton's MLIR Dialects, referencing the [Triton Ops documentation](https://triton-lang.org/main/dialects/TritonOps.html) and noting the minimal [programming guide](https://github.com/triton-lang/triton/tree/main/docs/programming-guide).
   - Challenges such as writing a Grouped GEMM with TMA in Triton were addressed, with mention of a [pull request](https://github.com/triton-lang/triton/pull/4498) aimed at enhancing functionality, though full support remains uncertain.
- **KernelBench's Crucial Benchmarking**: 🌽 [KernelBench](https://twitter.com/anneouyang/status/1864014135824162995) (Preview) was introduced as a new coding benchmark designed to evaluate LLMs' ability to generate **efficient** GPU kernels for neural network optimization.
   - Concerns were raised about some **fastest kernels** on the leaderboard appearing incomplete, with users sharing specific solutions like the [incomplete kernel](https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs/assets/solutions/fc7b0633e1f8dca6653f552f2eeef450.py) for scrutiny.
- **Tenstorrent's Tremendous AI Funding Surge**: A member announced that **Tenstorrent** secured **$700M** in funding this week, contributing to a notable recent surge in funding within the AI sector.
   - The announcement included a link to a [YouTube video](https://www.youtube.com/watch?v=_aqMdhAgGG8) featuring Jim Keller discussing AI's impending impact on computing.
- **Thunderkittens Tackle Race Conditions**: A user reported experiencing a **race condition** during custom kernel implementation using **TK's WGMMA+tma**, caused by alignment issues in the K dimension.
   - They developed an innovative **masking function** to handle out-of-bounds rows by loading zeros into shared memory, yet **memcheck/synccheck/initcheck** reported no errors, complicating debugging efforts.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Discord's Deceptive Bots Attack Community**: Several **bots** are infiltrating the Discord community, executing scams like **Ponzi schemes** or impersonating **Discord support**. Users were advised to [report these bots](https://discord.com/report) and avoid interacting with them.
   - Community members emphasized vigilance against these bots to maintain the integrity of the Discord environment.
- **Stable Diffusion Starters Seek Tool Guidance**: A newcomer expressed confusion over tools and models in **Stable Diffusion**, fearing scams. Users recommended **[Vast.ai](https://vast.ai/)** for cloud GPU rentals and suggested starting with **ComfyUI** tutorials by Scott on YouTube for streamlined workflows.
   - The community stressed the importance of utilizing reliable resources like **Vast.ai** to mitigate the risk of encountering scams during the onboarding process.
- **ComfyUI Champions Advanced AI Art Workflows**: **ComfyUI** was highlighted as an optimal platform for creating AI art, particularly beneficial for beginners. Users stressed the significance of **watching introductory videos** to maximize its potential.
   - Additionally, the necessity of a robust GPU for local AI operations was underscored, with discussions around cloud options presenting them as cost-effective alternatives.
- **LoRA Model Glitches in Stable Diffusion**: Users reported issues with **LoRA models**, noting the need for specific trigger words in prompts for correct functionality. Problems causing image results to appear jumbled were attributed to various **Stable Diffusion** settings.
   - The community discussed optimizing settings to resolve image generation inconsistencies and enhance overall performance.
- **Boosting SD with Performance Analysis Tools**: A user expressed intent to develop performance analysis tools for **Stable Diffusion**, citing the current deficiency in such resources. This initiative was met with agreement from others who believe the **SD ecosystem** requires enhancements to improve user experience.
   - The community recognizes the potential impact of performance tools in advancing the capabilities and usability of **Stable Diffusion**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Amazon Nova Models Announced**: At [AWS re:Invent](https://youtu.be/LY7m5LQliAo?t=6657), Amazon introduced its **Nova** family of foundation models, including text and video-generating models available on **Amazon Bedrock**, positioning itself against leading competitors like **GPT-4**.
   - Community feedback is emerging, focusing on **Nova's performance** compared to **OpenAI's offerings**, with initial benchmarks indicating competitive results.
- **AWS Launches New Usage API**: AWS released the **Usage API**, allowing developers to programmatically track usage and costs. This includes monitoring token usage by time and filtering by various identifiers.
   - The new functionality aims to enhance transparency and management for developers utilizing **AWS services**, facilitating better resource allocation.
- **PydanticAI Framework Released**: **Pydantic** launched **PydanticAI**, a framework designed to streamline the development of applications powered by large language models, emphasizing **type safety** and **modularity**. It is currently in **beta** and open-sourced under the **MIT License**.
   - The framework targets developers seeking accessible options to incorporate **LLMs** into their projects, promoting ease of integration and extensibility.
- **OpenAI's 12 Days of Announcements**: **OpenAI** commenced its **12 Days of Announcements** event on December 5th, featuring daily launches, demos, and updates. Early statistics include **300 million weekly active ChatGPT users** and **1 billion daily messages** sent on the platform.
   - Key highlights anticipated include the introduction of a potential **text-to-video AI tool**, generating excitement within the AI engineering community.
- **Genie 2 Debuts from Google**: **Google** unveiled **Genie 2**, an **autoregressive latent diffusion model** designed for **video generation** and **interactive environments**. The model leverages a **transformer dynamics** framework to enhance action controllability in generated content.
   - Community discussions are focused on the model's **output length** and its practicality for generating **videos**, indicating a keen interest in its applications.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Windows Download Glitches**: Users reported issues downloading the Windows x86 version of [LM Studio](https://lmstudio.ai), encountering messages about unavailable files.
   - Others suggested potential CDN problems and recommended using a VPN to attempt the download again.
- **Performance Degradation on Windows vs Mac for LM Studio**: A member experienced significant performance issues running **LM Studio** on Windows compared to Mac, including unexpected output characters from the model.
   - Troubleshooting suggestions included toggling the `Flash Attention` switch and verifying system specifications.
- **Leveraging LLMs as RPG Game Masters**: A user shared their experience using an **LLM** to conduct a pre-planned RPG adventure, highlighting the novelty of writing the outline in Thai to prevent foreknowledge.
   - The experiment resulted in engaging outcomes, sparking interest in discussing methodologies and community resources for AI-driven RPG gameplay.
- **Optimizing LM Studio with Local Network GPUs**: A user inquired about connecting **LM Studio** to a local server with multiple GPUs from their laptop for enhanced performance.
   - Another member confirmed feasibility, noting the requirement of a frontend to ensure proper functionality.
- **Skepticism Around Intel's Arc Battlemage GPUs**: Users expressed concerns about the new **Arc Battlemage** cards, questioning the reliability of **Intel GPUs** for AI tasks due to inadequate driver support.
   - *One comment highlighted that using fewer, larger memory GPUs like the 3090 is preferable.*



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Building AI apps on Vercel just got easier**: The latest [update from LlamaIndex](https://twitter.com/llama_index/status/1864002184138170677) simplifies AI app development on Vercel, enhancing integration capabilities with LlamaCloud.
   - This progression could boost developer productivity and streamline AI app deployment processes.
- **Amazon launches competitive Nova models**: Amazon's new family of foundation models, **Nova**, boasts competitive benchmarks and more attractive pricing compared to competitors; ensure support by installing via `pip install llama-index-llms-bedrock-converse` [link here](https://twitter.com/llama_index/status/1864080917029085459).
   - The foundation models aim to offer users a cost-effective and performance-driven alternative in the AI model landscape.
- **Rapid RAG implementation with LlamaIndex Workflows**: Learn to build a high-performance Retrieval-Augmented Generation (RAG) system with LlamaIndex Workflows, featuring an event-driven architecture [details here](https://twitter.com/llama_index/status/1864377849295327365).
   - The guide compares this approach with other frameworks such as LangGraph, emphasizing efficiency in complex AI scenarios.
- **Summary Index Performance Concerns**: A user raised issues about the slow response time with the **summaryindex** using **sentencesplitter**, stating it takes around **2 minutes** to generate a summary compared to **8 seconds** with ChatGPT.
   - They explored potential improvements but acknowledged that using routers and indexing methods introduces latency.
- **Optimizing Prompts for LLMs**: A user experiencing hallucinations with OpenAI LLMs was advised to try **prompt optimization** to improve response accuracy.
   - It was suggested that crafting better instructions can lead to enhanced performance from the language model.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Rerank 3.5's Multilingual Boost**: Cohere launched **Rerank 3.5**, supporting both **multilingual** and **English** rankings across **100+ languages**, enhancing search capabilities as detailed in our [blog post](https://cohere.com/blog/rerank-3pt5).
   - A user reported a **30% performance drop** with 'rerank-multilingual-v3.0', and concerns were raised about the new **rerank 3.5** model's effectiveness, prompting Cohere's **support team** to assist in troubleshooting.
- **Cohere Toolkit Error Fixes**: Users encountered warnings when running the **cohere-toolkit**, specifically related to *alembic* and compatibility issues with **PyTorch 2.5.1**.
   - Community members are seeking solutions, with suggestions to consult Cohere's **support team** for resolving these issues.
- **Harmony's LLM Matching Competition**: The **Harmony project** is hosting a competition on [DOXA AI](https://harmonydata.ac.uk/doxa/) to enhance their **LLM matching algorithms**, offering prizes up to **£500** in vouchers for participants.
   - Participants can join via Harmony's Discord server in the 🏅「matching-challenge」 channel, with no prior LLM experience required.
- **Model Deprecation Guidelines**: Cohere updated their **model deprecation** policies, outlining the lifecycle stages of models including **Active**, **Legacy**, and **Deprecated**, available in the [Deprecations — Cohere](https://docs.cohere.com/docs/deprecations) documentation.
   - Developers are encouraged to consult the documentation to identify recommended replacements for any deprecated endpoints and models.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pydantic AI Boosts DSLModel Capabilities**: Integrating [Pydantic AI](https://ai.pydantic.dev/) with **DSLModel** introduces an agent framework that enhances the usability of LLMs through Pydantic's robust features.
   - A member highlighted how **Pydantic** streamlines AI project development when combined with frameworks like **FastAPI**.
- **Master AI Development Live Demo Scheduled**: A live demo titled [Master AI Development: PydanticAI + DSPy + DSLModel Deep Dive](https://youtube.com/live/mBQFKo8bPBI) is set to explore advanced AI development technologies.
   - The event aims to demonstrate innovative methods for leveraging **PydanticAI** and associated tools in AI projects.
- **DSPy Optimizations Hit AWS Lambda's Time Limit**: Members discussed the challenges of executing **DSPy optimizations** on **AWS Lambda**, particularly the enforced **15-minute execution limit** for prolonged tasks.
   - A proposed solution involves using the **/tmp folder** for caching to address Lambda's read-only filesystem and improve processing speeds.
- **ProgramOfThought to Undergo Revamp in v2.6**: **ProgramOfThought** is slated for a revamp in **v2.6**, addressing concerns about its support status following **v2.5**.
   - Users are advised to employ the current version cautiously as the upcoming upgrade is anticipated within the year.
- **Developing Precision Metrics Amid Class Imbalance**: A member inquired about developing a **precision metric** for a specific class within a **multi-class classification** problem characterized by significant class imbalance.
   - **dspy.Example(batch=[...])** was recommended for handling the evaluation, though challenges persist due to the **class imbalance**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Sierra AI Info Session**: An exclusive [Sierra AI Info Session](https://www.youtube.com/watch?v=-iWdjbkVgGQ) was held, showcasing their conversational AI platform and inviting talented developers to participate.
   - Sierra AI is keen to connect with developers ahead of the hackathon, emphasizing the importance of the upcoming submission deadline on **December 17th**.
- **Hackathon Submission Process Transition**: The **LLM Agents MOOC Hackathon** has shifted its submission process from **Devpost to Google Forms**, with the [Submission Form](https://forms.gle/jNr8nSH9Cy9qpYcu5) now live.
   - Participants are encouraged to refer to the [Submission Requirements Guide](https://docs.google.com/document/d/1WgWLZocBFM08cVVxo9P-ZMCnHBLGmQ7v8PbH4-AwnSk/edit?usp=sharing) to prepare their projects for the **December 17th** deadline.
- **Certificate Declaration and Completion Tiers**: The **Certificate Declaration Form** is now available [here](https://forms.gle/nYGJLPTdb7af2Dq59), outlining the five course completion tiers: Trailblazer, Mastery, Ninja, Legendary, and Honorary.
   - Participants must complete all coursework, including **12 quizzes** and a written article, by **December 12, 2024**, to be eligible for their selected tier.
- **GPT-4 Data Leak Concerns**: Concerns were raised regarding a potential data leak in **GPT-4**, specifically whether it affects the consumer or enterprise versions, with implications of user data sharing defaults.
   - A possible **GPT-4 jailbreak** could expose real PII from the training set, drawing attention to comparisons with the historic **AOL case**.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Resolving Anthropic Branch TypeError**: A user encountered a **TypeError** related to the unexpected 'proxies' argument in the latest **Anthropic Development Branch** of Open Interpreter. [Discussion thread](https://discord.com/channels/1146610656779440188/1147665339266650133/1313500744263143464) suggests checking for a custom API base as the primary troubleshooting step.
   - Another member recommended verifying client initialization settings, indicating that the 'proxies' argument might be the sole change causing the issue.
- **Open Interpreter Installation Rewritten for Performance**: **Open Interpreter** has been completely rewritten to enhance performance. Users are encouraged to reinstall the latest development version using `pip install --force-reinstall git+https://github.com/OpenInterpreter/open-interpreter.git@development`.
   - The developer emphasized the importance of user feedback to identify any missing features and ensure the new implementation outperforms previous versions.
- **Enhanced Linux Compatibility Confirmed**: **Open Interpreter** operates smoothly on **Garuda-Linux**, an Arch-Linux fork, as confirmed by a user. [Full compatibility details](https://discord.com/channels/1146610656779440188/1147665339266650133/1313500744263143464) also highlight successful tests on **Manjaro** and **OpenSuse** distributions.
   - The extensive testing across multiple Linux versions underscores the software's adaptability and reliability in diverse environments.
- **LiveKit Powers Remote Device Connections**: **LiveKit** is utilized by **O1** to connect devices like **iPhones** with laptops or **Raspberry Pi** for handling requests. This setup facilitates efficient remote access through the local **OpenInterpreter** instance.
   - The integration allows users to control their machines remotely, leveraging **LiveKit**'s capabilities to enhance device interoperability.
- **OpenInterpreter's CLI Maintains Robust Functionality**: Despite being in **CLI form**, **OpenInterpreter** provides effective computer operation capabilities. Users can bypass approval requirements using the `interpreter -y` command for seamless code execution.
   - This feature ensures user safety by requiring approval before executing code, while still offering flexibility for advanced operations.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Genie 2 Takes Center Stage**: A request was made to add information about **Genie 2**, a large-scale foundation world model, to torchtune within the next day. More details can be found in the [official blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/).
   - The acknowledgements highlight contributions from key figures like **Jack Parker-Holder** and **Stephen Spencer**, emphasizing collaborative efforts in the project's development.
- **Federated Learning Shows Promise**: The underlying **federated learning** approach may yield better results than fully synchronous methods, as discussed in a shared [paper](https://arxiv.org/pdf/2411.19870).
   - *Only 22 hours left on training* indicates nearing completion.
- **Generalist Agents Team Advances**: The **Generalist Agents team**, led by Vlad Mnih, made significant strides with contributions from members like **Harris Chan** and **Maxime Gazeau**, showcasing a comprehensive approach to agent development.
   - Further support from the **SIMA team**, including **Frederic Besse** and **Tim Harley**, underscores the diverse expertise within the initiative.
- **Community-led GPU Contributions Potential**: There's interesting potential for community-led efforts similar to **Folding@home**, with individuals contributing GPU time.
   - This could become crucial as models outgrow individual data centers.
- **MMLU Pro Sets Validation Standards**: To validate a block in the discussed framework, the model needs to achieve **90%** on **MMLU Pro**.
   - This highlights the rigorous performance standards necessary for successful deployments.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Mechanistic Interpretability Enhances Cellular Analysis**: Researchers introduce **mechanistic interpretability**, a tool to explore how cells model their environments, shifting focus from genes to **gene regulatory modules** and **sub-cellular locations**.
   - This approach may allow the construction of a 'folk psychology of cellular behavior', providing insights into the **inner life of cells**.
- **Diffusion Model's Non-commercial License Restricts Adoption**: A member highlighted that the diffusion model's **non-commercial license** should deter attempts to implement it widely.
   - This restriction could impact the adoption and experimentation with the model among developers.
- **EDM2 Framework Applied to Text-Conditioned Diffusion Models**: A member inquired about utilizing the **EDM2 framework** for training diffusion models with text conditioning.
   - They referenced a [paper](https://arxiv.org/pdf/2312.02696) showcasing **impressive results**, highlighting a gap in specific implementations.
- **Class Conditioning Limits Diffusion Model Flexibility**: The paper discussed **class conditioning**, limiting the model to generating outputs for a few predefined classes.
   - This limited approach contrasts with the desired flexibility of text conditioning, allowing broader creativity in generation.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **SAM from Meta Stuns with User-Friendly Demo**: A member showcased **SAM from Meta** on its [demo website](https://segment-anything.com/demo), highlighting its **600M image embedding transformer** running in the cloud and smaller models operating directly in the browser.
   - The demo underscores the **effectiveness** of SAM models out of the box and sets a **quality baseline** for future **tinygrad** models and community traction.
- **Web Models Surge with ONNX Integration**: Discussions emphasized the development of **Web models** like **ONNX in the cloud**, enhancing **accessibility** in machine learning tools.
   - These models offer functionalities that run both in the cloud and directly in the browser, demonstrating potential for increased **user engagement**.
- **Adjusting Threadgroup/Grid Sizes in tinygrad**: A user inquired about altering **threadgroup/grid sizes** during graph rewrite optimizations in `uopgraph.py`, to which George Hotz responded they can be modified in **OptOps** within **kernel.py**.
   - This flexibility allows for customized optimization strategies in **tinygrad**'s architecture.
- **BEAM Search Insights Shared**: A user posted on [BEAM Search](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241203_beam.md), providing an explanation of **beam search** and **kernel optimization options** within **tinygrad**.
   - The resource serves as a valuable guide for understanding these concepts and their application in **tinygrad** development.
- **JIT Functions Overwrite Outputs**: A note about **JIT functions** revealed that after the first call, jitted functions **reuse the same output buffer**, which may overwrite previous results.
   - To preserve results, it's necessary to use `.clone().realize()` after each call.



---



## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **ADOPT Optimizer Integration into Axolotl**: The **ADOPT optimizer** has been integrated into the Axolotl codebase to enhance **training stability**, as detailed in [pull request #2104](https://github.com/axolotl-ai-cloud/axolotl/pull/2104).
   - This update ensures compatibility with the current **torch version** and incorporates the latest modifications from the original author [here](https://github.com/iShohei220/adopt).
- **ADOPT Optimizer Achieves Optimal Convergence**: Members discussed the capability of the **ADOPT optimizer** to achieve **optimal convergence** with any beta value.
   - This flexibility is considered a key strength, allowing for versatile training scenarios.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Unternet seeks Open Source Engineer**: [Unternet is hiring an **Open Source Engineer**](https://discord.com/channels/1089876418936180786/1313839138562248737) to contribute to open source projects, create technical documentation, and engage with the community.
   - The job position emphasizes the importance of collaborating with the community while also developing technical documentation, aimed at individuals passionate about open source contributions.
- **Community Engagement Opportunity**: The job position emphasizes the importance of collaborating with the community while also developing technical documentation.
   - This role is aimed at individuals passionate about open source contributions.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla Model Fails to Start**: A user encountered an error when attempting to start their **Gorilla model**, indicating a dependency issue related to the **tokenizer**.
   - The error message highlighted the absence of the **protobuf library**, despite it being installed in their environment.
- **Protobuf Library Not Recognized**: The user confirmed that the **protobuf** package was installed with version **5.29.0**, but the system still reported it as missing.
   - This has led to questions about what could be causing the environment to not recognize the installed package.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Member Follows Up on Ticket Message**: A member prompted **Nick** to check a message they sent about their **ticket**, requesting him to look at it when he has time.
   - They emphasized the importance of timely responses, hinting at the need for quick resolution.
- **Lack of Additional Context in Ticket Conversation**: The conversation regarding the **ticket** did not provide any further context beyond the follow-up.
   - There were no additional comments or links discussed.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1313490637697323070)** (476 messages🔥🔥🔥): 

> `Cursor outages, Changes to Cursor features, Windsurf vs. Cursor performance, OpenAI 12 Days of Announcements, Issues with Cursor's performance` 


- **Cursor experiences outage**: Many users reported issues with Cursor being down, experiencing significant delays, and unable to generate responses.
   - Users expressed frustration with the lack of updates on the status and quality of responses, with some reverting to ChatGPT or switching to Windsurf.
- **Changes to Cursor features spark concerns**: The removal of long context mode and the new interface changes in Cursor have led to dissatisfaction among users.
   - Many users noted a decline in the effectiveness of the model responses, suggesting possible downgrades in model quality or performance issues.
- **Windsurf emerges as a reliable alternative**: Some users have reported positive experiences with Windsurf, claiming that it better handles coding tasks without altering too much code.
   - This has led to discussions on whether Cursor's recent updates are a direct response to Windsurf's features and hype.
- **OpenAI announces 12 days of updates**: OpenAI is set to announce new updates daily for the next 12 days, starting from tomorrow, which has generated excitement among users.
   - Users are hopeful that these announcements will lead to improvements in existing tools, possibly addressing Cursor's recent challenges.
- **Issues with Cursor's performance persist**: Many developers noted that Cursor's recent updates have not only slowed down responses but have also led to increased errors in code editing.
   - Users are questioning the effectiveness of the changes and are seeking potential solutions or workarounds.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com">Medium: Read and write stories.</a>: On Medium, anyone can share insightful perspectives, useful knowledge, and life wisdom with the world.</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">How to do `Fix in Composer` and `Fix in Chat` actions from keyboard</a>: These 2:     I could not find it in settings.</li><li><a href="https://forum.cursor.com/t/long-context-mode-gone-in-newest-update/29449/34">Long context mode gone in newest update</a>: Thanks for sharing your thoughts, and apologies for the radio silence! I wanted to explain our reasoning behind the 0.43 feature deprecations. We love shipping early experiments to get feedback (findi...</li><li><a href="https://forum.cursor.com/t/feature-request-long-context-mode-upvote/32187">Feature request: Long context mode (upvote!)</a>: Being able to utilize full LLM context would be very helpful. Please restore this 🙂</li><li><a href="https://medium.com/@NFAblog/connect-github-codespaces-to-cursor-ai-ai-friendly-vs-code-clone-243fa5f79414">Connect Github CodeSpaces to Cursor Ai (Ai friendly vs code clone)</a>: Connecting GitHub Codespaces to CURSOR.DEV: A Developer’s Guide</li><li><a href="https://status.cursor.com/">Cursor Status</a>: no description found</li><li><a href="https://github.com/TheGalaxyStars/KEPLER-COMMUNITY">GitHub - TheGalaxyStars/KEPLER-COMMUNITY</a>: Contribute to TheGalaxyStars/KEPLER-COMMUNITY development by creating an account on GitHub.</li><li><a href="https://cursor.directory/">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://github.com/getcursor/cursor/issues/2027">WSL extension is supported only in Microsoft versions of VS Code · Issue #2027 · getcursor/cursor</a>: If you can, please include a screenshot of your problem Please include the name of your operating system If you can, steps to reproduce are super helpful I am developing using Windows 11 + WSL: Ubu...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1313603284367577128)** (198 messages🔥🔥): 

> `JAX vs PyTorch Performance, Apple's use of AWS AI chips, Training methods and frameworks, Schedule-free optimizers, Embedding techniques for images with coordinates` 


- **JAX's Adoption Among Large Labs**: There is a debate regarding whether leading AI labs primarily use JAX over PyTorch, with varying opinions on the performance benefits and industry usage.
   - Some members argue that while JAX is favored for TPUs, many organizations still rely heavily on PyTorch, especially for GPU tasks.
- **Apple's Relationship with AWS**: Apple confirmed at an AWS event that they utilize AWS custom AI chips, stating they have a robust partnership in AI research.
   - Discussions indicated that despite Apple's use of AWS hardware, they still prefer GPU options for their substantial machine learning tasks.
- **The Evolution of Training Frameworks**: There's discussion on the use of different optimizers in ML training, specifically whether schedule-free optimizers like muon have gained traction over AdamW.
   - While schedule-free optimizers are noted as niche, it seems that AdamW continues to be widely adopted in practice.
- **Optimizing Image Embeddings**: One user is exploring methods to incorporate 2D coordinates into image embeddings, debating whether to concatenate channels or apply alternative techniques.
   - The discussion includes references to rotary embeddings and examples such as StyleGAN, highlighting various approaches to improve model efficiency.
- **New Developments in ML Research**: A mention of a Github repository from KellerJordan reveals the use of the muon optimizer, sparking curiosity about its capabilities compared to existing methods.
   - An older academic paper on nanogpt was referenced, suggesting a competitive landscape around novel optimizers and their evaluations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.macrumors.com/2024/12/03/apple-amazon-ai-chips-search/">Apple Uses Amazon's Custom AI Chips for Search Services</a>: Apple uses custom Inferentia and Graviton artificial intelligence chips from Amazon Web Services for search services, Apple machine learning and AI...</li><li><a href="https://news.ycombinator.com/item?id=39876444">JAX is used by almost every large genAI player (Anthropic, Cohere, DeepMind, Mid... | Hacker News</a>: no description found</li><li><a href="https://github.com/stanford-cs149/asst4-trainium">GitHub - stanford-cs149/asst4-trainium</a>: Contribute to stanford-cs149/asst4-trainium development by creating an account on GitHub.</li><li><a href="https://github.com/apple/axlearn">GitHub - apple/axlearn: An Extensible Deep Learning Library</a>: An Extensible Deep Learning Library. Contribute to apple/axlearn development by creating an account on GitHub.</li><li><a href="https://github.com/KellerJordan/modded-nanogpt">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 5 minutes</a>: NanoGPT (124M) in 5 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1313556962859941970)** (114 messages🔥🔥): 

> `Gradient Synchronization in Large Models, Performance of Second Order Optimizers, Random Number Generators, Flow Matching vs Diffusion Training, Machine Unlearning Literature` 


- **Gradient Sync Not a Major Concern in Large Models**: It's noted that once models surpass the 400 billion parameter mark, syncing gradients becomes less significant, as the bulk of synchronization load is not linked to gradient syncing alone.
   - Reducing optimizer state by **4 bytes** is emphasized as a meaningful improvement, particularly for distributed training efforts.
- **Debate on Second Order Optimizer Effectiveness**: Some members express skepticism about the benefits of second-order optimizers in non-convex optimization, citing mixed results in empirical studies, despite some reports of improved convergence.
   - Others suggest that a second-order optimizer would be more effective with tiny eigenvalues but empirically expect no significant differences in performance.
- **Generating Random Number Generators (RNGs)**: A discussion arises about the feasibility of generating algorithms for RNGs, with suggestions to avoid reinventing established algorithms due to the complexities involved in ensuring randomness quality.
   - It is noted that existing RNGs like **Threefry** and **Philox** are parallel friendly and effective compared to trying to create new ones from scratch.
- **Flow Matching Versus Diffusion Training**: Flow matching has gained attention for its simpler formulation and straighter sampling trajectories, which raises the question of its advantages over diffusion models.
   - Despite differences in formulation, flow matching is shown to be equivalent to diffusion models when applied to Gaussian distributions, allowing for the integration of techniques from both methodologies.
- **Challenges with Machine Unlearning**: There's a lack of confidence in measuring how much fine-tuning can revert a model's performance to its pre-trained state, as most studies use proxies for assessing performance consistency.
   - Members recommend exploring the machine unlearning literature for insights, while admitting that the current approaches might not reliably quantify model behavior after unlearning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://proceedings.mlr.press/v139/wortsman21a.html">Learning Neural Network Subspaces</a>: Recent observations have advanced our understanding of the neural network optimization landscape, revealing the existence of (1) paths of high accuracy containing diverse solutions and (2) wider mi...</li><li><a href="https://diffusionflow.github.io/">Diffusion Meets Flow Matching</a>: no description found</li><li><a href="https://arxiv.org/abs/2006.08381">DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning</a>: Expert problem-solving is driven by powerful languages for thinking about problems and their solutions. Acquiring expertise means learning these languages -- systems of concepts, alongside the skills ...</li><li><a href="https://arxiv.org/abs/2401.14953">Learning Universal Predictors</a>: Meta-learning has emerged as a powerful approach to train neural networks to learn new tasks quickly from limited data. Broad exposure to different tasks leads to versatile representations enabling ge...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1313918808397840394)** (1 messages): 

> `Scaling Law Codebases, Examples of Scaling Experiments` 


- **Inquiry on Scaling Law Resources**: A member started playing with **scaling law** and asked for recommendations on a good **codebase** or example code for scaling experiments.
   - *Thanks a lot* for any help provided in finding resources!
- **Request for Scaling Experiment Examples**: Another user expressed interest in looking for various **examples** of scaling experiments to better understand the concept.
   - They seek guidance from the community to point out helpful documentation or repositories.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

deku7041: https://transformer-circuits.pub/
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1313499025277976607)** (7 messages): 

> `External Loadable Evals, lm-eval-harness, Dataset Visibility and Versioning, Reproducibility Concerns` 


- **Proposing External Loadable Evals**: Thoughts emerged on making evals externally loadable like datasets via [Hugging Face](https://huggingface.co), allowing users to load dataset and eval configurations without changes to *lm-eval-harness*.
   - *Jonabur* emphasized the potential for defining an eval 'format' for better integration.
- **Existing External Load Capability**: Currently, it's somewhat possible to load external evals using [include_path](https://github.com/EleutherAI/lm-evaluation-harness/blob/f49b0377bf559f5558e8cd9ebd1190218c7df2a4/lm_eval/__main__.py#L197), which allows passing a directory with configurations.
   - *Baber_* shared insights on the advantages of this existing capability.
- **Reproducibility versus External Evaluations**: *Baber_* raised concerns about visibility and versioning when using an external repo for evaluations, highlighting challenges to reproducibility.
   - *Jonabur* agreed on the significance of reproducibility in evaluation processes.
- **Dataset Versioning as a Challenge**: Discussion turned to whether versioning and reproducibility could also pose issues for the raw datasets used in existing evals.
   - *Baber_* acknowledged this concern but noted it hasn't been a significant problem yet.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/f49b0377bf559f5558e8cd9ebd1190218c7df2a4/lm_eval/__main__.py#L197),">lm-evaluation-harness/lm_eval/__main__.py at f49b0377bf559f5558e8cd9ebd1190218c7df2a4 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1313790501127979071)** (1 messages): 

> `Mira Virtual AI tools, Multimodal conversions, Consumer-level GPU frameworks` 


- **Introducing Mira Virtual AI tools**: A member showcased their [GitHub project](https://github.com/Mirror-Prismals/Mira-Virtual-Ai) called **Mira Virtual Ai**, which offers utility tools for multimodal conversions and other fundamental tasks designed to run on consumer hardware.
   - These scripts can operate on just **2GB of VRAM**, are self-contained, and aim to provide accessible AI solutions for users with limited coding experience.
- **Focus on ease of use and accessibility**: The member emphasized that their tools are tailored for users who may not have coding skills, making it easier to experiment with AI locally.
   - They expressed their hope to bring **fun and automation** in multimodal tasks to a wider audience.



**Link mentioned**: <a href="https://github.com/Mirror-Prismals/Mira-Virtual-Ai">GitHub - Mirror-Prismals/Mira-Virtual-Ai: Ai Frameworks for Consumer Level GPU&#39;s</a>: Ai Frameworks for Consumer Level GPU&#39;s. Contribute to Mirror-Prismals/Mira-Virtual-Ai development by creating an account on GitHub.

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1313495582525882428)** (2 messages): 

> `Logging Configuration, Optimizer Performance Metrics` 


- **Understanding Logging Output**: A member sought clarification on the origin of specific log messages detailing **optimizer operations and timing metrics**.
   - They noted the messages contained detailed timing information for various optimizer steps including **fwd** and **bwd** operations.
- **Config Option Resolution Revealed**: The member identified that the **'wall_clock_breakdown'** configuration option enabled the detailed logging they were inquiring about.
   - This config option provides insights into the timing breakdown of different operations during training.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 messages): 

OpenAI: -# @everyone 12 Days of OpenAI
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1313502276760899614)** (242 messages🔥🔥): 

> `AI Translation Tools, Quantum Computing in Voting, Cohere AI Features, OpenAI File Processing Issues, Hungarian Translation Accuracy` 


- **Discussion on AI Translation Tools**: Members exchanged opinions on various AI translation tools, with **DeepL** being favored for its accuracy over **Google Translate** and **Microsoft**.
   - Suggestions were made to use **Cohere's API** or open-webui filters to achieve multilingual capabilities with chatbots.
- **Challenges with OpenAI's File Processing**: A user reported issues with **ChatGPT 4o** not processing files and images, prompting discussions about whether this was a common bug.
   - It was advised to ensure the use of the correct model and to consider submitting a bug report regarding the problem.
- **Quantum Computing Insights**: Discussions touched on the application of **quantum computing** in various fields, including potential benefits in voting systems through advanced algorithms.
   - Disagreements arose about the relevance of quantum algorithms in practical voting scenarios, emphasizing that *voters are not in superposition*.
- **Cohere AI's Hungarian Translation Features**: The **Cohere AI** platform was highlighted for having a model that supports over 100 languages, including **Hungarian**, and users shared their experiences.
   - It was noted that despite being a large model, the high accuracy for Hungarian translations makes it a strong choice for users needing multilingual support.
- **OpenAI Future Developments and Ideas**: Conversations also included reflections on OpenAI's direction, with some members suggesting improvements for inference models to enhance reasoning capabilities.
   - The potential of AI-driven tools was explored, including the integration of local models and multilingual support to advance learning and accessibility.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=nUa_r9GKjtI">Mark Johns (@Doomlaser) on Artificial Intelligence, Symbolic Logic, Corporate Fairness &amp; More ∰$❤️🏤.</a>: Follow https://x.com/DoomlaserCORP on Twitter.

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1313649255856209930)** (4 messages): 

> `GPT image reading limitations, LLMs and translation issues, Advanced Voice Mode for Custom GPTs` 


- **GPT No Longer Reads Images**: A member noted that **GPT** cannot read images anymore, raising questions about the impacts of this change.
   - This limitation highlights a shift in capabilities that members are curious to understand.
- **LLMs Struggle with Non-Translatable Strings**: A member humorously pointed out that **LLMs often fail** to identify non-translatable strings in code marked for `i18n`, showcasing their logic limitations.
   - *This gives an interesting insight* into the challenges faced by LLMs in code interpretation.
- **Inquiry About Advanced Voice Mode in Custom GPTs**: One member asked if there are any plans for **Advanced Voice Mode** to be implemented in Custom GPTs.
   - This inquiry reflects ongoing interest in enhancing Custom GPT features to better serve user needs.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1313608463850536981)** (11 messages🔥): 

> `Improving prompt engineering, Baiting models to think deeper, Using markdown for prompts, Research on GPT response time, Model comparison` 


- **Channel as a resource for prompt engineering**: A member suggested that the channel itself is a great starting point for improving prompt engineering for custom GPTs, recommending questions and direct messages for further options.
   - They emphasized the importance of context and attention in prompts.
- **Seeking strategies to deepen model thinking**: A member inquired about prompts to *bait* models like o1 to think more thoroughly before responding, referencing OpenAI research that suggests longer thinking yields better answers.
   - Another member cautioned against using 'bait' as it may alter the model’s interpretation of the prompt.
- **Markdown structure for prompts**: Members discussed using markdown to present prompts in a hierarchical structure, which may help introduce complex considerations and enhance prompt clarity.
   - One participant mentioned that tough questions could serve as a way to prompt deeper thinking in models.
- **Limitations in testing with OpenAI models**: Concerns were raised about the limitations of testing various prompting strategies with OpenAI models, with members noting that understanding 'normal' responses is subjective.
   - This lack of access to unlimited prompting limits experimentation in refining prompts for better responses.
- **Questioning effectiveness of prompts**: A possible strategy shared involved prompting the model to reflect deeply with the suggestion to consider multiple avenues before responding.
   - However, a member noted that determining the actual usefulness of such prompts is another challenge altogether.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1313608463850536981)** (11 messages🔥): 

> `Prompt Engineering, Baiting for Deeper Responses, YAML Prompt Structuring, Model Thinking Time, API Automation Test Cases` 


- **Exploring Prompt Engineering Strategies**: Members discussed ways to improve prompt engineering for building better custom GPTs using OpenAI ChatGPT, highlighting that this channel is a great starting point for inquiries.
   - One suggested using markdown language like YAML to structure prompts hierarchically, emphasizing the importance of context and attention.
- **Baiting GPT for Deeper Thinking**: A user inquired about methods to bait the OpenAI model (o1) to think longer before responding, referencing similar capabilities in other AI models like Deepseek.
   - Another member cautioned that 'baiting' alters the model's interpretation of the prompt and suggested asking tough questions instead.
- **Using Placebo Prompts for Reflection**: A member proposed using prompts like 'Reflect deeply and consider multiple possible avenues' to encourage deeper thought from the AI.
   - However, they acknowledged that the effectiveness of such approaches is difficult to evaluate thoroughly.
- **Uncertainty in Model Behavior**: There was a consensus on the uncertainty regarding how the OpenAI model processes prompts, especially in relation to undefined 'normal' behavior.
   - Members expressed a desire for more extensive testing if unlimited access to the model were available.
- **API Automation as a Testing Ground**: The discussion touched on the idea of using API automation for testing various prompt strategies efficiently.
   - This was identified as a good case to evaluate the nuances of prompting techniques and their outcomes with the model.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1313501394023616572)** (175 messages🔥🔥): 

> `Amazon Bedrock Models, Aider New Features, QwQ Model Performance, User Experience with Aider, Benchmark Results` 


- **Amazon Bedrock Nova Model Introduced**: Amazon announced several new foundation models, including the **Nova** series, available exclusively through Amazon Bedrock, featuring context lengths up to **300K tokens**.
   - Performance on benchmarks is comparable to **Llama 3**, with pricing designed to be competitive for different model capabilities.
- **Aider's New watch-files Feature**: The newly introduced `--watch-files` feature in Aider allows users to interact seamlessly with code through AI comments, triggering actions based on specified markers.
   - Documentation is still being finalized, but early feedback praises the functionality as a significant advancement.
- **QwQ Model underwhelms in Performance**: The **QwQ 32B Preview** model was reported to achieve a score of **54%** for whole edit formats and **50%** for diffs, indicating weaker performance compared to expectations.
   - Users are encouraged to consider using **Qwen** or **Sonnet** models for better results, reflecting concerns about QwQ's practical utility.
- **User Experience and Feedback**: There was some discussion regarding individuals' experiences with Aider, including frustrations with user interactions and platform familiarity.
   - Notably, one user expressed wanting a GUI over CLI, indicating a preference that mirrors sentiments in the community.
- **Development and Improvements Discussion**: There's ongoing dialogue about how to implement various features and improvements in Aider, including better support for the new Nova models.
   - Collaborators shared insights on benchmark results and potential architectural changes linked to adding new model support.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aws.amazon.com/blogs/aws/reduce-costs-and-latency-with-amazon-bedrock-intelligent-prompt-routing-and-prompt-caching-preview/">Reduce costs and latency with Amazon Bedrock Intelligent Prompt Routing and prompt caching (preview) | Amazon Web Services</a>: Route requests and cache frequently used context in prompts to reduce latency and balance performance with cost efficiency.</li><li><a href="https://aider.chat/docs/usage/browser.html">Aider in your browser</a>: Aider can run in your browser, not just on the command line.</li><li><a href="https://x.com/_philschmid/status/1864016010464080260">Tweet from Philipp Schmid (@_philschmid)</a>: Unexpected. @amazon is back with Foundation Models. As part of re:Invent they announced 6 new foundation models from text only to text-to-video! 👀 Nova models will be exclusively available through Am...</li><li><a href="https://aider.chat/2024/12/03/qwq.html">QwQ is a code architect, not an editor</a>: QwQ is reasoning model like o1, and needs to be used as an architect with another model as editor.</li><li><a href="https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/providers?model=amazon.titan-image-generator-v1"">no title found</a>: no description found</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: Intro and tutorial videos made by aider users.</li><li><a href="https://github.com/BerriAI/litellm/releases/tag/v1.53.5">Release v1.53.5 · BerriAI/litellm</a>: What&#39;s ChangedLiteLLM Minor Fixes &amp; Improvements (12/03/2024) by @krrishdholakia in #7008Add prompt caching flag for Azure OpenAI gpt-4o-2024-08-06 by @fengjiajie in #7020fix: Add credential t...</li><li><a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-options.html">Command line options in the AWS CLI - AWS Command Line Interface</a>: no description found</li><li><a href="https://youtube.com/@codingthefuture-jg1he?si=mjqG_DrpgMJcYG8C">Coding the Future With AI</a>: Welcome to Coding the Future With AI! Our channel is dedicated to helping developers and tech enthusiasts learn how to leverage AI to enhance their skills and productivity. Through tutorials, expert i...</li><li><a href="https://aider.chat/docs/config/options.html#--gitignore">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://aider.chat/docs/usage/watch.html">Aider in your IDE</a>: Aider can run in your browser, not just on the command line.</li><li><a href="https://youtu.be/t-i2x3APvGQ?si=pAp8W8-as258a-Sg">Unlock AI Coding with Workflow-Driven, Tuned Prompt Chains 🔑</a>: In this tutorial, we’re diving into a systematic approach to building software with AI, introducing you to a workflow-driven system powered by highly tuned p...</li><li><a href="https://github.com/Aider-AI/aider/issues/2525#issue-2715377909">Please add support for model context protocol from anthropic  · Issue #2525 · Aider-AI/aider</a>: Issue Please add support for model context protocol from anthropic Version and model info latest</li><li><a href="https://github.com/lee88688/aider-composer">GitHub - lee88688/aider-composer: Aider&#39;s VSCode extension, seamlessly integrated into VSCode</a>: Aider&#39;s VSCode extension, seamlessly integrated into VSCode  - GitHub - lee88688/aider-composer: Aider&#39;s VSCode extension, seamlessly integrated into VSCode</li><li><a href="https://github.com/BerriAI/litellm/pull/7019#issuecomment-2518028160">Add Amazon Nova models by iwamot · Pull Request #7019 · BerriAI/litellm</a>: TitleAdd Amazon Nova models.https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.htmlhttps://aws.amazon.com/bedrock/pricing/Relevant issuesType🆕 New FeatureChanges[REQUIRED] T...</li><li><a href="https://github.com/aj47/100x-orchestrator">GitHub - aj47/100x-orchestrator: A web-based orchestration system for managing AI coding agents. The system uses Aider (an AI coding assistant) to handle coding tasks and provides real-time monitoring of agent outputs through a user-friendly interface.</a>: A web-based orchestration system for managing AI coding agents. The system uses Aider (an AI coding assistant) to handle coding tasks and provides real-time monitoring of agent outputs through a us...</li><li><a href="https://github.com/BerriAI/litellm/pull/7008">LiteLLM Minor Fixes &amp; Improvements (12/03/2024) by krrishdholakia · Pull Request #7008 · BerriAI/litellm</a>: fix(key_management_endpoints.py): override metadata field value on updateallow user to override tagsfeat(init.py): expose new disable_end_user_cost_tracking_prometheus_only metricallow disabl...</li><li><a href="https://github.com/chrishayuk/mcp-cli">GitHub - chrishayuk/mcp-cli</a>: Contribute to chrishayuk/mcp-cli development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=9mciRwpcLNY)">Anthropic MCP with Ollama, No Claude? Watch This!</a>: anthropic released model context protocol which allows you to connect llm&#39;s to your own data and tools.  in this video chris shows how to decouple mcp from c...</li><li><a href="https://docs.litellm.ai/docs/providers/bedrock">AWS Bedrock | liteLLM</a>: ALL Bedrock models (Anthropic, Meta, Mistral, Amazon, etc.) are Supported</li><li><a href="https://aider.chat/docs/llms/bedrock.html">Amazon Bedrock</a>: aider is AI pair programming in your terminal</li><li><a href="https://github.com/Aider-AI/aider/issues/713">[FEATURE] Support Amazon Bedrock Claude Sonnet 3.5 · Issue #713 · Aider-AI/aider</a>: Issue I hope it will be available not only through Anthropic but also through Amazon Bedrock. https://aws.amazon.com/blogs/aws/anthropics-claude-3-5-sonnet-model-now-available-in-amazon-bedrock-the...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1313503299693187084)** (67 messages🔥🔥): 

> `Aider Docker Setup, Timeout Issues with Aider, Using Aider with Local Models, Using MCP with Aider, Function Refactoring with Aider` 


- **Aider Docker Setup for Local Models**: A member discussed using Aider in Docker, particularly with volumes shared between dev containers and Aider containers, facing permission issues with files.
   - The setup involves running Aider in a CentOS container while attempting to align user settings but has resulted in a 'Permission denied' error.
- **Handling Timeout Issues with Aider**: A member reported a timeout error while running Aider with a local server using `--timeout 5000`, suggesting it may stem from a litellm bug related to timeout settings.
   - Despite configuring both Aider and the local model with timeouts, the process encounters a connection error, and other users confirmed ongoing issues related to the same timeout settings.
- **Setting Up Model Configurations in Aider**: Members discussed the correct setup for `.aider.model.settings.yml` files, specifically mentioning issues with Aider not recognizing the settings for local models.
   - Clarification was sought regarding the location of these configuration files and how to ensure Aider picks them up successfully.
- **Exploring Function Refactoring with Aider**: A user inquired about using Aider to find all instances of a function in a codebase during refactoring, highlighting limitations in Aider's capability to automate this task.
   - Suggestions included using IDE tools or RAG tools for such tasks, with a recommendation to use shell commands to manually find function occurrences instead.
- **Using Architect Mode in Aider**: Users discussed the function and setup of architect mode within Aider, including uncertainty on setting custom models for this mode.
   - It was confirmed that the model specified via the `--model` argument would determine the architect mode, allowing flexibility in the choice of model to work with.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://aider.chat/docs/install/docker.html">Aider with docker</a>: aider is AI pair programming in your terminal</li><li><a href="https://tenor.com/view/yup-dale-doback-step-brothers-yes-i-agree-gif-1579811350903017250">Yup Dale Doback GIF - Yup Dale doback Step brothers - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/Aider-AI/aider/issues/2209#issuecomme">Feature request: support for llama.cpp · Issue #2209 · Aider-AI/aider</a>: llama.cpp running in server mode, how to use this? any documentation on usage?</li><li><a href="https://github.com/BerriAI/litellm/issues/7001">[Bug]: &quot;timeout&quot; and &quot;stream_timeout&quot; set at the model level in config.yaml do not work · Issue #7001 · BerriAI/litellm</a>: What happened? I am setting both &quot;timeout&quot; and &quot;stream_timeout&quot; in my config.yaml like below. - model_name: &quot;gpt-4o&quot; litellm_params: model: &quot;azure/gpt-4o&quot; api_k...</li><li><a href="https://m.youtube.com/watch?v=tElgVPUargw">AI Coding with Aider Architect, Cursor and AI Agents. (Plans for o1 BASED engineering)</a>: 🔥 The AI CODE Editor WAR is ON! Is Your Coding Workflow Ready for the o1 Release?Don’t Get COOKED and Left Behind! 🚀🔥🔗 Resources- 💻 Computer Use Bash &amp; ...</li><li><a href="https://youtu.be/9mciRwpcLNY?si=IqPQDJ-lgBlYGUre)">Anthropic MCP with Ollama, No Claude? Watch This!</a>: anthropic released model context protocol which allows you to connect llm&#39;s to your own data and tools.  in this video chris shows how to decouple mcp from c...</li><li><a href="https://github.com/Aider-AI/aider/issues/2209#issuecomment-2453597627">Feature request: support for llama.cpp · Issue #2209 · Aider-AI/aider</a>: llama.cpp running in server mode, how to use this? any documentation on usage?
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1313948551742492783)** (3 messages): 

> `MCP adoption, OpenAI's development strategy` 


- **MCP sparks interest**: A member expressed that the **MCP** is the future, stating there is no doubt in their mind about its significance.
   - This sentiment reflects a growing enthusiasm within the community regarding its potential impact.
- **Concerns about OpenAI's direction**: There are hopes that **OpenAI** will adopt MCP rather than *reinventing the wheel* with its development choices.
   - This highlights a desire for innovation while respecting existing advancements in the field.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1313504024993202197)** (119 messages🔥🔥): 

> `Mojo Networking Features, SIMD in Mojo, High-Performance File Server, Extensible Sockets Development, Async Programming Considerations` 


- **Mojo Networking Features Awaiting Language Updates**: A discussion highlighted the ongoing developments in Mojo's networking capabilities, focusing on swappable network backends aimed at achieving **25-40 Gbps of TCP throughput** per core, leveraging advancements in io_uring.
   - Basic networking features are expected post-update to establish efficient APIs catering to modern requirements, as pointed out by multiple members.
- **Exploration of SIMD in Mojo**: Members discussed the potential of using [SIMD](https://github.com/simdjson/simdjson) operations in Mojo, emphasizing its user-friendly implementation compared to C/C++ intrinsics.
   - Darkmatter noted that most SIMD intrinsics should ideally be embedded into the standard library, reducing reliance on direct intrinsic calls.
- **Building a High-Performance File Server**: One member mentioned developing a **high-performance file server** for a game, initially targeting a performance boost of **30% more packets/s** compared to Nginx's 200-byte HTTP header parsing.
   - The conversation included strategies about achieving efficiency and the need for robust network API support.
- **Development of Extensible Sockets Framework**: Discussion surfaced regarding the scaffolding built for [extensible sockets](https://github.com/martinvuyk/forge-tools/tree/main/src/forge_tools/socket), revealing the importance of **API coherence** amidst varying systems like io_uring and POSIX sockets.
   - Darkmatter urged collaboration among developers involved to align on decisions, promoting a **solid networking foundation** in Mojo.
- **Async Programming and Its Challenges**: The intricacies of **async programming** were debated, particularly in handling coroutines and static dispatch, focusing on potential performance pitfalls.
   - Participants emphasized the significance of understanding hardware differences and the need for constant refinement in the approach, with examples of using trait objects for optimal performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/s">s - Overview</a>: s has 49 repositories available. Follow their code on GitHub.</li><li><a href="https://godbolt.org/z/E3381jM43">Compiler Explorer - C (x86-64 clang (trunk))</a>: /* Type your code here, or load an example. */void square(__m128i a, __m128i b, __mmask8* k1, __mmask8* k2) {    _mm_2intersect_epi32(a, b, k1, k2);}</li><li><a href="https://github.com/marti">marti - Overview</a>: GitHub is where marti builds software.</li><li><a href="https://github.com/martinvu">MartinVu - Overview</a>: MartinVu has 5 repositories available. Follow their code on GitHub.</li><li><a href="https://mzaks.medium.com/counting-chars-with-simd-in-mojo-140ee730bd4d">Counting chars with SIMD in Mojo</a>: Mojo is a very young (actually a work in progress) programming language designed and developed by a new company called Modular. Here is a…</li><li><a href="https://github.com/intel/hyperscan">GitHub - intel/hyperscan: High-performance regular expression matching library</a>: High-performance regular expression matching library - intel/hyperscan
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1313507812080619585)** (112 messages🔥🔥): 

> `Inline Reference Concept, Memory Optimization Techniques, Compiler Support for Reference Traits, Bounds Checking for Mojo Lists, Auto-tuning in Compilation Phases` 


- **Inline References Spark Discussion**: The concept of an `InlineReference` type was proposed, allowing for more memory-efficient access patterns without storing addresses, potentially improving performance by enabling contiguous memory reads.
   - Discussion highlighted the balance needed between reference usability and compiler visibility, as well as concerns about the implications of integrating this feature.
- **Memory Optimization Strategies Explored**: A focus on small string and vector optimizations was discussed, emphasizing how these can enhance performance by enabling zero-copy scenarios during large array scans.
   - Community members expressed interest in understanding practical use cases for these optimizations and how they might be implemented effectively.
- **Compiler Traits Gain Traction**: The proposal for `Mutable` and `Addressable` traits ignited debate on their implications for compiler functionality, suggesting these traits could be supported natively while keeping their implementations opaque.
   - This model promises to grant greater freedom in how references are treated while potentially eliminating aliasing concerns during function execution.
- **Bounds Checking Mechanisms Under Review**: There are ongoing discussions concerning the lack of bounds checking on Mojo lists and its impact on safety, with debug checking currently in place for out-of-bounds access notifications.
   - Future developments may include improved bounds checking as the language matures, depending on user feedback and implemented features.
- **Auto-Tuning in Compilation Phases Considered**: Concerns were raised about whether the revised compilation structure allows for auto-tuning capabilities, hinting at previous features that may be reintegrated.
   - The need for specialized support in compilation phases is emphasized to improve performance and adaptability in future releases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/faq#distribution">MAX FAQ | Modular Docs</a>: Answers to questions we expect about MAX Engine.</li><li><a href="https://github.com/ParkMyCar/compact_str">GitHub - ParkMyCar/compact_str: A memory efficient string type that can store up to 24* bytes on the stack</a>: A memory efficient string type that can store up to 24* bytes on the stack - ParkMyCar/compact_str
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1313511899119358013)** (124 messages🔥🔥): 

> `Dynamic 4-bit Quantization, Training Qwen Models, Using Colab for Fine-tuning, Model Performance Issues, SGLang Opinions` 


- **Unsloth releases Dynamic 4-bit Quantization**: Unsloth announced their new **Dynamic 4-bit Quantization** aimed at improving model accuracy while using less VRAM than traditional 4-bit methods.
   - *Naive quantization* can hurt model accuracy, but their approach dynamically opts out of quantizing some parameters.
- **Issues with Qwen Model Fine-tuning**: Users reported that **Qwen 2 VL 7B finetunes** often ignore training data unless specific parameters like *repetition penalty* and *temperature* are adjusted.
   - The performance issues seem more pronounced in **Qwen** and **Pixtral models**, leading to bad results during training.
- **Using Colab for Fine-tuning with Large Datasets**: One user discussed using a **304k conversation dataset** on Colab A100 for fine-tuning *unsloth/Llama-3.2-1B-Instruct*.
   - Concerns were raised about optimizing training parameters as training on Colab can be expensive, especially for users in **LATAM**.
- **Feedback on SGLang**: User @bharatdeep04myfi_35111 inquired about experiences with **SGLang**, receiving feedback that it works but is slower compared to **VLLM**.
   - The general consensus suggested that while SGLang is functional, users might favor VLLM for better performance.
- **Dynamic 4-bit Mode Activation for Models**: It was clarified that to utilize the **Dynamic 4-bit mode**, users must change the model name to end with 'unsloth-bnb-4bit'.
   - This adjustment is essential to enable improved performance without needing to manually enable the feature.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh">no title found</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1864380913922265300">Tweet from Unsloth AI (@UnslothAI)</a>: We’re excited to introduce Unsloth Dynamic 4-bit Quantization!Naive quantization often hurts accuracy, making models unusable, but we dynamically opt not to quantize certain parameters. Our approach d...</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h6ojwr/quantizing_to_4bits_can_break_models_dynamic/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>: Hackable and optimized Transformers building blocks, supporting a composable construction. - facebookresearch/xformers</li><li><a href="https://www.youtube.com/watch?v=pwGzyh3IiLU">Sports! | Tim and Eric Awesome Show, Great Job!  Adult Swim DE</a>: Pünktlich zum Super Bowl führen euch Tim und Eric in die Welt des Sports ein...Ständig neue Videos gefällig? Abonniere den YouTube-Kanal von [adult swim] Deu...</li><li><a href="https://www.reddit.com/r/unsloth/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: Scripts for text classification with llama and bert - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: A course on aligning smol models.</a>: A course on aligning smol models. Contribute to huggingface/smol-course development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1313512234202431652)** (23 messages🔥): 

> `Citation Formats, Continued Pretraining, Model Comparisons, Reddit Communities` 


- **Clarifying Citation Formats**: A member asked how to cite Daniel Han and Michael Han properly, and another provided a sample citation format along with a [GitHub repository link](https://github.com/unslothai/unsloth).
   - The conversation included suggestions to add LaTeX/BibTeX citation code on the repo for ease of reference.
- **Importance of Continued Pretraining**: Discussion emphasized that **Continued Pretraining (CPT)** is essential for models like **Llama-3** to adapt to new domains and learn new tokens effectively.
   - Members noted that many base models are pretrained on large datasets but still require CPT in specific fields like law and medicine.
- **Model Comparison Sparks Debate**: The comparison between **Claude** and **CodeLlama** highlighted that CodeLlama is considered outdated, with members suggesting alternatives like **Qwen2.5-coder**.
   - Insights were shared that **Qwen2.5-coder** yields results similar to Claude, indicating its relevance in current discussions.
- **Reddit Communities Decline**: A member shared their frustration about the decline of Reddit communities, having previously been active in 50 subreddits, now reduced to a few that feel like 'graveyards'.
   - They noted that specific subreddits such as **localLlama** have become increasingly negative following a debacle, leading to dwindling engagement.
- **Understanding Model Limitations**: A question was raised regarding the implications of using trained dataset examples directly, tied to concepts of **temperature** and dataset variety.
   - One member remarked on their successful experiences with **wizardlmMath**, while expressing concerns toward models corroborating limited examples without creativity.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1313513970057084949)** (35 messages🔥): 

> `Fine-tuning Llama 3, GGUF conversion issues, ReadTimeout error in Google Colab, Using multiple GPUs with Unsloth, Adapter configuration errors in training` 


- **Fine-tuning Llama 3 faces issues**: Users are encountering problems with fine-tuning Llama 3, including runtime errors when saving models to GGUF due to missing files in llama.cpp.
   - Additionally, some users noted that switching to different notebook versions did not resolve these issues, and they are awaiting updates.
- **Conversion to GGUF methods**: Amid GGUF conversion challenges, users discussed potential solutions and alternative methods, with some suggesting use of different Colab setups.
   - Participants shared links and resources for proper conversion methods, noting that the only current options involve using the Unsloth framework.
- **Addressing ReadTimeout errors**: Several users are facing ReadTimeout errors when attempting to load models in Google Colab, indicating potential connectivity issues.
   - Others suggested that rebuilding Docker or checking internet access in containers may resolve these timeout problems.
- **Engagement with multiple GPUs**: Discussion around Unsloth hiding multiple GPUs raised questions about utilizing them simultaneously for other tasks while fine-tuning a model.
   - A possible pull request was mentioned to address this limitation, although it is still under review.
- **Adapter configuration errors during fine-tuning**: Some users experienced errors related to adapter configurations when trying to finetune models, specifically 'Requested bias: none' not being implemented.
   - Alternatives like adjusting the bias setting yielded similar errors, leading users to seek guidance for resolving these configuration issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/12hkbsOMJfYfmqJLA93cV5tGoPIeZ5gDK#scrollTo=oAC_WYSUX7k_">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit">unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/phi3?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-4285.">Finetune Phi-3 with Unsloth</a>: Fine-tune Microsoft&#x27;s new model Phi 3 medium, small &amp; mini easily with 6x longer context lengths via Unsloth!</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct/blob/main/config.json">config.json · unsloth/Phi-3.5-mini-instruct at main</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1313909574209634334)** (1 messages): 

> `Fimbulvntr's article` 


- **Fimbulvntr Debuts with New Article**: Fimbulvntr just published their first article on [X](https://x.com/fimbulvntr/status/1864350663204852054), showcasing an exciting perspective on a relevant topic.
   - The article can be directly accessed through this [link](http://x.com/i/article/1864344035466637312) for those interested in exploring further.
- **Fimbulvntr's Insights on New Trends**: In the article, Fimbulvntr discusses emerging trends in the tech landscape, emphasizing the importance of adaptability.
   - Readers are encouraged to engage with the content and provide their thoughts in the comments.



**Link mentioned**: <a href="https://x.com/fimbulvntr/status/1864350663204852054">Tweet from Fimbul (@fimbulvntr)</a>: http://x.com/i/article/1864344035466637312

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

edd0302: https://x.com/ruliad_ai/status/1864394941029322890
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1313503386049450047)** (156 messages🔥🔥): 

> `Amazon Nova Models, User Interface Issues, Perplexity AI Performance, Pro Subscription Concerns, Model Availability and Extensions` 


- **Amazon Nova Launch Impresses Users**: Users discussed the new **Amazon Nova** foundation models, noting their **speed** and **accuracy**, with eager anticipation for use in **Perplexity Pro**.
   - Early experimentation yielded positive feedback, as users highlighted the models' potential for high performance in AI-driven tasks.
- **Interface Complaints on Mac App**: Many users reported dissatisfaction with the **Mac app**, citing problems such as **slow performance** and an **awkward interface** compared to the web version.
   - Concerns about battery drain were also raised, prompting discussions about future fixes.
- **Pro Subscription Confusion**: Several users expressed frustration over subscription costs and inconsistencies, particularly regarding the **$4.99 first month** pricing turning into higher charges.
   - Users wondered about the financial model supporting students' free access, leading to a broader discussion about API access and pro features.
- **Issues with Model Access and Changes**: Concerns were raised about limited access to certain models like **O1-mini**, with users questioning whether these restrictions are tied to subscription levels or overall service changes.
   - Users also discussed confusion surrounding the **Complexity extension**, its legitimacy, and its inability to add new models to their interface.
- **Language and Response Quality**: Some users experienced unexpected language outputs from the AI, particularly with responses appearing in **Chinese** or other errors related to language preferences.
   - Discussions included tips on adjusting settings for response languages as well as recommendations for switching between models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sse-next-one.vercel.app/">Server Sent Events</a>: no description found</li><li><a href="https://tenor.com/view/men-i-trust-emma-poulx-show-me-how-you-deserve-this-gif-25757415">Men I Trust Emma Poulx GIF - Men I Trust Emma Poulx Show Me How - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws">Introducing Amazon Nova, our new generation of foundation models</a>: New state-of-the-art foundation models from Amazon deliver frontier intelligence and industry-leading price performance.</li><li><a href="https://tenor.com/view/bella-ketchup-swan-twilight-edward-gif-18684497">Bella Ketchup GIF - Bella Ketchup Swan - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/spelling-gif-9068510">Spelling GIF - Spelling - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=APO7WHP8Ozw">Real-time AI search battle: ChatGPT Search vs. Perplexity vs. Google vs. Copilot vs. Grok</a>: AI is taking over search. 🤖Whether you love em or hate em, LLM-powered searches are coming for your devices. ↳ ChatGPT Search and its Chrome extension. ↳ Go...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1313530032865411104)** (3 messages): 

> `Heisenberg Heat, Software Optimization Tools, Perplexity API Functionality` 


- **Heisenberg Heat Inquiry Sparks Interest**: A link was shared discussing the **Heisenberg Heat** concept, inviting exploration into its principles and implications.
   - Members are encouraged to delve deeper into the associated **theoretical inquiries** and **practical applications**.
- **Careit Ranked as Top Software Optimization Tool**: The discussion highlighted that **Careit** has achieved the **#1 Top Rank** in software optimization tools, thanks to the team's hard work.
   - This achievement has generated excitement and appreciation within the community for their effort and results.
- **Understanding Perplexity API Mechanisms**: A resource was shared explaining how the **Perplexity API** operates, detailing its features and capabilities.
   - The explanation aims to clarify its **functionality** and potential use cases for developers in the community.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1313531881316159613)** (8 messages🔥): 

> `API Payment Issues, Enterprise Waitlist, API Quality Complaints, Support Communication, GitHub Discussion Forum` 


- **API Payment Problems Despite Balance**: A member questioned why payments were still deducted from their credit card despite having a balance in the API called, mentioning they haven't received a response after two days of emailing support.
   - There's growing frustration about unresponsive support, with users considering switching providers due to these issues.
- **Enterprise Waitlist Details**: Another member shared that they were informed the waitlist for enterprise access is about a few weeks long, as confirmed by a team member's email communication.
   - This reflects ongoing demand and some backlog in processing enterprise applications.
- **Concerns Over API Quality**: A user raised significant concerns about the **API quality**, claiming it has become unusable for their use cases, which might lead them to change providers.
   - This complaint hints at a broader issue, with multiple users expressing dissatisfaction in recent weeks.
- **Questions About Support Email Effectiveness**: In a discussion about support responsiveness, a member suggested contacting the support email for assistance, highlighting the possible delay due to **enterprise inquiries**.
   - One member speculated on the effectiveness of contacting support given the current dissatisfaction with response times.
- **Accessing GitHub Discussion Forum**: A member pointed to the [GitHub discussion forum](https://github.com/ppl-ai/api-discussion/discussions) as a place to voice complaints about the API, encouraging others to post their issues there.
   - Another member mentioned they had also submitted a discussion topic regarding a malfunctioning system prompt, showing active participation in seeking solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/discussions/discussions">Forum - Perplexity</a>: no description found</li><li><a href="https://github.com/ppl-ai/api-discussion/discussions/80">Web search not being performed · ppl-ai/api-discussion · Discussion #80</a>: I&#39;m submitting a &quot;system&quot; prompt with some examples, and then a &quot;user&quot; prompt similar to below, asking it to search the internet. Up until a week or 2 ago, this was consistentl...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

alexatallah: 20% price cut for Claude 3.5 Haiku!
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1313491688936968252)** (148 messages🔥🔥): 

> `Hermes 405B Free Service Status, Gemini Ultra Access, Amazon Nova Model Discussion, Model Memory Functionality, Custom Provider Keys Beta` 


- **Hermes 405B Free Service Stopped**: The free service for **Hermes 405B** has been removed, likely due to provider decisions rather than OpenRouter actions, leading to disappointment among users.
   - Some users are exploring other options, but the **base 405B model** remains available for free despite the loss.
- **Gemini Ultra's Availability**: There are discussions surrounding **Gemini 1.0 Ultra**, which is rumored to be available but is currently subjected to allowlists for access.
   - Users feel that the rollout and versioning of Google's models lead to confusion, with speculation that Ultra might be discontinued soon.
- **Discussion on Amazon Nova for Creative Writing**: There is curiosity about the effectiveness of the **Amazon Nova** model for creative writing tasks, with users looking for personal experiences.
   - Speculation exists that while Nova is being evaluated, its capabilities compared to others like Runway remain uncertain.
- **Model Memory and Context Extension**: A user inquired about models having memory to retain previous interactions, with suggestions leaning towards self-hosting solutions for context extension.
   - Methods such as summarizing past messages to extend context length were recommended as alternatives.
- **Requesting Early Access to Custom Provider Keys**: Users are wanting access to the custom provider keys feature, which is currently in beta and might incur fees in the future.
   - To request early access, users are directed to a specific Discord channel for further information.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developers.cloudflare.com/ai-gateway/">AI Gateway · Cloudflare AI Gateway docs</a>: Cloudflare's AI Gateway allows you to gain visibility and control over your AI apps. By connecting your apps to AI Gateway, you can gather insights on how people are using your application with analyt...</li><li><a href="https://developers.cloudflare.com/ai-gateway/providers/open-router/">OpenRouter · Cloudflare AI Gateway docs</a>: OpenRouter ↗ is a platform that provides a unified interface for accessing and using large language models (LLMs).</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://lambdalabs.com/blog/unveiling-hermes-3-the-first-fine-tuned-llama-3.1-405b-model-is-on-lambdas-cloud">Unveiling Hermes 3: The First Full-Parameter Fine-Tuned Llama 3.1 405B Model is on Lambda’s Cloud</a>: Introducing Hermes 3 in partnership with Nous Research, the first fine-tune of Meta Llama 3.1 405B model. Train, fine-tune or serve Hermes 3 with Lambda</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview">no title found</a>: no description found</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-405b-instruct:free">Llama 3.1 405B Instruct (free) - API, Providers, Stats</a>: The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.Meta&#x27;s latest cla...</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-sonnet">Claude 3.5 Sonnet - API, Providers, Stats</a>: New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Run Claude 3.5 Sonnet with API</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-1.0-ultra">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=HQ8AUBn-4DY">How To Tell If Your Social Media Addiction Has Gone Too Far</a>: How to tell if your obsession with FarmVille is a major problem: http://www.yourtango.com/201064181/social-media-addiction-are-you-riskPresenting A YourTango...</li><li><a href="https://aws.amazon.com/cn/bedrock/pricing/">Build Generative AI Applications with Foundation Models - Amazon Bedrock Pricing - AWS</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1313640513349357760)** (6 messages): 

> `Custom Key Beta Access` 


- **Community Eager for Custom Key Beta Access**: Several members expressed their desire for access to the **custom key beta** and raised their hands in requests.
   - One member pleaded, *'I would like the custom key beta access as well!'*, while another shared gratitude for the team's efforts regardless of the timeline.
- **Inquiry About Timeline for Key Access**: A member inquired about the estimated timeline for obtaining the **custom keys**, asking if anyone could provide a guess.
   - They acknowledged the uncertainty, stating, *'we totally get it, and thank all of you for all your hard work.'*


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1313497012351668256)** (115 messages🔥🔥): 

> `Distributed Training Run, Forge Reasoning API Beta, Live Memory in LLMs, Genesis of AI World Models, Nous Research Art and Design` 


- **Distributed Training Run Approaching Completion**: A distributed training run is currently underway and will finish in just over a day, with pre-arranged compute partners involved from the start.
   - More details regarding the training run are expected in the near future, and the potential for public involvement has been acknowledged.
- **Launch of Forge Reasoning API Beta**: Nous Research has launched the Forge Reasoning API Beta, aimed at improving inference times for various models and potentially enhancing the capabilities of Hermes 70B.
   - This new development follows community interest in large-scale foundation models and their practical applications.
- **Discussion on Implementing Live Memory in LLMs**: Members explored ideas on implementing live memory within LLM architecture, debating between using function calls or RAG methods for better consistency and performance.
   - There was a consensus that classical approaches could better ground neural networks in reliable ways while achieving style consistency.
- **Innovations in Generative World Models**: The conversation shifted to the creation of generative 'world models,' likened to video games, and how they could incorporate classical software for reliable data manipulation.
   - Participants suggested using hybrid systems to improve output quality by combining neural and traditional methodologies.
- **Artistic Contributions to Nous Research**: Community members expressed interest in the artistic direction of Nous Research, revealing that John Galt serves as their principal designer.
   - The interplay between art and AI system design was humorously noted, reflecting the unique culture within the team.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://distro.nousresearch.com/">Nous DisTrO</a>: Distributed training over the internet</li><li><a href="https://modal.com/pricing">Plan Pricing</a>: Simple, transparent pricing that scales based on the amount of compute you use.</li><li><a href="https://x.com/jparkerholder/status/1864314826891079787">Tweet from Jack Parker-Holder (@jparkerholder)</a>: Introducing 🧞Genie 2 🧞 - our most capable large-scale foundation world model, which can generate a diverse array of consistent worlds, playable for up to a minute. We believe Genie 2 could unlock th...</li><li><a href="https://x.com/NousResearch/status/1856417883934601246">Tweet from Nous Research (@NousResearch)</a>: Today we are launching the Forge Reasoning API Beta, an advancement in inference time scaling that can be applied to any model or a set of models, for a select group of people in our community.https:/...</li><li><a href="https://www.jetson-ai-lab.com/tutorial_llamaspeak.html#function-calling">
   llamaspeak - NVIDIA Jetson AI Lab
  </a>: no description found</li><li><a href="https://x.com/SHL0MS/status/1864371949322829978?t=yDG98l6fCD23fuGjamiC2Q&s=19">Tweet from 𒐪 (@SHL0MS)</a>: hello @s8n 😈God and Satan are now united as @NousResearch models. we will iterate on both in the coming days to refine their dynamic and posting stylesQuoting 𒐪 (@SHL0MS) as many of you have already...</li><li><a href="https://www.youtube.com/watch?v=gzuYdUAPXxw">Elliott Smith - 13 - Independence Day</a>: Town Hall, New York, New YorkSetlistSon of SamHappinessBetween the BarsLARose ParadePretty Mary KAngelesNeedle in the HaySay YesWaltz #2St. Ide&#39;s HeavenEasy ...</li><li><a href="https://github.com/archit-spec/modal-scripts/blob/main/jupyter_training.py#L75">modal-scripts/jupyter_training.py at main · archit-spec/modal-scripts</a>: example modal scripts for training slm/tts models. Contribute to archit-spec/modal-scripts development by creating an account on GitHub.</li><li><a href="https://www.are.na/john-galt/nous-research-john-galt">NOUS RESEARCH / JOHN GALT | Are.na</a>: A sample of my work with Nous Research.</li><li><a href="https://github.com/archit-spec/modal-scripts/tree/main">GitHub - archit-spec/modal-scripts: example modal scripts for training slm/tts models</a>: example modal scripts for training slm/tts models. Contribute to archit-spec/modal-scripts development by creating an account on GitHub.</li><li><a href="https://t.co/5be7RgCUTL">DeMo: Decoupled Momentum Optimization</a>: Training large neural networks typically requires sharing gradients between accelerators through specialized high-speed interconnects. Drawing from the signal processing principles of frequency decomp...</li><li><a href="https://github.com/bloc97/DeMo">GitHub - bloc97/DeMo: DeMo: Decoupled Momentum Optimization</a>: DeMo: Decoupled Momentum Optimization. Contribute to bloc97/DeMo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1313556527017230449)** (5 messages): 

> `Nous Research Interest, Linux from Scratch as a Benchmark, Precision in Voice Agents, Momentum Concept in Residual Stream` 


- **Techno-Socialist's Interest in Nous**: A member expressed a keen interest in **Nous** as a **Techno-Socialist**, highlighting the alignment of their values with the project's goals.
   - This reflects the growing curiosity within the community about the social implications of **AI advancements**.
- **Using Linux from Scratch as Benchmark**: A query was raised regarding the practicality of using the **Linux from Scratch** book as a benchmark for evaluating AI agents.
   - This suggests interest in establishing **concrete metrics** for assessing agent performance in real-world applications.
- **Achieving Precision in Voice Agents**: A member inquired about methods to achieve **precision in voice agents**, particularly for specific use cases such as sales.
   - The discussion pointed towards **fine-tuning** on tailored datasets as a potential approach for enhancing accuracy.
- **Incorporating Momentum in Mathematical Concepts**: One member proposed the idea of integrating the concept of **momentum** into the **residual stream** architecture, questioning its mathematical foundation.
   - This raised an interesting conversation about whether **addition and skip connections** suffice for achieving similar effects.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

jellyberg: https://theaidigest.org/agent
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1313539156831637626)** (5 messages): 

> `DisTro issues, Logical Consistency, DeLorean Reference` 


- **DisTro and Flux Capacitor Conundrum**: A member humorously questioned if DisTro was invented alongside the **flux capacitor**, implying confusion about its functionality.
   - *It refuses to acknowledge that there is a problem here* was a notable sentiment expressed.
- **Consistency in Logic Discussion**: A member remarked, *it's logical and consistent - if nothing else...*, suggesting that some aspects of the conversation held up under scrutiny.
   - This comment followed a prior humorous critique, reflecting a light-hearted yet critical tone.
- **Desire for a DeLorean**: In a playful tone, a member expressed a wish for their very own **DeLorean**, referencing its iconic status.
   - This comment captures a nostalgic whimsy that reflects the enthusiasm shared in the discussion.



**Link mentioned**: <a href="https://hermes.nousresearch.com)">no title found</a>: no description found

  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1313860690754404462)** (1 messages): 

> `NotebookLM+Spotify, Spotify Wrapped AI Podcast` 


- **NotebookLM partners with Spotify**: NotebookLM and **Spotify** have teamed up to create a personalized **AI podcast** summarizing your year in audio, announced on [December 4, 2024](https://blog.google/technology/google-labs/notebooklm-spotify-wrapped/).
   - The **Spotify Wrapped AI podcast** offers a dynamic audio recap, utilizing **NotebookLM** to unpack users' favorite tracks and artists.
- **Exciting features of Spotify Wrapped AI podcast**: This year’s **Spotify Wrapped** enhances user experience with AI features, presenting tailored content about listening habits.
   - As listeners engage with the podcast, they are treated to **AI hosts** that explore what defined their year in music.



**Link mentioned**: <a href="https://blog.google/technology/google-labs/notebooklm-spotify-wrapped/">Listen to your first-ever 2024 Spotify Wrapped AI podcast, built with Google&#x27;s NotebookLM</a>: NotebookLM is partnering with Spotify to create a personalized Wrapped AI podcast.

  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1313563337820209193)** (23 messages🔥): 

> `AI audio generation, NotebookLM for sports journalism, Legal content simplification, Multilingual AI discussions, Creative projects using AI` 


- **AI generates hilarious multilingual audio**: An audio clip showcasing AI's ability to speak multiple languages received positive feedback, with one member noting that it sometimes loses its focus but manages to return to proper language at times.
   - Another member inquired whether Polish was included, indicating mixed results with language settings.
- **NotebookLM revolutionizes sports feature stories**: A member highlighted the potential of using NotebookLM for creating nightly pregame and postgame feature stories for professional sports teams, suggesting it could easily scale across teams.
   - They emphasized the simplicity of generating content while branding avatars, which could enhance fan engagement.
- **Legal content made easy with NotebookLM**: Another member praised NotebookLM for its effectiveness in parsing complex legal jargon, making legal content more accessible for average users, especially concerning data laws across states.
   - This was cited as a daily tool for simplifying legal information.
- **Unique creative projects using AI**: One user shared a parody panel discussion in German created by AI, exploring philosophical themes such as the meaning of life, showcasing the humorous capabilities of AI-generated content.
   - Members were intrigued by the potential of AI in producing engaging and entertaining dialogue.
- **Generating audio from export chat logs**: A user expressed excitement over generating audio from chat log exports discussing Thai food deals, and acknowledged the effective audio features integrated into NotebookLM.
   - There were mentions of permissions required to share this content publicly, highlighting collaborative community aspects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.fxguide.com/fxpodcasts/zap-andersson-exploring-the-intersection-of-ai-and-rendering/">Zap Andersson: Exploring the Intersection of AI and Rendering</a>: Zap Andersson tests AI Video tools and shares his tips and tricks from his bizarre YouTube series: UNREAL MYSTERIES</li><li><a href="https://notebooklm.google.com/notebook/50b3f4f0-7701-4242-a705-1bf9fd7a0c35?_gl=1*1loyke6*_ga*MTgzODEzOTkwNS4xNzMxNzQ5NjYx*_ga_W0LDH41ZCB*MTczMzMyMDIyMC4yLjEuMTczMzMyMDIyMC42MC4wLjA.&original_referer=https:%2F%2Fnotebooklm.google%23&pli=1">no title found</a>: no description found</li><li><a href="https://notebooklm.google.com/notebook/50b3f4f0-7701-4242-a705-1bf9fd7a0c35/audio">no title found</a>: no description found</li><li><a href="https://youtu.be/D7qZ2VphetU">NBA CUP POC</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1313490556407648317)** (92 messages🔥🔥): 

> `Notebook LM Language Settings, Notebook LM PDF Capabilities, Notebook LM Features Requests, Google Job Listings, Notebook LM Podcast Integration` 


- **Challenges with Language Settings in Notebook LM**: Users are struggling with changing language settings in Notebook LM, particularly for podcasts. One user mentioned that although they set their Google account to Indonesian, it did not change the language of the podcast content.
   - Another user pointed out confusion and disappointment when trying to generate audio content in Portuguese after uploading a script.
- **Concerns About PDF Reading Capabilities**: Questions arose about Notebook LM's ability to read lengthy PDFs and extract relevant information accurately, especially compared to other AI models. Users expressed frustration over receiving incomplete access to documents and summary updates.
   - One individual specifically noted that after uploading two PDFs, the summary only reflected the first document, highlighting a need for better refresh options.
- **Feature Requests and User Experience Enhancements**: Users have requested features like the ability to categorize notebooks and generate transcripts for podcasts, which could align with enterprise policies. Feedback on the current framework indicates a desire for functionalities that allow manual edits to maintain compliance.
   - Another request centered around saving commonly used question templates for ease of use across different notebooks, underscoring a user's need for efficient study tools.
- **Google Job Opportunities Shared**: A Google employee shared links to open positions at Google, providing insights into qualifications required for software engineering roles. The positions discussed have extensive experience requirements, suggesting a strong focus on technical expertise.
   - The conversation also humorously touched on the idea of hiring a 'NotebookLM hype guy', illustrating enthusiasm for the product despite not being a technical developer.
- **Expression of Enthusiasm for Notebook LM's Developments**: Users expressed excitement about Notebook LM's integration with Spotify and its impact on personal experiences. Many noted their anticipation for mainstream adoption and the potential of the technology, indicating a vibrant community backing the innovation.
   - Comments showcased a blend of humor and admiration for the product, with mentions of personal experiences that resonated with the audience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.fxguide.com/fxpodcasts/zap-andersson-exploring-the-intersection-of-ai-and-rendering/">Zap Andersson: Exploring the Intersection of AI and Rendering</a>: Zap Andersson tests AI Video tools and shares his tips and tricks from his bizarre YouTube series: UNREAL MYSTERIES</li><li><a href="https://www.google.com/about/careers/applications/jobs/results/137740784886522566-senio">Senior Software Engineer, Full Stack, Labs — Google Careers</a>: no description found</li><li><a href="https://www.google.com/about/careers/applications/jobs/results/137740784886522566-senior-software-engineer-full-stack-labs">Senior Software Engineer, Full Stack, Labs — Google Careers</a>: no description found</li><li><a href="https://www.google.com/about/careers/applications/jobs/results/101552613576581830-software-engineer-iii-full-stack-labs">Software Engineer III, Full Stack, Labs — Google Careers</a>: no description found</li><li><a href="https://youtu.be/wEAeP1Po3EI?feature=shared">🎉🤖 AI Clown Bot Unmasked! 🤖🎉</a>: 💥 World Takeover... With Clownery? 💥🚨 Watch the first-ever podcast by an AI Clown Bot 🚨This isn’t your typical clown act. Imagine:🎭 AI-generated visuals...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1313894573990481991)** (3 messages): 

> `NeurIPS Meetup, Interconnects Open Hangouts` 


- **Colleagues plan to meet at NeurIPS**: A member expressed excitement about attending NeurIPS and hopes to join an **Interconnects meetup**.
   - They mentioned they will be attending with a colleague.
- **Nat's Open Hangout Plans**: Nat indicated they will propose a few times later in the week for **open hangouts** during NeurIPS.
   - Nat stated they will provide the details via email next Wednesday.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1313591150568931328)** (26 messages🔥): 

> `Amazon Foundation Models, Genie 2, 12 Days of OpenAI, ChatGPT interface updates, Anduril and OpenAI partnership` 


- **Amazon Unveils New Foundation Models**: Amazon launched **6 new foundation models** at re:Invent, including **Micro**, **Lite**, **Pro**, and **Premier**, with capabilities ranging from text to text-to-video generation, available exclusively through [Amazon Bedrock](https://link.to.amazonbedrock).
   - *Performance benchmarks match Llama 3*, and with **up to 300K tokens**, Amazon aims to offer diverse solutions for developers.
- **Introducing Genie 2 for Embodied Agents**: [Genie 2](https://fxtwitter.com/jparkerholder/status/1864314826891079787) promises to generate varied, consistent worlds for **up to one minute**, enhancing capabilities for embodied agents.
   - Members are excited about *the potential it holds for future AI innovations*.
- **OpenAI's 12 Days of Live Streams Kick Off**: '**12 Days of OpenAI**' begins, featuring **12 livestreams** showcasing various announcements, with **a press release about new hires* on Day 1.
   - Members speculate about potential *interface changes, new plans, and updates* directly linked to this event.
- **Possible Updates in ChatGPT Interface**: Members discuss *potential features like a 'pro plan', updated voices*, and new image generation capabilities possibly linked to the upcoming updates in ChatGPT.
   - The excitement is palpable as speculation continues about *functional enhancements and the unveiling of Sora API*.
- **Anduril Partners with OpenAI**: A partnership between **Anduril** and OpenAI aims to advance **U.S. artificial intelligence** leadership with systems powered by Lattice for integrated security across domains.
   - The partnership emphasizes a commitment to *supporting armed forces missions through innovative technologies*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/jparkerholder/status/1864314826891079787">Tweet from Jack Parker-Holder (@jparkerholder)</a>: Introducing 🧞Genie 2 🧞 - our most capable large-scale foundation world model, which can generate a diverse array of consistent worlds, playable for up to a minute. We believe Genie 2 could unlock th...</li><li><a href="https://x.com/_philschmid/status/1864016010464080260">Tweet from Philipp Schmid (@_philschmid)</a>: Unexpected. @amazon is back with Foundation Models. As part of re:Invent they announced 6 new foundation models from text only to text-to-video! 👀 Nova models will be exclusively available through Am...</li><li><a href="https://www.anduril.com/article/anduril-partners-with-openai-to-advance-u-s-artificial-intelligence-leadership-and-protect-u-s/">Anduril Partners with OpenAI to Advance U.S. Artificial Intelligence Leadership and Protect U.S. and Allied Forces</a>: Anduril Industries, a defense technology company, and OpenAI, the maker of ChatGPT and frontier AI models such as GPT 4o and OpenAI o1, are proud to announce a strategic partnership to develop and res...</li><li><a href="https://x.com/OpenAI/status/1864328928267259941">Tweet from OpenAI (@OpenAI)</a>: 12 days.12 livestreams.A bunch of new things, big and small.12 Days of OpenAI starts tomorrow.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1313612647287226509)** (18 messages🔥): 

> `Mistral Large performance, OpenAI office in Zürich, Giffmana ethics debate` 


- **Mistral Large shines in CLI tests**: A member praised **Mistral Large 2**, stating it outperforms both **3.5 Sonnet** and **GPT-4** in handling bash scripts and queries, supported by a [tweet](https://x.com/TheXeophon/status/1833921199170355480) that stated it knows the shell inside out.
   - Another user humorously noted that with AI and an online bash interpreter, remembering **ffmpeg flags** is no longer necessary.
- **Speculation around OpenAI's new Zurich office**: Discussions hinted that OpenAI would establish an office in **Zürich** for a notable figure and his associates, raising questions about their recent social media posts regarding ethics in ML.
   - Concerns were voiced about the departure of **GDM** personnel from their comfortable **TPU** setups for new opportunities.
- **Ethics in AI under scrutiny**: Users speculated about the ethical implications of a figure's recent posts, linking it to his transition to OpenAI and suggesting deeper motivations for his online discussions.
   - One member pointed out that the shift to GPUs for 'home experiments' explained his prior activities and motivations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1833921199170355480">Tweet from Xeophon (@TheXeophon)</a>: btw: The best model for bash is -and I kid you not- Mistral Large 2. It outperforms 3.5 Sonnet, GPT-4 whenever I tested it, whether it is scripts or general questions. The latter often try weird thing...</li><li><a href="https://x.com/_xjdr/status/1833921835320443002">Tweet from xjdr (@_xjdr)</a>: @TheXeophon lol thats the model that is powering this interaction (with deepseek coder as the backup)
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1313578112864354304)** (13 messages🔥): 

> `Amazon's New Foundation Models, Concerns about NVIDIA's SANA Licensing, IFEval Benchmark Saturation` 


- **Amazon's New Foundation Models Are Here**: Amazon has announced **6 new foundation models** during re:Invent, including **Nova Micro** (text-only) and **Reel** (video-generation) models available exclusively through Amazon Bedrock.
   - The models will support **up to 300K tokens** and **200+ languages**, with pricing details like **$0.035** for Micro models.
- **NVIDIA's SANA License Sparks Outrage**: The **SANA model** from NVIDIA is fast but its **license restricts usage** to non-commercial applications and only on NVIDIA GPUs, which many find unreasonable.
   - Concerns were raised about its enforcements, such as limitations preventing running on AMD machines and the company retaining rights to generated outputs.
- **Discussion on IFEval Benchmark's Relevance**: Members questioned whether the **IFEval benchmark** is still relevant or if it has become saturated, with many achieving high scores easily.
   - Comments indicated a perception that **90% benchmarking** is becoming commonplace, leading to discussions about what the new meta benchmark might be.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_philschmid/status/1864016010464080260">Tweet from Philipp Schmid (@_philschmid)</a>: Unexpected. @amazon is back with Foundation Models. As part of re:Invent they announced 6 new foundation models from text only to text-to-video! 👀 Nova models will be exclusively available through Am...</li><li><a href="https://fxtwitter.com/cloneofsimo/status/1864309440356470894">Tweet from Simo Ryu (@cloneofsimo)</a>: SANA from nvidia is fast and quite good, but its license is pure tragic.Its for noncommercial use only (sure) but for some fucked up reason you can only run it on NVIDIA-gpusLike how the fuck is that ...</li><li><a href="https://x.com/cloneofsimo/status/1864312857674043599">Tweet from Simo Ryu (@cloneofsimo)</a>: * forgets to put http://SANATransformerModel.to(&#34;cuda&#34;)* model runs on intel CPUnvidia:Quoting 青龍聖者 (@bdsqlsz) @cloneofsimo device=cpu ×deivce=cuda √
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1313559319685632040)** (32 messages🔥): 

> `Reward Function Design, Challenges with Stabilization, Experimentation Procedures` 


- **Struggles with Stabilization Challenges**: The conversation highlighted the complexities of starting in the air vs taking off, with members sharing that *starting in the air is much harder* and *taking off might help avoid quirks*.
   - One member suggested visualizing RL rollouts to determine if the RL method was exploiting the simulator effectively.
- **Simplified Reward Functions Win**: One member proposed that a basic reward function, like *minimizing yaw, pitch, and roll*, could simplify the learning process, suggesting a big negative reward if the simulator is dead.
   - Discussion included a lighthearted mention of ensuring the RL system doesn't just *fly off into the beyond*.
- **Invaluable Experiment Logging Tips**: Experimentation and logging results were emphasized, with a quote on the importance of actually logging experiments instead of relying on memory.
   - Members shared their reflections on previous experiments, indicating a gap in proper documentation for future reference.
- **Intuitive Nature of Reward Functions**: The intuition behind designing reward functions was discussed, with the consensus that *it's mostly intuitive* and that simpler approaches typically yield better results.
   - A member inquired about resources for reward function design, emphasizing the need for a clear understanding of the subject over time.
- **Unique Behavior Observed with Wheelies**: One member observed an interesting behavior in their simulation where the model would perform *wheelies*, lifting one side while failing to raise the back half, leading to flips.
   - This *quirky behavior* emphasized the importance of correct reward functions and the need for adjustments to achieve stable outcomes.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1313853849735528459)** (17 messages🔥): 

> `OLMo1 Naming Controversy, Discussion on Naming Trends, Nerdsniped Reactions` 


- **OLMo1 Naming Controversy**: A user expressed their dislike for having **O1** in the name, calling it **cringe**.
   - *We like O1Mo lol* sparked some laughter and debate around the naming conventions.
- **Debate on Alternative Naming Routes**: A discussion unfolded about possibly going the **qwen route** when considering new names.
   - A member suggested the name **OwOlmo**, continuing the playful naming debate.
- **OLMoFans as a Proposed Name**: Another member mentioned that they've discussed using **OlmoFans** as a potential name.
   - Laughter continued as they acknowledged the humor in the situation with **O1 blog posts still having the juice**.
- **Community Reactions to Naming Ideas**: Members remarked on the **nerdsniped** reactions people have towards the naming topic.
   - That discussion reflected the amusing nature of their fandom around the subject.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1313683518982983700)** (17 messages🔥): 

> `Efficient Gram Matrix Computation, Triton for Upper Triangle, cuBLAS and cutlass for Gram matrices, HPC Interview Expectations` 


- **Exploring Efficient Gram Matrix Computation**: A user raised the question of how to compute the upper triangle of a Gram matrix (**A@A^T**) efficiently without using a standard matmul followed by a triu.
   - Responses suggested leveraging Triton to compute only the relevant tiles and **cuBLAS's syrk** and **cutlass** as potential alternatives.
- **Triton for Custom Kernels**: Discussion on whether writing a custom kernel in Triton would be difficult for someone without prior experience arose, with members indicating that studying the matmul kernel would be crucial.
   - Community members suggested that modifications to support the upper triangle computation could be straightforward after understanding matmul.
- **Resources for Matmul Kernel in Triton**: Members shared resources to help speedrun learning the matmul kernel in Triton, including an [official tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
   - However, it was noted that the resources might not be beginner-friendly, which could pose challenges for new learners.
- **Understanding Gram Matrices**: A distinction was made between different definitions of the Gram matrix, with the member confirming they were interested in **A@A^T** specifically.
   - Another member pointed out that different forms exist in literature, indicating some confusion around the terminology.
- **Interview Expectations for GPU Programming Role**: A user inquired about what to expect in interviews for a GPU programming role on an HPC/Storage team, especially given their background in leetcode-style interviews.
   - Concerns were raised regarding the differences between a general interview and those specific to GPU programming roles, particularly for a new graduate applying for a mid-level position.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/syrkx/cublas_syrkx_example.cu">CUDALibrarySamples/cuBLAS/Level-3/syrkx/cublas_syrkx_example.cu at master · NVIDIA/CUDALibrarySamples</a>: CUDA Library Samples. Contribute to NVIDIA/CUDALibrarySamples development by creating an account on GitHub.</li><li><a href="https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/02%20Kernels/02%20matmul.cu#L42).">cuda-course/05_Writing_your_First_Kernels/02 Kernels/02 matmul.cu at master · Infatoshi/cuda-course</a>: Contribute to Infatoshi/cuda-course development by creating an account on GitHub.</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/examples/31_basic_syrk/basic_syrk.cu">cutlass/examples/31_basic_syrk/basic_syrk.cu at main · NVIDIA/cutlass</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1313499429835116595)** (8 messages🔥): 

> `Triton MLIR Dialects Documentation, Grouped GEMM with TMA, Support Channels for Triton, Kernel Crashes Related to Stages, Triton Gist Issues` 


- **Availability of Triton MLIR Dialects Documentation**: Users discussed the availability of documentation for Triton MLIR Dialects, pointing to the [Triton Ops documentation](https://triton-lang.org/main/dialects/TritonOps.html) as a resource.
   - A minimal [programming guide](https://github.com/triton-lang/triton/tree/main/docs/programming-guide) was also noted, although it appears unfinished.
- **Challenges in Grouped GEMM with TMA**: A user inquired about writing a Grouped GEMM with TMA in Triton, specifically regarding passing tensors of descriptors instead of addresses.
   - There was mention of a [pull request](https://github.com/triton-lang/triton/pull/4498) that aims to address this limitation but may not fully support the desired functionality.
- **Seeking Support for Triton Skill Issues**: Members discussed where to seek support for Triton issues that might be skill-related rather than bugs, suggesting this channel as a resource.
   - The Triton Slack was recommended for serious bugs.
- **Kernel Crashes and Shared Memory Issues**: A user shared a gist showing two versions of a kernel, noting that the loop version crashes with an error.
   - Another member highlighted that this crashing issue is related to the number of stages, observing a successful run only with `num_stages=1`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/dialects/dialects.html">Triton MLIR Dialects and Ops &mdash; Triton  documentation</a>: no description found</li><li><a href="https://triton-lang.org/main/dialects/TritonOps.html">TritonOps &mdash; Triton  documentation</a>: no description found</li><li><a href="https://github.com/triton-lang/triton/tree/main/docs/programming-guide">triton/docs/programming-guide at main · triton-lang/triton</a>: Development repository for the Triton language and compiler - triton-lang/triton</li><li><a href="https://github.com/triton-lang/triton/pull/4498">[nvidia] Support passing TMA descriptors by-value by embg · Pull Request #4498 · triton-lang/triton</a>: MotivationCurrently, Triton passes TMA descriptors by-ref through global memory. This has a number of problems:Significant launch overhead (5-10us) for the host-to-device memcpyUsers must inser...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1313692632584814695)** (5 messages): 

> `Hiring for Code Work, GEMM Kernel Performance, Cache Behavior in GPU Computing` 


- **Hiring for Code Work Discussion**: A user expressed interest in hiring someone for code work and was advised to post in a specific channel for better responses.
   - This suggests a community-focused environment where users can seek help or hire assistance effectively.
- **Performance Hurdles in GEMM Kernel Optimizations**: A user reported a significant performance hit when issuing `cp.async.cg` instructions compared to `cp.async.ca` in a GEMM kernel on an A100.
   - They noted that while using **L1 cache**, they did not encounter these issues, pointing to the complexities of cache behaviors in GPU operations.
- **Layout Changes Improve Memory Access**: The same user solved their initial performance issue by switching from a specific layout during memory access, which helped eliminate bank conflicts.
   - They indicated that this layout adjustment leveraged **swizzling**, ensuring optimized access patterns despite their initial intuition being incorrect.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1313524615611682836)** (30 messages🔥): 

> `Efficient ML courses, Stanford's CS 229S course, CUDA vs Triton, MIT Han Lab course, Washington's CSE 599K course` 


- **Discussion on Efficient ML Courses**: Members highlighted several **efficient machine learning courses**, including MIT's course on [Efficient AI Computing](https://hanlab.mit.edu/courses/2024-fall-65940) and Stanford's [CS 229S - Systems for Machine Learning](https://cs229s.stanford.edu/fall2023/). The MIT course covers **model compression**, **pruning**, and optimization for **resource-constrained devices**.
   - Participants also noted the need for practical implementation resources, with some finding certain courses more theoretical than applied.
- **Course Resources and Assignments**: Several members discussed the **availability of assignments** for various courses, noting that Stanford's CS 229S has **labs available** through Google Colab for ease of use. Additionally, Washington's course [CSE 599K](https://courses.cs.washington.edu/courses/cse599k/24au/) provides an **in-depth understanding** of ML systems with various assignments.
   - Members encouraged checking prerequisite knowledge and resources to fully benefit from these learning opportunities.
- **CUDA Familiarity Before Triton Usage**: A member questioned whether familiarity with **CUDA** is recommended before diving into **Triton**, expressing a preference for the **intuitiveness** of CUDA for writing kernels. Another member shared the perspective that focusing deeply on one language or framework is more beneficial than the choice of framework.
   - The exchange emphasized the balance between understanding low-level kernel development and optimizing one's skills across different platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cs229s.stanford.edu/fall2023/">Home</a>: Systems for Machine Learning</li><li><a href="https://courses.cs.washington.edu/courses/cse599k/24au/">CSE 599K</a>: no description found</li><li><a href="https://hanlab.mit.edu/courses/2024-fall-65940">MIT 6.5940 Fall 2024 TinyML and Efficient Deep Learning Computing</a>: no description found</li><li><a href="https://hanlab.mit.edu">MIT HAN Lab</a>: Welcome to MIT HAN Lab, where efficiency meets performance, innovation converges with excellence in the realm of artificial intelligence (AI) and computer architecture. Our lab stands at the forefront...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1313786282421391420)** (7 messages): 

> `CUDA Prerequisites, Warps Scheduling Confusion, Core Definition in GPU, Mixed Execution Units` 


- **Beginner's Guide to CUDA Topics**: A new member sought guidance on prerequisites for learning CUDA and the PMPP book, wondering about subjects like **Operating Systems** and **Computer Architecture**.
   - *You can just read the book* was a humorous response suggesting direct engagement with the material.
- **Confusion on Warp Scheduling in A100**: A new user expressed confusion about **warp scheduling** described in the PMPP book, specifically about the disparity between **2048 threads** and **64 cores** in the A100 GPU.
   - They referenced [NVIDIA documentation](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) stating that each dispatch unit can assign instructions for **32 threads per clock**, questioning the book's claims.
- **Clarifying Core Counts in A100**: Discussion arose over the definition of **cores** in the A100 as one member clarified that cores refer to different execution units on the GPU, commonly called *pipes*.
   - They further explained the complexity of executing operations with 64 cores and how a good mix of light integer and heavy float operations could lead to effective execution of up to **128 cores**.



**Link mentioned**: <a href="https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/">NVIDIA Ampere Architecture In&#x2d;Depth | NVIDIA Technical Blog</a>: Today, during the 2020 NVIDIA GTC keynote address, NVIDIA founder and CEO Jensen Huang introduced the new NVIDIA A100 GPU based on the new NVIDIA Ampere GPU architecture. This post gives you a look&#8...

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1313502237321859084)** (33 messages🔥): 

> `Mastodon overview, Nuclear power and GPUs, Environmental impact of GPUs, Efficient training frameworks, AI funding news` 


- **Curiosity about Mastodon**: A member expressed curiosity about [Mastodon](https://letmegooglethat.com/?q=mastodon), indicating a desire for clarification on the platform.
   - *Pessimistic_neko* suggested that this inquiry could easily be answered with a simple search.
- **Nuclear power's potential for energy efficiency**: A discussion emerged around the prospect of **big tech companies** utilizing nuclear power to fuel their GPU operations, with implications for efficiency and sustainability.
   - *Marksaroufim* noted that nuclear power’s reliability could support daytime energy demands while training models at night.
- **The environmental impact of GPUs**: A member highlighted the lack of public awareness regarding the environmental effects of using clusters of GPUs, noting that this discussion is not commonly had.
   - *Rizzware* mentioned the challenge of communicating energy impacts to a broader audience outside tech fields.
- **Creating smarter training frameworks**: Ideas were shared about developing training frameworks that dynamically optimize electricity costs by scheduling model training during cheaper power periods.
   - *S1r_o* humorously suggested the system could adjust training times based on predicted electricity prices.
- **Exciting funding news in AI**: A member announced that **Tenstorrent** raised **$700M** this week, contributing to a recent surge of funding in the AI sector.
   - The announcement included a link to a [YouTube video](https://www.youtube.com/watch?v=_aqMdhAgGG8) featuring Jim Keller discussing AI's impending impact on computing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://letmegooglethat.com/?q=mastodon">Mastodon</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=_aqMdhAgGG8">Tenstorrent&#39;s Keller: We&#39;re in an AI Hype Cycle</a>: Jim Keller of Tenstorrent expects AI to dominate the computing sector over the next decade. He joins Caroline Hyde to discuss on &quot;Bloomberg Technology.&quot;-----...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

0x000ff4: okay I have updated my PR about the kto loss
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1313573550548058133)** (3 messages): 

> `KernelBench, GPU kernels evaluation, Leaderboard issues` 


- **KernelBench Launches with Exciting Features**: Introducing 🌽 [KernelBench](https://twitter.com/anneouyang/status/1864014135824162995) (Preview), a new coding benchmark designed to evaluate the ability of LLMs to generate **efficient** GPU kernels for optimizing neural network performance.
   - This tool aims to enhance benchmarking practices within the domain of GPU computations.
- **Kernel Performance Concerns on Leaderboard**: One user noted that some of the **fastest kernels** on the leaderboard appear to be incomplete, highlighting potential issues in performance evaluation.
   - They provided a link to a specific kernel solution believed to be under scrutiny: [incomplete kernel](https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs/assets/solutions/fc7b0633e1f8dca6653f552f2eeef450.py).
- **KernelBench GitHub Repository Available**: The development of KernelBench can be explored further on its [GitHub page](https://github.com/ScalingIntelligence/KernelBench), inviting contributions and collaboration.
   - This platform allows users to engage in ongoing development and testing of the benchmarking tool.



**Link mentioned**: <a href="https://github.com/ScalingIntelligence/KernelBench">GitHub - ScalingIntelligence/KernelBench</a>: Contribute to ScalingIntelligence/KernelBench development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1313598091684745227)** (1 messages): 

> `Race Condition in TK's WGMMA+tma, Custom Kernel Implementation, Masking Technique for Matrix, Shared Memory Utilization, CUDA Version Compatibility` 


- **Race Condition with Masking in Custom Kernel**: A user reported encountering a **race condition** while implementing a custom kernel using **TK's WGMMA+tma** due to alignment issues in the K dimension.
   - The user found that their masking technique wasn't consistent unless they called it **10 times**, raising concerns over thread synchronization.
- **Innovative Masking Technique for Matrix Operations**: They developed a new **masking function** based on `load` to handle out-of-bounds rows by loading zeros into shared memory.
   - However, despite this innovation, **memcheck/synccheck/initcheck** reports no errors, complicating the debugging process.
- **Exploring Shared Memory Issues**: The implementation's dependence on shared memory and barriers prompted the user to consider potential impacts from recent **refactoring** in the codebase.
   - Given the history of the fork used, they pondered whether updating to the **latest version** could solve some integration problems.
- **CUDA Compatibility Concerns**: The discussion includes the user operating under **CUDA 12.5** while utilizing **bf16s**, leading them to question compatibility with their current setup.
   - They expressed admiration for **ThunderKittens**, acknowledging its ease of use compared to alternatives, despite the technical hurdles.


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1313507390288695296)** (88 messages🔥🔥): 

> `Scams and Bots in Discord, Starting with Stable Diffusion, Using ComfyUI for AI Art Generation, Troubleshooting Stable Diffusion and LoRA, Performance Analysis Tools for SD` 


- **Scams and Bots plague the Discord Community**: It was noted that several **bots** are present in the community, attempting to perform scams such as **Ponzi schemes** or impersonating **Discord support**.
   - *User advice was to report these bots to Discord and avoid interactions with them.*
- **Starting with Stable Diffusion and Tools**: A newcomer seeks advice on getting into **Stable Diffusion**, expressing confusion over tools and models, and a wariness of scams.
   - Users recommended **Vast.ai** for cloud GPU rentals and suggested starting with **ComfyUI** tutorials by Scott on YouTube for better workflows.
- **ComfyUI: Ideal for Advanced AI Art Workflows**: One user recommended **ComfyUI** as a platform for creating AI art, especially for beginners, and emphasized the importance of watching introductory videos.
   - The importance of having a good GPU for local AI work was highlighted, with some users discussing cloud options as a cost-effective solution.
- **Troubleshooting Issues with Stable Diffusion**: Several users reported issues with **LoRA models**, recognizing the necessity of using trigger words in prompts for them to function correctly.
   - Others experienced problems where image results would become jumbled, attributing this to various settings in **Stable Diffusion**.
- **Need for Better Performance Analysis Tools**: A user expressed interest in contributing performance analysis tools to the **Stable Diffusion** community, noting the current lack of such resources.
   - Others agreed, stating that the **SD ecosystem** needs enhancements in performance analysis to improve user experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nvlabs.github.io/Sana/">Sana</a>: no description found</li><li><a href="https://vast.ai/">Rent GPUs | Vast.ai</a>: Reduce your cloud compute costs by 3-5X with the best cloud GPU rentals. Vast.ai&#x27;s simple search interface allows fair comparison of GPU rentals from all providers.</li><li><a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=AbB33AxrcZo&list=PLIF38owJLhR1EGDY4kOnsEnMyolZgza1x">ComfyUI - Getting Started : Episode 1 -  Better than AUTO1111 for Stable Diffusion AI Art generation</a>: Today we cover the basics on how to use ComfyUI to create AI Art using stable diffusion models.  This node based editor is an ideal workflow tool to leave ho...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1313548920646930506)** (72 messages🔥🔥): 

> `Amazon Nova Models, AWS announcements, PydanticAI, OpenAI's 12 Days, Genie 2 by Google` 


- **Amazon Nova Models Announced**: Amazon unveiled its new family of foundation models, Nova, at AWS re:Invent, competing with top models like GPT-4. The announcement included multiple text and video-generating models available on Amazon Bedrock.
   - Feedback from the community is still emerging, with initial thoughts on their performance compared to OpenAI's offerings.
- **AWS Launches New APIs**: AWS launched several API updates, including a Usage API that allows developers to track usage and costs programmatically. Monitoring features include token usage by time and filtering by various identifiers.
   - This functionality aims to improve transparency and management for developers using AWS services.
- **PydanticAI Framework Released**: Pydantic introduced PydanticAI, which aims to simplify the development of applications powered by large language models, emphasizing type safety and modularity. The tool is currently in beta and open-sourced under the MIT License.
   - The framework is positioned as an accessible option for developers looking to leverage LLMs in their projects.
- **OpenAI's 12 Days of Announcements**: OpenAI kicked off a 12-day event featuring daily launches, demos, and updates, beginning on December 5th. Initial stats shared include 300 million weekly active ChatGPT users and 1 billion daily messages on the platform.
   - The anticipation builds around notable announcements, including a potential text-to-video AI tool.
- **Genie 2 Debuts from Google**: Google has launched Genie 2, an autoregressive latent diffusion model designed for video generation and interactive environments. It utilizes a transformer dynamics model and aims to enhance action controllability in generated content.
   - Community discussions highlight curiosity regarding the model's output length and practicality, especially in terms of generated videos.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/12/4/24312352/openai-sora-o1-reasoning-12-days-shipmas">OpenAI’s 12 days of ‘shipmas’ include Sora and new reasoning model</a>: OpenAI has 12 days of Christmas planned.</li><li><a href="https://x.com/sama/status/1864335461268754712?s=46">Tweet from Sam Altman (@sama)</a>: 🎄🎅starting tomorrow at 10 am pacific, we are doing 12 days of openai. each weekday, we will have a livestream with a launch or demo, some big ones and some stocking stuffers. we’ve got some great st...</li><li><a href="https://x.com/hello__caitlin/status/1864367028758565216?s=46">Tweet from c a i t l i n (@hello__caitlin)</a>: I learned 2 things from this year’s Spotify Wrapped:1. I should probably cancel my Spotify premium 2. Their EOY metrics are 100% made-up.</li><li><a href="https://nvlabs.github.io/Sana/">Sana</a>: no description found</li><li><a href="https://x.com/reach_vb/status/1863956316634403260?s=46">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: smol course - learn about instruction tuning, model evaluation, synthetic datasets, inference and more!!🔥100% free and full open source, learn with the community and end the year with a bang! 💥</li><li><a href="https://x.com/mrdbourke/status/1863870479167279486?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Daniel Bourke (@mrdbourke)</a>: New video: Tracking every item in my house with video using Google Gemini 🎥 -&gt; 🛋️I call it &#34;KeepTrack&#34; 😎Input: 10-minute casual walk around video.Output: Structured database w/ 70+ items...</li><li><a href="https://www.interconnects.ai/p/openais-o1-using-search-was-a-psyop">OpenAI&#x27;s o1 using &quot;search&quot; was a PSYOP</a>: How to understand OpenAI&#x27;s o1 models as really just one wacky, wonderful, long chain of thought</li><li><a href="https://x.com/iScienceLuvr/status/1864217903232385348">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Leading computer vision researchers Lucas Beyer (@giffmana), Alexander Kolesnikov (@__kolesnikov__), Xiaohua Zhai (@XiaohuaZhai) have left Google DeepMind to join OpenAI!They were behind recent SOTA v...</li><li><a href="https://ndurner.github.io/amazon-nova">Amazon Nova foundation model release</a>: Since there’s community interest in how to set up AWS to use the new Amazon Nova models, here’s a step-by-step guide to get everyone started:</li><li><a href="https://x.com/exaailabs/status/1864013080944062567?s=46">Tweet from Exa (@ExaAILabs)</a>: Announcing Exa Websets - a breakthrough toward perfect web search.Sign up for the waitlist below👇</li><li><a href="https://x.com/ExaAILabs/status/1806444570210934949">Tweet from Exa (@ExaAILabs)</a>: How does Exa serve billion-scale vector search?We combine binary quantization, Matryoshka embeddings, SIMD, and IVF into a novel system that can beat alternatives like HNSW.@shreyas4_  gave a talk tod...</li><li><a href="https://x.com/lmarena_ai/status/1864062852589605156?s=46">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: Congrats to @amazon on releasing the latest frontier model, Nova!⭐Nova is competitive with top models like GPT-4o on standard benchmarks. Now, the real challenge begins—Nova is in Arena for human eval...</li><li><a href="https://www.interconnects.ai/?r=1h4isl&utm_campaign=referrals-subscribe-page-share-screen&utm_medium=web">Interconnects | Nathan Lambert | Substack</a>: Linking important ideas of AI. The border between high-level and technical thinking. Read by leading engineers, researchers, and investors on Wednesday mornings. Click to read Interconnects, by Nathan...</li><li><a href="https://x.com/openainewsroom/status/1864373399218475440?s=46">Tweet from OpenAI Newsroom (@OpenAINewsroom)</a>: Fresh numbers shared by @sama earlier today: 300M weekly active ChatGPT users1B user messages sent on ChatGPT every day1.3M devs have built on OpenAI in the US</li><li><a href="https://x.com/openainewsr">Tweet from FxTwitter / FixupX</a>: Sorry, that user doesn't exist :(</li><li><a href="https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws">Introducing Amazon Nova, our new generation of foundation models</a>: New state-of-the-art foundation models from Amazon deliver frontier intelligence and industry-leading price performance.</li><li><a href="https://x.com/chipro/status/1864384749911065035">Tweet from Chip Huyen (@chipro)</a>: It’s done! 150,000 words, 200+ illustrations, 250 footnotes, and over 1200 reference links.My editor just told me the manuscript has been sent to the printers. - The ebook will be coming out later thi...</li><li><a href="https://bsky.app/profile/jparkerholder.bsky.social/post/3lcijlzafhs2b">Jack Parker-Holder (@jparkerholder.bsky.social)</a>: Introducing 🧞Genie 2 🧞 - our most capable large-scale foundation world model, which can generate a diverse array of consistent worlds, playable for up to a minute. We believe Genie 2 could unlock th...</li><li><a href="https://x.com/openaidevs/status/1864369714925064606?s=46">Tweet from OpenAI Developers (@OpenAIDevs)</a>: 🆕 The Usage API—track API usage and costs programmatically.👀 Monitor token use by minute/hour/day🔎 Filter usage by API key, project ID, user ID, model, and more💹 Check daily spend via the Costs en...</li><li><a href="https://x.com/skirano/status/1864014133756129752">Tweet from Pietro Schirano (@skirano)</a>: I added a new MCP server that lets Claude think step by step before answering.Claude is able to decide upfront how many thinking steps are needed, retrace its thoughts, and even branch off if it sees ...</li><li><a href="https://bsky.app/profile/m--ric.bsky.social/post/3lcifklp5wc2b">@m--ric.bsky.social</a>: 𝗦𝗵𝗼𝘄𝗨𝗜: 𝗮 𝘀𝗺𝗮𝗹𝗹 𝗲𝗻𝗱-𝘁𝗼-𝗲𝗻𝗱 𝗮𝗴𝗲𝗻𝘁 𝘁𝗵𝗮𝘁 𝗰𝗮𝗻 𝗻𝗮𝘃𝗶𝗴𝗮𝘁𝗲 𝗮𝗻𝘆 𝗨𝗜 📲 and beats much larger VLMs!New paper by NUS &amp; Microsoft, agent that acts on any UI (Deskto...</li><li><a href="https://x.com/dylan522p/status/1864089972644749722">Tweet from Dylan Patel (@dylan522p)</a>: 400k Tranium 2 cluster for Anthropic by AmazonWho said scaling was dead again?See full architecture, server, and software details below.Quoting Dylan Patel (@dylan522p) Amazon’s AI Self SufficiencyTra...</li><li><a href="https://x.com/giffmana/status/1864214549076844556">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: Yoooo, just woke up! What happened? Why they liked this random post?Anyways getting some breakfast with the kiddo, catching up with y’all later!Quoting Lucas Beyer (bl16) (@giffmana) Alright y&#39;all...</li><li><a href="https://x.com/openai/status/1864328928267259941?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from OpenAI (@OpenAI)</a>: 12 days.12 livestreams.A bunch of new things, big and small.12 Days of OpenAI starts tomorrow.</li><li><a href="https://x.com/_philschmid/status/1864016010464080260?s=46">Tweet from Philipp Schmid (@_philschmid)</a>: Unexpected. @amazon is back with Foundation Models. As part of re:Invent they announced 6 new foundation models from text only to text-to-video! 👀 Nova models will be exclusively available through Am...</li><li><a href="https://news.ycombinator.com/item?id=39509937">Genie: Generative Interactive Environments | Hacker News</a>: no description found</li><li><a href="https://x.com/lukeharries_/status/1864017453358932448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Luke Harries (@LukeHarries_)</a>: Building AI agents that can speak used to take monthsAt @elevenlabsio, we saw it first hand. We worked with 100s of startups who all spent 3-6 months building the same Conversational AI stack:- Speech...</li><li><a href="https://x.com/swyx/status/1864137540518990281">Tweet from swyx 🔜 @NeurIPSConf x Latent.Space (@swyx)</a>: Impressed by Amazon Nova: 6 new inhouse models competitive with Gemini/Llama/ Dalle3/SD3.5/ @runwayml video, with speech/full omnimodal along the way. Clipped the entire @AWSCloud Re:invent keynote fo...</li><li><a href="https://venturebeat.com/programming-development/python-data-validator-pydantic-launches-model-agnostic-ai-agent-development-platform/">Python data validator Pydantic launches model agnostic, AI agent development platform</a>: A new agent framework designed to simplify the development of production-grade applications powered by large language models</li><li><a href="https://venturebeat.com/programming-development/python-data-validator-pydantic-launch">Python data validator Pydantic launches model agnostic, AI agent development platform</a>: A new agent framework designed to simplify the development of production-grade applications powered by large language models</li><li><a href="https://x.com/sama/status/1864335461268754712">Tweet from Sam Altman (@sama)</a>: 🎄🎅starting tomorrow at 10 am pacific, we are doing 12 days of openai. each weekday, we will have a livestream with a launch or demo, some big ones and some stocking stuffers. we’ve got some great st...</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking">servers/src/sequentialthinking at main · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card">The Amazon Nova family of models: Technical report and model card</a>: We present Amazon Nova, a new generation of state-of-the-art foundation models that deliver frontier intelligence and industry-leading price performance. Amazon Nova Pro is a highly-capable multimodal...</li><li><a href="https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/">Genie 2: A large-scale foundation world model</a>: Generating unlimited diverse training environments for future general agents</li><li><a href="https://wattenberger.com/thoughts/fish-eye">LLMs are a tool for thought</a>: no description found</li><li><a href="https://x.com/Wattenberger/status/1863977304126603309">Tweet from Amelia Wattenberger 🪷 (@Wattenberger)</a>: 🐟 some musings on how we might use LLMs🐠 to interact with text at multiple levels of abstraction🐡 inspired by the fish-eye lens</li><li><a href="https://youtu.be/v-EYzZCLF48?si=6zA8LCMxk3VQDXWw">Introducing ElevenLabs Conversational AI</a>: Conversational AI is here.Build AI agents that can speak in minutes with low latency, full configurability, and seamless scalability.Let us take care of Spee...</li><li><a href="https://techcrunch.com/2024/12/03/amazon-announces-nova-a-new-family-of-multimodal-ai-models/">Amazon announces Nova, a new family of multimodal AI models | TechCrunch</a>: At its re:Invent 2024 conference, Amazon Web Services (AWS), Amazon&#039;s cloud computing division, announced a new family AI models called Nova.</li><li><a href="https://youtu.be/LY7m5LQliAo?si=gHqvXgAz6Bv9fZIB&">AWS re:Invent 2024 - CEO Keynote with Matt Garman</a>: ​​AWS CEO Matt Garman talks about how AWS is innovating across every aspect of the world’s leading cloud. Explore how AWS are reinventing foundational buildi...</li><li><a href="https://youtu.be/LY7m5LQliAo?si=gHqvXgAz6Bv9fZIB&t=6657">AWS re:Invent 2024 - CEO Keynote with Matt Garman</a>: ​​AWS CEO Matt Garman talks about how AWS is innovating across every aspect of the world’s leading cloud. Explore how AWS are reinventing foundational buildi...</li><li><a href="https://buttondown.com/ainews/archive/ainews-olympus-has-dropped-aka-amazon-nova/">[AINews] Olympus has dropped (aka, Amazon Nova Micro|Lite|Pro|Premier|Canvas|Reel)</a>: Amazon Bedrock is all you need? AI News for 12/2/2024-12/3/2024. We checked 7 subreddits, 433 Twitters and 29 Discords (198 channels, and 2914 messages) for...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: announced next week's monster paper club https://x.com/swyx/status/1864423257266639166
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1313574753709326428)** (51 messages🔥): 

> `LM Studio Download Issues, Performance Issues with Windows, RPG Experiment with LLM, Chat API Functionality, Local Network GPU Usage` 


- **Download Issues for Windows x86 Version**: A user reported an inability to download the Windows x86 version from [lmstudio.ai](https://lmstudio.ai), receiving a message that the file is unavailable.
   - Other users suggested potential CDN issues and recommended using a VPN to attempt the download again.
- **Windows Performance Slowdown with LM Studio**: A member experienced significant performance issues while running LM Studio on Windows compared to Mac, with unexpected output characters from the model.
   - Suggestions included toggling the `Flash Attention` switch and checking system specs to troubleshoot the issue.
- **Experimenting with LLM as RPG Game Master**: A user shared their experience using an LLM to conduct a pre-planned RPG adventure, emphasizing the novelty of writing the outline in Thai to avoid foreknowledge.
   - The experiment yielded engaging results, prompting interest in discussing the methodology and community resources for AI RPG players.
- **Chat API Functionality and Features**: A user inquired about utilizing RAG features within API calls, expressing a desire for visibility of input in the API mode.
   - Discussions revealed the need for custom coding for file attachments and system prompts in API usage, alongside recommendations to compare performance with existing solutions.
- **Using LM Studio with Local Network GPUs**: A user asked about the possibility of connecting to a local server with multiple GPUs from their laptop to run LM Studio.
   - Another member confirmed that it is possible, requiring a frontend for proper functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tlkh/asitop">GitHub - tlkh/asitop: Perf monitoring CLI tool for Apple Silicon</a>: Perf monitoring CLI tool for Apple Silicon. Contribute to tlkh/asitop development by creating an account on GitHub.</li><li><a href="https://lmstudio.ai/docs/cli/log-stream">lms log stream - CLI | LM Studio Docs</a>: Stream logs from LM Studio. Useful for debugging prompts sent to the model.</li><li><a href="https://lmstudio.ai/docs/api/rest-api">LM Studio REST API (beta) - API | LM Studio Docs</a>: The REST API includes enhanced stats such as Token / Second and Time To First Token (TTFT), as well as rich information about models such as loaded vs unloaded, max context, quantization, and more.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1313516118757081199)** (13 messages🔥): 

> `Arc Battlemage Cards, Running LMS on iGPU, Choosing Models for Writing Assistant, PCIe Configuration with 3090s` 


- **Intel's Arc Battlemage Cards face skepticism**: Some users voiced concerns about the new **Arc Battlemage cards**, suggesting that **Intel GPUs** may not be reliable for AI tasks due to poor driver support.
   - *One comment highlighted that using fewer, larger memory GPUs like the 3090 is preferable*.
- **Forcing LMS to run on iGPU**: A user questioned how to force **LMS** to run on the **iGPU** instead of the **dGPU**, noting a lack of options after selecting Vulkan runtime.
   - *The response indicated that adjusting the CUDA visible devices is the current method to choose which GPU LMS uses*.
- **Selecting the right model for summarizing notes**: A member asked for advice on choosing a model for **summarizing notes** that would fit their PC specs, which include a **4070Ti Super**.
   - *Others recommended ensuring the model size in GB fits the available VRAM, aiming for sufficient headroom for performance.*
- **PCIe configuration impact with dual 3090s**: A user inquired about potential performance hits when using a second **3090** on PCIe 4.0 x8 via a riser cable due to space constraints.
   - *It was confirmed that while a secondary card would work, splitting models across two cards can lead to performance issues on Windows.*


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1313561971517558826)** (9 messages🔥): 

> `Building AI apps on Vercel, Intelligent legal document navigation, Amazon Nova foundation models, AI agents with Google Cloud connections, Super-fast RAG with LlamaIndex` 


- **Building AI apps on Vercel just got easier**: The latest [update from LlamaIndex](https://twitter.com/llama_index/status/1864002184138170677) simplifies AI app development on Vercel, enhancing integration capabilities with LlamaCloud.
   - This progression could boost developer productivity and streamline AI app deployment processes.
- **Navigating legal documents like a pro**: An article showcases building an *intelligent legal document navigation system* using advanced multi-graph and multi-agent techniques [here](https://twitter.com/llama_index/status/1864037791019188331).
   - It details how to create document hierarchies and implement a smart traversal workflow for legal documents.
- **Amazon launches competitive Nova models**: Amazon's new family of foundation models, **Nova**, boasts competitive benchmarks and more attractive pricing compared to competitors; ensure support by installing via `pip install llama-index-llms-bedrock-converse` [link here](https://twitter.com/llama_index/status/1864080917029085459).
   - The foundation models aim to offer users a cost-effective and performance-driven alternative in the AI model landscape.
- **Connect AI agents to Google Cloud with LlamaIndex**: LlamaIndex has launched new open-source integrations for Google Cloud’s AlloyDB and Cloud SQL for PostgreSQL, making AI agent development seamless [source](https://twitter.com/llama_index/status/1864364299063578964).
   - This initiative allows developers to effectively leverage cloud databases for enhanced AI functionalities.
- **Rapid RAG implementation with LlamaIndex Workflows**: Learn to build a high-performance Retrieval-Augmented Generation (RAG) system with LlamaIndex Workflows, featuring an event-driven architecture [details here](https://twitter.com/llama_index/status/1864377849295327365).
   - The guide compares this approach with other frameworks such as LangGraph, emphasizing efficiency in complex AI scenarios.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1313509318208913488)** (53 messages🔥): 

> `Summary Index Performance, Using Workflows for Chat History, AI Community Collaboration, Prompt Optimization for LLMs, Error Handling in BM25Retriever` 


- **Summary Index Performance Concerns**: A user raised issues about the slow response time with the **summaryindex** using **sentencesplitter**, stating it takes around **2 minutes** to generate a summary compared to **8 seconds** with ChatGPT.
   - They explored potential improvements but acknowledged that using routers and indexing methods introduces latency.
- **Workflows Simplify Chat Sessions**: A member inquired about managing chat history in **workflows**, specifically asking about options to pass messages easily between steps.
   - Recommendations included using the **Context** feature and **ChatMemoryBuffer** for message management.
- **Growing Community Partnerships**: A member expressed interest in collaborating with the **AIVisuals** community and requested a description and join link for the LlamaIndex community.
   - This indicates a potential for expanding partnerships to enhance community resources.
- **Optimizing Prompts for LLMs**: A user experiencing hallucinations with OpenAI LLMs was advised to try **prompt optimization** to improve response accuracy.
   - It was suggested that crafting better instructions can lead to enhanced performance from the language model.
- **Troubleshooting BM25Retriever Errors**: A user reported a **ValueError** when using **BM25Retriever**, indicating that exactly one of index, nodes, or docstore needs to be passed.
   - This highlights the challenges faced when configuring retrievers in the LlamaIndex library.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/i8bow7sr">Voice &amp; Video AI Agents Hackathon · Luma</a>: Gen AI AgentsCreatorsCorner, collaborating with AWS, Temporal, Modal, Tandem, Marly, Retell, Senso, Unified, Speedlegal, Corval, Simli, PolyAPI and others…</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/#working-with-global-contextstate">Workflows - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1313518093439336488)** (15 messages🔥): 

> `Rerank 3.5 Multilingual Support, Google Gemini Functionality, Cohere Toolkit Errors, R+ Word Usage Observations, General AI Preferences` 


- **Rerank 3.5 supports multilingual functionality**: Members confirmed that you can switch to **Rerank 3.5**, which supports both **multilingual** and **English** rankings despite initial documentation suggesting it was limited.
   - As one member noted, *'the documentation states that Rerank 3.5 allows re-ranking for English.'*
- **Clarification needed on Google Gemini**: A member asked for explanations about **Google Gemini**, noting inconsistent access to Google Drive documents among other issues.
   - Responses suggested seeking help in other forums like Reddit due to the limitations in resolving this within the current channel.
- **Cohere Toolkit encounters errors**: A user reported warnings while running the **cohere-toolkit**, particularly related to *alembic* and the absence of support for libraries like PyTorch and TensorFlow.
   - They mentioned that their **PyTorch version** is **2.5.1**, prompting the inquiry if anyone knows how to fix the issues.
- **R+ unusual word repetition noted**: A member highlighted that even with high temperature settings, the word *'section'* appears frequently in **R+** responses.
   - Observations noted that this peculiar behavior occurs every sixth or seventh response, raising questions about the generation pattern.
- **General preference discussion on AIs**: A user initiated a light-hearted chat by asking which **AI** people prefer the most, fostering community engagement.
   - Responses were mixed with fun, but no specific preferences were elaborated upon.


  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1313552078953648159)** (1 messages): 

> `Rerank 3.5, Model deprecations, Multilingual performance, Enhanced reasoning capabilities` 


- **Rerank 3.5 launched with SOTA performance**: Cohere announced the release of **Rerank 3.5**, delivering state-of-the-art performance in processing complex user queries and enhanced reasoning skills.
   - *Check out the full details in our [blog post](https://cohere.com/blog/rerank-3pt5)*, which highlights improved compatibility with diverse data types and languages.
- **Improved multilingual capabilities across 100+ languages**: **Rerank 3.5** boasts improved performance in over **100 languages** including Arabic, French, Japanese, and Korean, enabling better searches in multilingual environments.
   - This enhancement allows users to extract relevant information from long documents like emails and reports more efficiently, catering to global applications.
- **Documentation on model deprecations now available**: Cohere provided an update on **model deprecations**, detailing the lifecycle stages of models including **Active**, **Legacy**, and **Deprecated**.
   - Developers can refer to [this documentation](https://docs.cohere.com/docs/deprecations) for recommended replacements for any deprecated endpoints and models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/rerank-3pt5">Introducing Rerank 3.5: Precise AI Search</a>: Rerank 3.5 delivers improved reasoning and multilingual capabilities to search complex enterprise data with greater accuracy. </li><li><a href="https://docs.cohere.com/docs/deprecations">Deprecations — Cohere</a>: Learn about Cohere&#x27;s deprecation policies and recommended replacements
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1313752063402053704)** (6 messages): 

> `API Key Types, ReRanker Performance Issues, Cohere Team Access, Model Sharing` 


- **API Key Types Clarified**: Cohere offers two types of API keys: **trial** (limited usage) and **production** (less limited). Users can **create** these keys on the [API keys page](https://dashboard.cohere.com/api-keys) and check [rate limits](https://docs.cohere.com/v2/docs/rate-limits) for various endpoints.
   - For more information on pricing, users can refer to the [pricing docs](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work).
- **ReRanker Performance Drops Noted**: A user reported a **30% performance drop** with the 'rerank-multilingual-v3.0' model since a change occurred yesterday. The new **rerank 3.5** model performed even worse, which prompted the concern.
   - Cohere's **support team** acknowledged the issue and will assist in troubleshooting the problem.
- **Accessing Cohere Team**: A user experienced difficulties switching teams within Cohere's platform despite being invited. They were advised to check with their **team admin** to ensure they were properly added.
   - If the issue persists after being added, the user was encouraged to contact support at **support@cohere.com**.
- **Collaboration on Model Sharing**: It was clarified that users can share model keys with coworkers if they are using the same fine-tuned model. Cohere API keys grant access to those models, allowing collaboration.
   - For more details on available models, users can check the [Cohere models documentation](https://docs.cohere.com/v2/docs/models).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/v2/docs/rate-limits">API Keys and Rate Limits — Cohere</a>: This page describes Cohere API rate limits for production and evaluation keys.</li><li><a href="https://docs.cohere.com/v2/docs/models">Models Overview — Cohere</a>: Cohere has a variety of models that cover many different use cases. If you need more customization, you can train a model to tune it to your specific use case.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1313738460632449025)** (2 messages): 

> `V3.5 Launch, Fine-Tuning API, Base Model Updates` 


- **V3.5 Launch does not include Fine-Tuning API updates**: A user inquired whether the fine-tuning API has transitioned to **v3.5** with its launch.
   - Another member responded that currently, **rerank 3.5** is not offered via fine-tuning API, and the base model remains at **2.0**.
- **Base Model Status Undefined**: In response to queries, it was clarified that the fine-tuning API is not using **v3.5** yet and continues with the base model **2.0**.
   - This means users will not have access to updates or features from **v3.5** when using the fine-tuning API.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1313518121478520904)** (1 messages): 

> `Harmony project, LLM matching competition, Data harmonisation, Natural Language Processing, Discord Community` 


- **Harmony project aids questionnaire harmonisation**: The **Harmony project** is focused on retrospectively harmonizing questionnaire items and metadata, and is available at [Harmony Data](https://harmonydata.ac.uk/). It leverages **Natural Language Processing** to assist researchers in comparing questionnaires across different studies and languages.
   - For those interested, there are methods to [compare questionnaire items](https://harmonydata.ac.uk/compare-harmonise-instruments/gad-7-vs-beck-anxiety-inventory/) and ensure compatibility across various versions of the same questionnaire.
- **Competition for LLM matching algorithms**: The Harmony project is running a competition on [DOXA AI](https://harmonydata.ac.uk/doxa/) to enhance their LLM matching algorithms, with prizes of up to **£500** in vouchers. Participants can register to fine-tune their own language models for mental health data.
   - You don't need prior experience with LLMs to join, and there's an opportunity to engage via the Harmony's Discord server, particularly in the 🏅「matching-challenge」 channel.
- **Evaluating Harmony's algorithm performance**: A blog post evaluates Harmony's matching algorithm, noting its occasional misjudgment in sentence similarity as perceived by psychologists, which can lead to discrepancies in questionnaire comparisons. Performance metrics and insights can be found in the [evaluation blog post](https://harmonydata.ac.uk/nlp-semantic-text-matching/measuring-the-performance-of-nlp-algorithms/).
   - This evaluation raises important points about improving algorithm accuracy and enhancing user trust in the tool's functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://harmonydata.ac.uk/">Harmony | A global platform for contextual data harmonisation</a>: A global platform for contextual data harmonisation</li><li><a href="https://harmonydata.ac.uk/doxa/">Competition to train a Large Language Model for Harmony on DOXA AI | Harmony</a>: A global platform for contextual data harmonisation
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1313568775341080648)** (3 messages): 

> `Pydantic AI, DSLModel development, AI Development Live Demo` 


- **Pydantic AI integrates smoothly with DSLModel**: The addition of [Pydantic AI](https://ai.pydantic.dev/) to DSLModel provides an agent framework that enhances the usability of LLMs with Pydantic's powerful features.
   - A member expressed excitement about how **Pydantic** facilitates more ergonomic development in AI projects when paired with frameworks like FastAPI.
- **Get started with DSLModel**: Developers can easily install DSLModel via pip using the command `pip install dslmodel` to begin leveraging its capabilities.
   - The project is discussed further in an introduction video titled [Welcome to DSLModel](https://www.loom.com/share/67dd1db910ae424eb89e249e676bbaf0).
- **Live Demo Event on AI Development**: A live demo event titled **Master AI Development: PydanticAI + DSPy + DSLModel Deep Dive** will explore cutting-edge technologies in AI development.
   - The event can be viewed on [YouTube](https://youtube.com/live/mBQFKo8bPBI) and aims to uncover innovative ways to utilize PydanticAI and related tools in projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/live/mBQFKo8bPBI">Master AI Development: PydanticAI + DSPy + DSLModel Deep Dive (Live Demo)</a>: https://ai.pydantic.dev/https://dspy.ai/https://pypi.org/project/dslmodel/🚀 Join us live as we explore the cutting edge of AI development! Discover how to c...</li><li><a href="https://ai.pydantic.dev/">Introduction</a>: Agent Framework / shim to use Pydantic with LLMs</li><li><a href="https://pypi.org/project/dslmodel/">dslmodel</a>: Pydantic + DSPy instances from prompts and Jinja.
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1313549828353032222)** (19 messages🔥): 

> `DSPy Optimizations on AWS Lambda, ProgramOfThought Deprecation, Precision Evaluation in Multi-Class Classification` 


- **DSPy Optimizations Face AWS Lambda's 15-Minute Limit**: Members discussed the challenges of running **DSPy optimizations** on **AWS Lambda**, notably the **15-minute execution limit** for long-running tasks.
   - One user suggested utilizing a **/tmp folder** for caching due to Lambda's read-only filesystem to mitigate speed issues.
- **ProgramOfThought to be Revamped in v2.6**: Concerns were raised regarding the support status of **ProgramOfThought** post **v2.5**, with members noting that it will be revamped in **v2.6** expected this year.
   - Users were advised to use the current version with caution as the upgrade approaches.
- **Precision Evaluation Approach for Class Imbalance**: A member inquired about constructing a metric for evaluating **precision** on a specific class in a **multi-class classification problem** amidst significant class imbalance.
   - Others suggested using **dspy.Example(batch=[...])** to handle the evaluation but acknowledged the difficulty due to the **class imbalance**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1313553912363749387)** (2 messages): 

> `Sierra AI Info Session, Hackathon Submission Form, Submission Requirements Guide, Google Forms for submissions, Judging panel and timeline` 


- **Join the Sierra AI Info Session!**: An exclusive info session with **Sierra AI**, a leading conversational AI platform, is happening now. You can watch it live [here](https://www.youtube.com/watch?v=-iWdjbkVgGQ).
   - Sierra is eager to connect with talented developers, so don't miss out on this opportunity!
- **Hackathon Submission Form is Live!**: The **Submission Form and Requirements Guide** for the LLM Agents MOOC Hackathon are now available, with a project submission deadline of **December 17th**. The submission process has transitioned from **Devpost to Google Forms**.
   - You can find all details to submit your projects through the links provided, and be eligible for evaluation from a distinguished panel of judges.
- **Important Links for Submission**: Participants can access the **Hackathon Submission Form** [here](https://forms.gle/jNr8nSH9Cy9qpYcu5) and the **Submission Requirements Guide** [here](https://docs.google.com/document/d/1WgWLZocBFM08cVVxo9P-ZMCnHBLGmQ7v8PbH4-AwnSk/edit?usp=sharing).
   - Make sure to prepare diligently and submit your innovative solutions to be in the running for prizes!
- **Timeline for Hackathon Winners**: Winners of the LLM Agents MOOC Hackathon will be announced in the first half of **January 2025**. Participants are encouraged to submit by the deadline for evaluation.
   - The organizers look forward to seeing creative solutions from all participants, and questions are welcomed in the channel.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=-iWdjbkVgGQ">LLM Agents MOOC Hackathon - Sierra Information Session</a>: no description found

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1313721707990749305)** (1 messages): 

> `Certificate Declaration Form, Course Completion Tiers, Submission Checklist, Important Due Dates` 


- **Certificate Declaration Form Released**: The **Certificate Declaration Form** has been released for those attempting to earn a course completion certificate, which can be completed using the link [here](https://forms.gle/nYGJLPTdb7af2Dq59). Participants must use the same email address throughout their submissions to ensure they receive their certificate.
   - *Make sure you use the same email address when submitting all of your work*; capitalization matters, but extra punctuation does not.
- **Five Course Completion Tiers Explained**: Participants can earn a certificate from one of five completion tiers: **Trailblazer, Mastery, Ninja, Legendary, or Honorary Tier**. Only one certificate can be earned, and progress is tracked via email.
   - Reminder that each tier has its own specific requirements that must be fulfilled in order to qualify.
- **Checklist for Certificate Eligibility**: To earn the certificate, complete all necessary coursework, including **12 quizzes, a written article**, and any tier-specific requirements like lab submissions and project forms. A checklist is available on the course website for reference.
   - Ensure the **Certificate Declaration Form** is also submitted to earn the certificate.
- **Important Due Dates for Submissions**: All article submissions, quizzes, and labs are due by **December 12, 2024**, at 11:59PM PST. Hackathon project submissions and the Certificate Declaration Form are due by **December 17, 2024**, at 11:59PM PST.
   - Be mindful of these deadlines to secure your course completion certificate.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1313601425355440199)** (14 messages🔥): 

> `Project Submission Requirements, Quizzes and Certificate Deadlines, Certification Declaration Categories, Feedback on MOOC Experience` 


- **Project Submission Details Clarified**: Members inquired about what specific files should be included in their project submissions, and it was confirmed that more than one track for evaluation is acceptable. For detailed requirements, they were directed to the [project submission document](https://docs.google.com/document/d/1WgWLZocBFM08cVVxo9P-ZMCnHBLGmQ7v8PbH4-AwnSk/edit?tab=t.0#heading=h.s229pxj2lhn2).
   - The member was also reminded to check if they could apply for both Masters and Trailblazer categories.
- **Quizzes and Certificates Open Until December**: It was confirmed that all quizzes remain open until **December 12th**, and the certificate declaration is due on **December 17th**. This allows ample time for participants to complete their necessary assessments.
   - Members expressed interest in understanding what they must complete to receive their certificates.
- **Masters vs Trailblazer Certification Categories**: A member asked if participants can apply for both Masters and Trailblazer categories, and was assured they could be downgraded automatically to the trailblazer tier if necessary. No new form submission would be required in such a case.
   - This offers flexibility for participants who do not meet the Masters threshold.
- **Positive Feedback on MOOC**: Participants expressed gratitude for the organization of the MOOC, highlighting the supportive environment provided throughout the course. They emphasized that the course helped distill the complexities of the current LLM ecosystem.
   - Additionally, the speaker lineup was praised for adding value to the learning experience, with a focus on understanding each speaker's background.



**Link mentioned**: <a href="https://docs.google.com/document/d/1WgWLZocBFM08cVVxo9P-ZMCnHBLGmQ7v8PbH4-AwnSk/edit?tab=t.0#heading=h.s229pxj2lhn2">Hackathon Track Submission Requirements</a>: General Submission Requirements for All Tracks Video Presentation: Provide a link to a YouTube video (maximum 3 minutes; please upload to YT) presenting an overview of your project and demonstrating k...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1313653357201592351)** (4 messages): 

> `GPT-4 leaks, Automated closed captioning` 


- **Concerns Over GPT-4 Data Leak**: *A user raised questions about the source of the leak* regarding **GPT-4**, specifically if it pertained to the consumer or enterprise version. There are implications that the default setting may have been reset to share user data for modeling purposes for at least **30 days**.
   - Another comment noted a potential jailbreak of **GPT-4** that could reveal real PII from the training set, referencing the landmark **AOL case**.
- **Request for Automated Closed Captioning**: *A member highlighted the absence of automated closed captioning* for the last lecture, noting its importance for people with hearing disabilities. They suggested enabling this feature to improve accessibility.


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1313500744263143464)** (15 messages🔥): 

> `Anthropic Development Branch, Open Interpreter Installation Issues, Linux Compatibility, Feedback on OpenAI SDKs` 


- **User struggles with Anthropic branch**: A user reported encountering a `TypeError` when attempting to use the latest development branch with Anthropic, specifically stating the argument 'proxies' was unexpected.
   - Another member suggested checking for a custom API base, indicating this was likely the only change affecting the client initialization.
- **Installation commands and recommendations**: A suggestion was made to reinstall the latest development version of Open Interpreter using the command `pip install --force-reinstall git+https://github.com/OpenInterpreter/open-interpreter.git@development`.
   - The developer confirmed that the project has been completely rewritten for better performance and encouraged users to report any missing features.
- **Feedback sought for the new implementation**: Discussion ensued about the need for user feedback on the new OpenAI compatible implementation to ensure it surpasses previous versions.
   - A developer expressed a desire to offer comprehensive support across all OpenAI SDKs after receiving input from users.
- **User experiences across Linux distributions**: A user confirmed that Open Interpreter was operational on Garuda-Linux, an Arch-Linux fork, appreciating its compatibility.
   - This user also shared experience with multiple other Linux distributions, such as Manjaro and OpenSuse, highlighting their extensive testing.
- **Open Interpreter approval requirements**: Another user noted that Open Interpreter requires approval before executing code but can bypass this with the `interpreter -y` command.
   - This reveals part of the functionality built into the software for user safety before allowing code execution.



**Link mentioned**: <a href="https://tenor.com/view/so-close-this-the-office-gif-1505267913606309297">So Close GIF - So Close This - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1313557411969241088)** (1 messages): 

> `LiveKit usage, Remote Control via O1, Computer as a Tool, CLI capabilities of OI` 


- **Leveraging LiveKit for Device Connection**: O1 typically utilizes **LiveKit** to connect two devices, such as an **iPhone** for communication and a laptop or **Raspberry Pi** for receiving requests.
   - This setup allows for efficient remote access to control your machine via the local OpenInterpreter instance.
- **Enhanced Capacity with O1**: The capacity with O1 surpasses that of other setups in terms of **computer use**, enabling more flexibility when using devices as tools.
   - Even in the **CLI form**, OpenInterpreter remains capable of effectively operating the computer.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

pjbontrager: I don’t know what you’re talking about 😗😅
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1313934393399574548)** (2 messages): 

> `Genie 2 Foundation Model, Generalist Agents Team` 


- **Genie 2 Takes Center Stage**: A request was made to add information about **Genie 2**, a large-scale foundation world model, to torchtune within the next day. More details can be found in the [official blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/).
   - The acknowledgements highlight contributions from key figures like **Jack Parker-Holder** and **Stephen Spencer**, emphasizing collaborative efforts in the project's development.
- **Generalist Agents Team Highlights**: The **Generalist Agents team**, led by Vlad Mnih, made significant strides with contributions from members like **Harris Chan** and **Maxime Gazeau**. These efforts underline the project's comprehensive approach to agent development.
   - Further support from the **SIMA team**, including Frederic Besse and Tim Harley, showcases the diverse expertise brought together for this initiative.
- **Community Reaction to Updates**: An enthusiastic reaction to the updates was expressed, described simply as *insane*. This reflects the community's eagerness for advancements in AI projects.



**Link mentioned**: <a href="https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/">Genie 2: A large-scale foundation world model</a>: Generating unlimited diverse training environments for future general agents

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1313501402873593948)** (12 messages🔥): 

> `Federated learning approaches, Community-led GPU contributions, MMLU performance validation, Training timelines, Meta's technology comparison` 


- **Federated Learning Shows Promise**: The underlying **federated learning** approach may yield better results than fully synchronous methods, as discussed in a shared [paper](https://arxiv.org/pdf/2411.19870).
   - *Only 22 hours left on training* indicates a nearing completion.
- **Community Contributions Could Revive Folding@home Model**: There's interesting potential for community-led efforts similar to **Folding@home**, with individuals contributing GPU time.
   - This could become crucial as models outgrow individual data centers.
- **MMLU Pro Sets Validation Bar**: To validate a block in the discussed framework, the model needs to achieve **90%** on **MMLU Pro**.
   - This highlights the rigorous performance standards needed for successful deployments.
- **Comparison of Meta's Technologies**: In light of discussions surrounding fat clusters, concerns arose about whether **Meta** possesses similar technologies.
   - One contributor expressed that larger models may necessitate interesting approaches, regardless of having many GPUs.
- **Excitement Over Resource-Intensive Developments**: Users expressed enthusiasm about advancements in AI and related fields, noting the potential implications of faster training timelines.
   - One noted, *Damn that's crazy*, reflecting excitement about ongoing progress in the area.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sakana.ai/cycleqd/">no title found</a>: no description found</li><li><a href="https://distro.nousresearch.com/">Nous DisTrO</a>: Distributed training over the internet
</li>
</ul>

</div>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1313711136960548925)** (5 messages): 

> `Mechanistic Interpretability, Cellular Behavior, Epistemic Advantage` 


- **Digging into Cellular Mindset with Mechanistic Interpretability**: Researchers highlight a new tool called **mechanistic interpretability** to explore how cells model their environments, shifting focus from genes to concepts like **gene regulatory modules** and **sub-cellular locations**.
   - This approach may allow us to construct a 'folk psychology of cellular behavior' and understand the **inner life of cells** in more relatable terms.
- **Absurdity in Therapy for Cells**: A member pointed out the absurdity of considering cells as needing therapy, acknowledging that cells do not think like humans do.
   - Nonetheless, rethinking our understanding in this way could provide **epistemic advantages** compared to traditional views on cellular function.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1313703981482250350)** (3 messages): 

> `Non-commercial license concerns, EDM2 framework diffusion models, Class conditioning in diffusion models` 


- **Non-commercial license may limit implementations**: A member noted that the diffusion model's **non-commercial license** should deter attempts to implement it widely.
   - This restriction could impact the adoption and experimentation with the model among developers.
- **Inquiry on EDM2 framework for diffusion models**: Another member asked if anyone has utilized the **EDM2 framework** for training diffusion models with text conditioning.
   - They pointed to a [paper](https://arxiv.org/pdf/2312.02696) that showcases **impressive results**, highlighting a gap in the specific implementation.
- **Limitations of class conditioning in diffusion models**: The paper mentioned class conditioning, indicating that the model can only generate outputs specific to a few **predefined classes**.
   - This limited approach is contrasted with the desired flexibility of text conditioning to allow broader creativity in generation.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1313783589472833578)** (1 messages): 

> `Web models, SAM from Meta, Tinygrad showcase` 


- **Web models gaining traction**: A priority discussed is the development of **Web models** such as ONNX in the cloud, enhancing accessibility in machine learning tools.
   - These models demonstrate potential for user engagement by offering functionalities that run in both the cloud and directly in the browser.
- **Interesting demo on SAM from Meta**: A member presented on **SAM from Meta**, highlighting its demo website which is deemed user-friendly, showcasing models that are effective out of the box.
   - The **600M image embedding transformer** runs in the cloud, while smaller models operate directly in the browser, illustrating an example of practical application.
- **Quality baseline for upcoming models**: The SAM demo sets a possible quality baseline for future models and webpages designed to showcase **tinygrad**, aiming to increase traction in the community.
   - While not flawless, the improvements noted in the demo aptly reflect the advancements expected in upcoming AI tools.



**Link mentioned**: <a href="https://segment-anything.com/demo.">Segment Anything</a>: Meta AI Computer Vision Research

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1313628553568321596)** (6 messages): 

> `Threadgroup/Grid Sizes, BEAM Search Explanation, Shared Output Buffers in JIT, Manual Upcasting for Loops` 


- **Threadgroup/Grid Sizes can be changed in OptOps**: A user inquired whether the threadgroup/grid sizes can be altered during graph rewrite optimizations in `uopgraph.py`. George Hotz replied that they can be modified, specifically in **OptOps** within **kernel.py**.
- **Explained BEAM Search and Kernel Optimization Options**: A user shared a post on [BEAM Search](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241203_beam.md) along with an explanation of the kernel optimization options. The post serves as a resource for understanding these concepts in **tinygrad**.
- **JIT Functions Reuse Output Buffers**: A note about JIT functions revealed that after the first call, jitted functions reuse the same output buffer, which may overwrite previous results. To preserve results, it's necessary to use `.clone().realize()` after each call.
- **Manual Upcasting for Large Loops**: A user asked about the possibility of manually forcing upcasting for a large loop. The conversation continues with no direct answer provided.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/m">m - Overview</a>: Typist, engineer, code poet, lover of beautiful data structures. - m</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241203_beam.md">tinygrad-notes/20241203_beam.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Axolotl AI ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/1313703914096427079)** (1 messages): 

> `Office Hours, Axolotl Survey, Axolotl Swag` 


- **Reminder for Upcoming Office Hours**: A friendly reminder that office hours are set for this Thursday, **12/5** at **1pm Eastern** / **10am Pacific**.
   - The team is eager to engage with everyone during this session!
- **Input Wanted: Fill Out the Axolotl Survey**: To tailor the discussion effectively, participants are invited to complete the **Axolotl Survey**.
   - Feedback will guide improvements, and those who participate will receive exclusive **Axolotl swag**!
- **Limited Edition Axolotl Swag for Survey Completion**: As a token of appreciation for completing the survey, respondents will receive soon-to-be-released **Axolotl swag** (while supplies last).
   - This incentive underscores the team's commitment to valuing participant time and input.



**Link mentioned**: <a href="https://gravel-salmon-db9.notion.site/1421d2ab4f4081168f6fe3770fae446c">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1313557490226565193)** (3 messages): 

> `ADOPT optimizer, Axolotl updates` 


- **ADOPT Optimizer Updates Integrated into Axolotl**: The latest updates for the **ADOPT optimizer** have been integrated into the Axolotl codebase, aiming to improve **training stability**. Check out the changes in the [pull request #2104](https://github.com/axolotl-ai-cloud/axolotl/pull/2104).
   - The pull request ensures compatibility with the **torch version** and incorporates recent modifications made by the original author [here](https://github.com/iShohei220/adopt).
- **ADOPT Optimizer's Key Strengths Discussed**: A member inquired about the advantages of the **ADOPT optimizer** after its implementation, indicating a curiosity about its benefits.
   - In response, it was noted that the optimizer can achieve **optimal convergence** with any beta value.



**Link mentioned**: <a href="https://github.com/axolotl-ai-cloud/axolotl/pull/2104">Check torch version for ADOPT optimizer + integrating new ADOPT updates by bursteratom · Pull Request #2104 · axolotl-ai-cloud/axolotl</a>: DescriptionMake sure the torch version is compatible when ADOPT optimizer is used.Incorporated latest changes to ADOPT optimizer made by original author. https://github.com/iShohei220/adoptMotiv...

  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1313947537845129267)** (1 messages): 

> `Open Source Engineer Roles, Unternet Hiring` 


- **Unternet seeks Open Source Engineer**: [Unternet is hiring an Open Source Engineer](https://discord.com/channels/1089876418936180786/1313839138562248737) to contribute to open source projects, create technical documentation, and engage with the community.
   - Interested candidates are welcomed to inquire further in the thread linked above.
- **Community Engagement Opportunity**: The job position emphasizes the importance of collaborating with the community while also developing technical documentation.
   - This role is aimed at individuals passionate about open source contributions.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1313856112520593461)** (1 messages): 

> `Gorilla Model Issue, Protobuf Dependency Error` 


- **Gorilla Model Fails to Start**: A user encountered an error when attempting to start their model using the command, indicating a dependency issue related to the tokenizer.
   - *The error message highlighted the absence of the protobuf library,* despite it being installed in their environment.
- **Protobuf Library Not Found**: The user confirmed that the protobuf package was installed with version **5.29.0**, but the system still reported it as missing.
   - This has led to questions about what could be causing the environment to not recognize the installed package.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1313660575003443271)** (1 messages): 

> `Ticket Messaging` 


- **Follow-Up on Ticket Message**: A member prompted Nick to check a message they sent about their ticket, requesting him to look at it when he has time.
   - They emphasized the importance of timely responses, hinting at the need for quick resolution.
- **No Additional Context Provided**: The conversation did not provide any further context beyond the follow-up about the ticket.
   - There were no additional comments or links discussed.


  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
