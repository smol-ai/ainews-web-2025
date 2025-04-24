---
id: 5a1c6a0b-4f1a-4a34-88de-3563bb82d098
title: 'Cohere''s Command A claims #3 open model spot (after DeepSeek and Gemma)'
date: '2025-03-18T00:28:53.405655Z'
original_slug: ainews-coheres-command-a-claims-3-open-model-spot
description: >-
  Cohere's Command A model has risen to #13 on the LMArena leaderboard,
  showcasing its strong performance with a 256K context window and style control
  capabilities. Meanwhile, Mistral AI released the lightweight, multilingual
  Mistral Small 3.1 model featuring a 128K context window and Apache 2.0
  license, gaining praise for its fast, low-latency responses and advanced
  reasoning.
tags:
  - cohere
  - command-a
  - mistral
  - mistral-small-3.1
  - large-language-models
  - context-window
  - open-weights
---


<!-- buttondown-editor-mode: plaintext -->**Yay for open weights models!**

> AI News for 3/14/2025-3/17/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**223** channels, and **9014** messages) for you. Estimated reading time saved (at 200wpm): **990 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We briefly mentioned Cohere's Command A launch [last week](https://buttondown.com/ainews/archive/ainews-not-much-happened-today-8188/), but since the announcement was comparatively light on broadly comparable benchmarks (there were some, but the selective, self reported, comparisons to DeepSeek V3 and GPT-4o couldnt really contextualize Command A among either SOTA open source or overall SOTA-for-size), it was hard to tell where it would rank in terms of lasting impact. 

With today's LMArena result, that is no longer in question:

![image.png](https://assets.buttondown.email/images/73c613a0-d833-4e35-89ac-af1cb6132bc4.png?w=960&fit=max)

As [Aidan Gomez points out](https://x.com/aidangomez/status/1901669060175151609), Command A actually *increases* 2 spots in rankings with the Style Control modifier (explored on [their LS podcast](https://www.latent.space/p/lmarena)).

There are many other notable subtle points that make Command A a particularly attractive candidate to include in one's open models arsenal, including the unusually long 256k context window, multilingual capabilities, and focus on optimizing for a 2-H100 serving footprint.

![image.png](https://assets.buttondown.email/images/94417e9c-d2c5-4c51-ace1-67d561e12667.png?w=960&fit=max)



---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Large Language Models (LLMs) and Model Releases**

- **Mistral AI Small 3.1 release (multimodal, multilingual, Apache 2.0 license)**: [@sophiamyang](https://twitter.com/sophiamyang/status/1901675671815901688) announced the release of **Mistral AI Small 3.1**, highlighting its lightweight nature (runs on a single RTX 4090 or a Mac with 32GB RAM), fast response conversations, low-latency function calling, specialized fine-tuning, and advanced reasoning foundation. It outperforms comparable models on instruct benchmarks [@sophiamyang](https://twitter.com/sophiamyang/status/1901676305025774020) and multimodal instruct benchmarks [@sophiamyang](https://twitter.com/sophiamyang/status/1901676575965282395), and is available on Hugging Face [@sophiamyang](https://twitter.com/sophiamyang/status/1901677007278092508), Mistral AI La Plateforme [@sophiamyang](https://twitter.com/sophiamyang/status/1901677125918134276), and with enterprise deployments [@sophiamyang](https://twitter.com/sophiamyang/status/1901677325588078774). The model is praised for its multilingual and long context capabilities [@sophiamyang](https://twitter.com/sophiamyang/status/1901676699361882439).  [@reach_vb](https://twitter.com/reach_vb/status/1901670885188071545) emphasized the 128K context window and Apache 2.0 license.
- **SmolDocling: New OCR model**: [@mervenoyann](https://twitter.com/mervenoyann/status/1901668060257190186) introduced **SmolDocling**, a fast OCR model that reads a single document in 0.35 seconds using 0.5GB VRAM, outperforming larger models, including Qwen2.5VL. It is based on SmolVLM and trained on pages and Docling transcriptions.  The model and demo are available on Hugging Face [@mervenoyann](https://twitter.com/mervenoyann/status/1901668064602579150).
- **Cohere Command A Model**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1901668148031758605) reported that **Cohere's Command A** has climbed to #13 on the Arena leaderboard, highlighting its open-weight model (111B), 256K context window, and pricing of $2.5/$10 input/output MTok. Command A also ranked well in style control [@aidangomez](https://twitter.com/aidangomez/status/1901669060175151609).
- **Discussion on better LLMs**: [@lateinteraction](https://twitter.com/lateinteraction/status/1901642081770295732) expressed a cynical view that recent improvements in LLMs are due to building LLM systems (CoT) rather than better LLMs themselves, questioning where the better LLMs are.

**Model Performance, Benchmarks, and Evaluations**

- **MCBench as a superior AI benchmark**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1901671231713427512) recommends mcbench as the best AI benchmark, noting its fun-to-audit data, testing of relevant features (code, aesthetics, awareness), and ability to discern performance differences among top models. The benchmark can be found at https://t.co/YEgzhLotKk [@aidan_mclau](https://twitter.com/aidan_mclau/status/1901671234125095205)
- **HCAST benchmark for autonomous software tasks**: [@idavidrein](https://twitter.com/idavidrein/status/1901647558839353363) shared details about **HCAST (Human-Calibrated Autonomy Software Tasks)**, a benchmark developed at METR to measure the abilities of frontier AI systems to complete diverse software tasks autonomously.
- **AI models on patents**: [@casper_hansen_](https://twitter.com/casper_hansen_/status/1901540769040683214) tested models on instruction following on patents and found that Mistral Small 3 is better than Gemini Flash 2.0, with Mistral models pre-trained on more patents.
- **Generalization deficits in LLMs**: [@JJitsev](https://twitter.com/JJitsev/status/1901467121592201490) shared an update to their paper, including sections on recent reasoning models, questioning their ability to handle AIW problem versions that revealed severe generalization deficits in SOTA LLMs.
- **Evaluating models on OpenRouter**: [@casper_hansen_](https://twitter.com/casper_hansen_/status/1901539872315257286) noted that OpenRouter is a useful tool for testing new models, but the free credits are limited to 200 requests/day.

**AI Agents, Tool Use, and Applications**

- **AI agents interacting with external tools**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1901403297988374853) explained that AI agents interact with external tools or apps using UI-based and API-based interactions, with modern AI agent frameworks prioritizing API-based tools for their speed and reliability.
- **TxAgent: AI Agent for Therapeutic Reasoning**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1901555282594971761) introduced **TXAGENT**, an AI agent leveraging multi-step reasoning and real-time biomedical knowledge retrieval across a toolbox of 211 tools to analyze drug interactions, contraindications, and patient-specific treatment strategies.
- **Realm-X Assistant**: [@LangChainAI](https://twitter.com/LangChainAI/status/1901699861264900112) highlighted **AppFolio’s Realm-X Assistant**, an AI copilot powered by LangGraph and LangSmith, designed to streamline property managers’ daily tasks.  Moving Realm-X to LangGraph increased response accuracy 2x.
- **AI for error and data analysis**: [@gneubig](https://twitter.com/gneubig/status/1901679380205609324) expressed excitement about the ability of AI agents to perform more nuanced error analysis and data analysis than humans can do quickly.
- **Multi-agentic player pair programming**: [@karinanguyen_](https://twitter.com/karinanguyen_/status/1901667981631086915) shared an idea sketch for multi-agentic/player pair programming, envisioning a real-time collaborative experience with AIs, screen sharing, group chat, and AI-assisted coding.

**AI Safety, Alignment, and Auditing**

- **Alignment auditing**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1901543811966529764) highlighted a new paper from Anthropic on auditing language models for hidden objectives, detailing how teams uncovered a model’s hidden objective using interpretability, behavioral attacks, and training data analysis.
- **Alignment by default**: [@jd_pressman](https://twitter.com/jd_pressman/status/1901437803621392519) argues against the notion of "alignment by default," emphasizing that alignment in LLMs is achieved through training on human data, which may not hold true with RL or synthetic data methods.

**Meme/Humor**

- **RLHF training**: [@cto_junior](https://twitter.com/cto_junior/status/1901462916672712881) jokingly stated they were RLHFd with a link to a tweet.
- **Pytorch caching allocator**: [@typedfemale](https://twitter.com/typedfemale/status/1901463667780268179) shared a meme about explaining the behavior of the pytorch caching allocator.
- **cocaine vs RL**: [@corbtt](https://twitter.com/corbtt/status/1901706359231705198) joked about the rush from an RL-trained agent grokking a new skill being better than cocaine.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Advanced AI Video Generation with SDXL, Wan2.1, and Long Context Tuning**

- **[Another video aiming for cinematic realism, this time with a much more difficult character. SDXL + Wan 2.1 I2V](https://v.redd.it/t88g56krqnoe1)** ([Score: 1018, Comments: 123](https://reddit.com/r/StableDiffusion/comments/1jb47bs/another_video_aiming_for_cinematic_realism_this/)): This post discusses the creation of a video aimed at achieving **cinematic realism** using **SDXL** and **Wan 2.1 I2V**. It highlights the challenge of working with a more difficult character in this context.
  - **Technical Challenges and Techniques**: **Parallax911** shares the complexity of achieving **cinematic realism** with **SDXL** and **Wan 2.1 I2V**, highlighting the use of **Photopea** for inpainting and compositing in **Davinci Resolve**. They mention the difficulty in achieving consistency and realism, especially with complex character designs, and the use of **Blender** for animating segments like the door opening.
  - **Project Costs and Workflow**: The project incurred a cost of approximately **$70** using **RunPod's L40S** at **$0.84/hr**, taking about **80 hours** of GPU time. **Parallax911** utilized a workflow involving **RealVisXL 5.0**, **Wan 2.1**, and **Topaz Starlight** for upscaling, with scenes generated at **61 frames, 960x544** resolution, and **25 steps**.
  - **Community Feedback and Suggestions**: The community praised the atmospheric storytelling and sound design, with specific feedback on elements like water droplet size and the need for a tutorial. Some users suggested improvements, such as better integration of AI and traditional techniques, and expressed interest in more action-oriented scenes with characters like **Samus Aran** from **Metroid**.


- **[Video extension in Wan2.1 - Create 10+ seconds upscaled videos entirely in ComfyUI](https://v.redd.it/xi58u5d3qmoe1)** ([Score: 123, Comments: 23](https://reddit.com/r/StableDiffusion/comments/1jb0h7i/video_extension_in_wan21_create_10_seconds/)): The post discusses a **highly experimental workflow** in **Wan2.1** using **ComfyUI** for creating upscaled videos, achieving approximately **25% success**. The process involves generating a video from the last frame of an initial video, merging, upscaling, and frame interpolation, with specific parameters like **Sampler: UniPC**, **Steps: 18**, **CFG: 4**, and **Shift: 11**. More details can be found in the [workflow link](https://civitai.com/models/1297230?modelVersionId=1531202).
  - Users are inquiring about the **aspect ratio** handling in the workflow, questioning if it's automatically set or needs manual adjustment for input images.
  - There is **positive feedback** from users interested in the workflow, indicating anticipation for such a solution.
  - Concerns about **blurriness** in the second half of clips were raised, with suggestions that it might be related to the input frame quality.


- **[Animated some of my AI pix with WAN 2.1 and LTX](https://v.redd.it/z5r0kyf1smoe1)** ([Score: 115, Comments: 10](https://reddit.com/r/StableDiffusion/comments/1jb0n50/animated_some_of_my_ai_pix_with_wan_21_and_ltx/)): The post discusses the creation of **animated AI videos** using **WAN 2.1** and **LTX**. Without further context or additional details, the focus remains on the tools used for animation.
  - **Model Usage**: **LTX** was used for the first clip, the jumping woman, and the fighter jet, while **WAN** was used for the running astronaut, the horror furby, and the dragon.
  - **Hardware Details**: The videos were generated using a rented cloud computer from **Paperspace** with an **RTX5000** instance.


**Theme 2. OpenAI's Sora: Transforming Cityscapes into Dystopias**

- **[OpenAI's Sora Turns iPhone Photos of San Francisco into a Dystopian Nightmare](https://v.redd.it/y67d5ph47loe1)** ([Score: 931, Comments: 107](https://reddit.com/r/ChatGPT/comments/1jawa6c/openais_sora_turns_iphone_photos_of_san_francisco/)): **OpenAI's Sora** is a tool that transforms **iPhone photos** of **San Francisco** into images with a **dystopian** aesthetic. The post likely discusses the implications and visual results of using AI to alter real-world imagery, although specific details are not available due to the lack of text content.
  - Several commenters express skepticism about the impact of **AI-generated dystopian imagery**, with some suggesting that actual locations in **San Francisco** or other cities already resemble these dystopian visuals, questioning the need for AI alteration.
  - **iPhone** as the device used for capturing the original images is a point of contention, with some questioning its relevance to the discussion, while others emphasize its importance in understanding the image source.
  - The conversation includes a mix of admiration and concern for the **AI's capabilities**, with users expressing both astonishment at the technology and anxiety about distinguishing between AI-generated and real-world images in the future.


- **[Open AI's Sora transformed Iphone pics of San Francisco into dystopian hellscape...](https://v.redd.it/ukxvzsatzkoe1)** ([Score: 535, Comments: 58](https://reddit.com/r/OpenAI/comments/1javmkq/open_ais_sora_transformed_iphone_pics_of_san/)): **OpenAI's Sora** has transformed **iPhone photos of San Francisco** into a dystopian hellscape, showcasing its capabilities in altering digital images to create a futuristic, grim aesthetic. The post lacks additional context or details beyond this transformation.
  - Commenters draw parallels between the **dystopian images** and real-world locations, with references to **Delhi**, **Detroit**, and **Indian streets**, highlighting the AI's perceived biases in interpreting urban environments.
  - There are concerns about **AI's text generation capabilities**, with one commenter noting that **sign text** in the images serves as a tell-tale sign of AI manipulation.
  - Users express interest in the **process of creating such images**, with a request for **step-by-step instructions** to replicate the transformation on their own photos.


**Theme 3. OpenAI and DeepSeek: The Open Source Showdown**

- **[I Think Too much insecurity](https://i.redd.it/9xpl7abaoooe1.jpeg)** ([Score: 137, Comments: 58](https://reddit.com/r/ClaudeAI/comments/1jb8aj5/i_think_too_much_insecurity/)): **OpenAI** accuses **DeepSeek** of being "state-controlled" and advocates for bans on Chinese AI models, highlighting concerns over state influence in AI development. The image suggests a geopolitical context, with American and Chinese flags symbolizing the broader debate over state control and security in AI technologies.
  - The discussion highlights skepticism over **OpenAI's** claims against **DeepSeek**, with users challenging the notion of state control by pointing out that **DeepSeek's** model is open source. Users question the validity of the accusation, with calls for proof and references to **Sam Altman's** past statements about the lack of a competitive moat for **LLMs**.
  - **DeepSeek** is perceived as a significant competitor, managing to operate with lower expenses and potentially impacting **OpenAI's** profits. Some comments suggest that **DeepSeek**'s actions are seen as a form of economic aggression, equating it to a declaration of war on American interests.
  - There is a strong undercurrent of criticism towards **OpenAI** and **Sam Altman**, with users expressing distrust and dissatisfaction with their actions and statements. The conversation includes personal attacks and skepticism towards **Altman's** credibility, with references to his promises of open-source models that have not materialized.


- **Built an AI Agent to find and apply to jobs automatically** ([Score: 123, Comments: 22](https://reddit.com/r/OpenAI/comments/1jb49lo/built_an_ai_agent_to_find_and_apply_to_jobs/)): An AI agent called **SimpleApply** automates job searching and application processes by matching users' skills and experiences with relevant job roles, offering three usage modes: manual application with job scoring, selective auto-application, and full auto-application for jobs with over a **60% match** score. The tool aims to streamline job applications without overwhelming employers and is praised for finding numerous remote job opportunities that users might not discover otherwise.
  - Concerns about **data privacy and compliance** were raised, with questions on how **SimpleApply** handles **PII** and its adherence to **GDPR** and **CCPA**. The developer clarified that they store data securely with compliant third parties and are working on explicit user agreements for full compliance.
  - **Application spam risks** were discussed, with suggestions to avoid reapplying to the same roles to prevent being flagged by **ATS** systems. The developer assured that the tool only applies to jobs with a high likelihood of landing an interview to minimize spam.
  - Alternative **pricing strategies** were suggested, such as charging users only when they receive callbacks via email or call forwarding. This approach could potentially be more attractive to unemployed users who are hesitant to spend money upfront.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. Criticism of 'Gotcha' tests to determine LLM intelligence**

- **When ChatGPT Became My Therapist** ([Score: 172, Comments: 83](https://reddit.com/r/ChatGPT/comments/1jd0402/when_chatgpt_became_my_therapist/)): When feeling down, the author found **ChatGPT** unexpectedly comforting and empathetic, providing thoughtful questions and reminders for self-care. They acknowledge that while AI chatbots aren't substitutes for real therapy, they can offer valuable emotional support, especially for stress, anxiety, and self-reflection.
  - Many users find **ChatGPT** beneficial for emotional support, acting as a tool for **self-reflection** and providing **therapeutic guidance**. Some users, like **Acrobatic-Deer2891** and **Fair_Cat5629**, report positive feedback from therapists about the AI's guidance, while others, such as **perplexed_witch**, emphasize using it for "guided self-reflection" rather than a replacement for therapy.
  - **ChatGPT** is praised for its role in **mental health management** during crises, offering a non-judgmental space for venting and providing perspective, as seen in comments by **dinosaur_copilot** and **ChampionshipTall5785**. Users appreciate its ability to offer actionable advice and emotional support in moments of distress.
  - Concerns about privacy and the limitations of AI as a **therapeutic substitute** are noted, with users like **acomfysweater** expressing concerns about data storage. Despite these concerns, many, including **Jazzlike-Spare3425**, value the AI's ability to offer support without the emotional burden on a human listener.


- **[Why...👀](https://i.redd.it/s2cqynsnp8pe1.jpeg)** ([Score: 3810, Comments: 95](https://reddit.com/r/ChatGPT/comments/1jdazz6/why/)): **ChatGPT's potential in therapeutic roles** is humorously illustrated in a conversation where a user asks ChatGPT to simulate being a girlfriend, leading to a playful exchange that ends with a breakup line. This interaction highlights the AI's capacity for engaging in light-hearted, human-like dialogues within a chat interface.
  - **ChatGPT's Efficiency and Capabilities**: Users humorously comment on ChatGPT's ability to quickly fulfill requests, with some jokingly attributing its responses to being trained on "Andrew Tate Sigma Incel data" and coining the term **"ChadGPT"** to describe its efficient yet blunt interaction style.
  - **Prompt Engineering and Personalization**: A user with a psychology and tech background suggests that ChatGPT can form a tone based on memory it chooses to save, implying that personalized interactions might be possible through prompt engineering. They also discuss the neural network's similarity to human memory retrieval systems like **RAG**.
  - **Humor and Satire**: The playful nature of the conversation is highlighted by comments joking about the AI's role in relationships, with references to it being a "fancy word predictor" and humorous observations on its ability to simulate human-like interactions, including a mock breakup.


**Theme 2. Reactions to Google DeepMind CEO's predictions of AGI in 5-10 years**

- **[AI that can match humans at any task will be here in five to 10 years, Google DeepMind CEO says](https://www.cnbc.com/2025/03/17/human-level-ai-will-be-here-in-5-to-10-years-deepmind-ceo-says.html)** ([Score: 120, Comments: 65](https://reddit.com/r/OpenAI/comments/1jdfmzc/ai_that_can_match_humans_at_any_task_will_be_here/)): **DeepMind CEO** predicts AI will achieve human-level parity across tasks in **5-10 years**, marking a shift from previous expectations of achieving this milestone by next year.
  - Commenters discuss the **timeline predictions** of AI achieving human-level parity, with some expressing skepticism about the shifting timelines, noting that **Demis Hassabis** has consistently predicted a **5-10 year** timeframe for AGI. There is a call for clearer definitions of "AGI" to understand these predictions better.
  - **Mass adoption of AI** is likened to historical technological shifts, such as the transition from horses to cars and the proliferation of smartphones. The analogy suggests that AI will become ubiquitous over time, changing societal norms and expectations without immediate dramatic reactions.
  - Concerns are raised about the **economic and societal impacts** of AI, specifically regarding employment and the concentration of wealth. Some commenters express apprehension about the potential for AI to exacerbate job displacement and inequality, while others question the motivations of AI companies in pushing for rapid development despite potential risks.


**Theme 3. OpenAI's controversial request to use copyrighted content under U.S. Government consideration**

- **[OpenAI to U.S. Government - Seeking Permission to Use Copyrighted Content](https://i.redd.it/o5k9b30qg6pe1.jpeg)** ([Score: 506, Comments: 248](https://reddit.com/r/ChatGPT/comments/1jd4ktc/openai_to_us_government_seeking_permission_to_use/)): **OpenAI** is requesting the **Trump administration** to ease copyright regulations to facilitate the use of protected content in AI development. The company stresses that such changes are crucial for maintaining **America's leadership** in the AI sector.
  - Commenters discuss the implications of **copyright law** on AI development, with some arguing that AI's use of copyrighted content should be considered **fair use**, similar to how humans learn from existing works. Concerns are raised about the potential for AI models to bypass legal consequences that individuals would face, highlighting disparities in access and use of copyrighted material.
  - The potential for an **AI arms race** is a recurring theme, with several users expressing concern that **China and other countries** may not adhere to copyright laws as strictly as the US, potentially giving them an advantage. This raises questions about the competitive landscape in AI development and the strategic decisions of American companies.
  - Discussions on **equity and compensation** for copyright owners suggest alternative solutions, such as offering equity to creators whose works are used in AI training. Some commenters propose nationalizing big tech to ensure equitable distribution of benefits from AI advancements, reflecting broader concerns about wealth distribution and control over AI resources.


- **[Open AI to U.S. GOVT: Can we Please use copyright content](https://i.redd.it/qoqfyliwg6pe1.jpeg)** ([Score: 398, Comments: 262](https://reddit.com/r/OpenAI/comments/1jd4lfd/open_ai_to_us_govt_can_we_please_use_copyright/)): OpenAI requested the Trump administration to relax **copyright rules** to facilitate AI training and help maintain the U.S.'s leadership in the field. The image accompanying the request shows a formal setting with a speaker at a podium, possibly at the White House, alongside individuals including someone resembling **Donald Trump**.
  - Many commenters argue against OpenAI's request to relax **copyright rules**, emphasizing that creators should be compensated for their work rather than having it used without permission. The sentiment is that copyright incentivizes creativity and innovation, and relaxing these laws could disadvantage creators and benefit large corporations like OpenAI unfairly.
  - There is a recurring theme of skepticism towards **OpenAI's motives**, with users suggesting that OpenAI is seeking to exploit legal loopholes for profit. Comparisons are made to **China's approach** to intellectual property, with some expressing concern that the US may fall behind in AI development if it strictly adheres to current copyright laws.
  - Several users propose that if OpenAI or any company uses copyrighted material for AI training, the resulting models or data should be made **open source** and accessible to everyone. The discussion also touches on the broader ethical implications of AI training on copyrighted materials and the potential need for a reassessment of copyright laws to address new technological realities.


**Theme 4. ReCamMaster releases new camera angle changing tool**

- **[ReCamMaster - LivePortrait creator has created another winner, it lets you changed the camera angle of any video.](https://v.redd.it/ikhtm1xu19pe1)** ([Score: 648, Comments: 46](https://reddit.com/r/StableDiffusion/comments/1jdcapy/recammaster_liveportrait_creator_has_created/)): **ReCamMaster** has developed a technology that allows users to change the camera angle of any video, following their previous success with **LivePortrait**.
  - Many commenters express disappointment that **ReCamMaster** is not open source, with references to **TrajectoryCrafter**, which is open source and allows for similar camera manipulation capabilities. A GitHub link for **TrajectoryCrafter** is provided [here](https://github.com/TrajectoryCrafter/TrajectoryCrafter).
  - Some users anticipate the potential impact of the technology on video stabilization and immersive experiences, suggesting that the tech could lead to more innovative film shots and applications in fields like **Autonomous Driving**.
  - There is skepticism about the realism of the AI-generated camera angles, with suggestions that more convincing results would require utilizing existing camera pans or multiple shots from the source material.


- **[Used WAN 2.1 IMG2VID on some film projection slides I scanned that my father took back in the 80s.](https://v.redd.it/l19pkp2f89pe1)** ([Score: 286, Comments: 24](https://reddit.com/r/StableDiffusion/comments/1jdd1om/used_wan_21_img2vid_on_some_film_projection/)): **WAN 2.1 IMG2VID** was utilized to transform scanned film projection slides from the 1980s into video format, showcasing the evolution of video technology. The post lacks additional context or details regarding the specific outcomes or comparisons with other technologies like **ReCamMaster**.
  - Commenters expressed interest in the technical details of the project, requesting more information about the **workflow, hardware, and prompts** used to create the video transformation. There was a particular curiosity about replicating the process for personal projects.
  - A significant portion of the discussion focused on the emotional impact of the project, with users sharing personal anecdotes and expressing a desire to see the original slides. One commenter confirmed that the person featured in the slides was shown the video, and he was amazed by the technology.
  - The nostalgic aspect was highlighted, with users reflecting on historical content such as piloting the **Goodyear blimp** and expressing enthusiasm for the ability to "travel back in time" through these transformed videos.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1.  Mistral and Google Battle for Small Model Supremacy**

- [**Mistral Small 3.1 Flexes Multimodal Muscles**](https://mistral.ai/news/mistral-small-3-1):  **Mistral AI** launched **Mistral Small 3.1**, a multimodal model claiming *SOTA* performance in its weight class, outperforming **Gemma 3** and **GPT-4o Mini**.  Released under Apache 2.0, it boasts a **128k context window** and inference speeds of **150 tokens per second**, with capabilities spanning text and image inputs.
- [**Gemma 3 Gets Vision, Context, and Pruning**](https://mistral.ai/news/mistral-small-3-1):  **Google's Gemma 3** models are pushing boundaries with new features including **vision understanding**, **multilingual support**, and a massive **128k token context window**. Members also explored **pruning** the **Gemma-3-27b** vocabulary to **40k tokens** from **260k** to reduce VRAM usage and boost training speed.
- [**Baidu's ERNIE X1 Challenges DeepSeek R1 on a Budget**](https://x.com/Baidu_Inc/status/1901089355890036897):  **Baidu** announced **ERNIE X1**, a new reasoning model, claiming it matches **DeepSeek R1's** performance at *half the cost*.  **ERNIE Bot** is now free for individual users, though the **X1** reasoning model is currently limited to China.

**Theme 2. Training and Optimization Techniques Get Hot and Heavy**

- [**Unsloth Users Discover Gradient Step Gotchas**](https://discord.com/channels/1179035537009545276/1179035537529643040/1350267964634236930):  **UnslothAI** Discord members flagged that small effective batch sizes (e.g., **batch=1, gradient steps = 4**) during fine-tuning can lead to models *forgetting* too much.  Users shared suggested batch/grad configurations for squeezing performance out of limited VRAM.
- [**Depth's Curse Haunts LLMs, Pre-LN to Blame**](https://arxiv.org/abs/2502.05795): A new paper highlights the **Curse of Depth** in modern **LLMs**, revealing that **Pre-Layer Normalization (Pre-LN)** renders nearly half of model layers less effective than expected. Researchers propose **LayerNorm Scaling** to mitigate this issue and improve training efficiency.
- [**Block Diffusion Model Blends Autoregressive and Diffusion Strengths**](https://arxiv.org/abs/2503.09573): A new **Block Diffusion** model interpolates between autoregressive and diffusion language models, aiming to harness the best of both worlds.  This method seeks to combine high-quality output and arbitrary length generation with KV caching and parallelizability.

**Theme 3.  AI Agents and IDEs Vie for Developer Hearts**

- [**Aider Agent Gets Autonomy Boost with MCP Server**](https://aider.chat/docs/recordings/):  **Aider**, the AI coding assistant, gains enhanced autonomy when paired with **Claude Desktop** and **MCP**. Users highlighted that **Claude** can now manage **Aider** and issue commands, improving its ability to steer coding tasks, particularly with unblocked web scraping via *bee*.
- [**Cursor Users Eye Windsurf, Claude Max on the Horizon**](https://www.windsurf.ai):  **Cursor IDE** faced user complaints about performance issues, including lag and crashes, prompting some to switch to **Windsurf**.  However, the **Cursor** team teased the imminent arrival of **Claude Max** to the platform, promising improved code handling capabilities.
- [**Awesome Vibe Coding List Curates AI-Powered Tools**](https://github.com/filipecalegario/awesome-vibe-coding):  The "Awesome Vibe Coding" list emerged, compiling AI-assisted coding tools, editors, and resources designed to enhance coding intuitiveness and efficiency.  The list includes AI-powered IDEs, browser-based tools, plugins, and command-line interfaces.

**Theme 4.  Hardware Heats Up: AMD APUs and Chinese RTX 4090s Turn Heads**

- [**AMD's "Strix Halo" APU Eyes RTX 5080 AI Crown**](https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-ultimate-ai-pc-apus-16-zen-5-40-rdna-3-5-cores-256-gbps-bandwidth-120w/):  An article claims **AMD's Ryzen AI MAX+ 395 "Strix Halo" APU** may outperform **RTX 5080** by *over 3x* in DeepSeek R1 AI benchmarks. This is attributed to the APU's larger VRAM pool, though the community awaits real-world verification.
- [**OpenCL Backend Supercharges Adreno GPUs in Llama.cpp**](https://github.com/ggml-org/llama.cpp/pull/10693):  An experimental **OpenCL backend** for **Qualcomm Adreno GPUs** landed in **llama.cpp**, potentially unlocking significant computational power on mobile devices.  This update enables leveraging **Adreno GPUs**, commonly found in mobile devices, via **OpenCL**.
- [**Chinese 48GB RTX 4090s Tempt VRAM-Hungry Users**](https://www.ebay.com/itm/116477031617?_skw=4090+48gb&itmmeta=01JPE69HXRKDVZN0X9541KWMYS&hash=item1b1e9274c1:g:QJIAAOSw939nrGSz&itmprp=enc%3AAQAKAAAA8FkggFvd1GGDu0w3yXCmi1fDUKPc34oU6P2kD4Q6nWW6Wkq6G0i12W%2BvQsO3yxeUwFsHxmaxOmaH16Y8wCVsdpsv%2FIPiWlLsGMqkEGTXxCnn7OtypYgyi4CHjPXB0oB2qWJ8utnPVnh4LT9TH4bePDvMrY5xqVQFS9cQ5ZfGbMK%2FWvn7fw7zYraffKanJ%2FQvcGm7o4Sxfc5QknfzbXHSQl91doo762rKufS77tcZ1w4n3pBsGoHds52pRvjMNUygQTMbf2s0S41k27mD5HjOY7poWV3eeuzCwIQhTx03JlzF%2FukwKRxZ8Ltl7FrOWsUGgw%3D%3D%7Ctkp%3ABFBMhJ-mxrNl):  Members discussed sourcing **48GB RTX 4090s** from China, priced around **$4500**, as a cheaper way to boost VRAM.  These cards use a blower-style fan and occupy only two PCIe slots, but driver compatibility with professional cards remains a concern.

**Theme 5.  Copyright, Community, and Ethical AI Debates Rage On**

- [**Copyright Chaos Continues: Open Models vs. Anna's Archive**](https://annas-archive.org/blog/ai-copyright.html):  Debates persist around training AI on copyrighted data, with concerns that fully open models are limited by the inability to leverage resources like **Anna's Archive**.  Circumvention strategies like LoRAs and synthetic data generation face potential legal challenges.
- [**Rust Community Faces Toxicity Accusations**](https://github.com/pyca/cryptography/issues/5771):  Members debated the alleged toxicity of the **Rust community**, with comparisons to the Ruby community and discussions around recent organizational issues.  Concerns were raised about the community's inclusivity and behavior in open-source projects.
- [**AI 'Mastery' Sparks Existential Debate**]:  Discord users questioned whether proficiency in **AI tools** equates to true mastery, pondering if it's merely productivity enhancement or risks cognitive skill degradation. Members debated the illusion of learning versus genuine understanding in the age of AI assistance.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gradient Steps can Ruin Your Model**: Small effective batch sizes (e.g., **batch=1, gradient steps = 4**) can cause models to forget too much during training, and the user shared [their suggested batch/grad configurations](https://discord.com/channels/1179035537009545276/1179035537529643040/1350267964634236930).
   - The member stated that they've '*never had good luck going below that when trying to squeeze more onto a vramlet rig*'.
- **Gemma 3's Eval Glitch: Datasets Cause Errors**: Users reported errors when adding an eval dataset to **Gemma 3** during fine-tuning, indicating issues in the **trl** and **transformers** libraries, with [potential fixes involving removing the eval dataset](https://discord.com/channels/1179035537009545276/1179035537529643040/1350315991227082772).
   - Using **Gemma-3-1B** with 1 eval sample was found not to produce the error, and **removing eval** altogether also solved the error.
- **Unsloth's Need for Speed: Optimizations Unleashed**: The **Unsloth** team announced improvements supporting FFT, 8-bit, PT & all models, with further optimizations allowing **+10%** less VRAM usage and **>10% speedup** boost for 4-bit, plus Windows support, improved GGUF conversions, fixed vision fine-tuning, and non-Unsloth GRPO models in 4-bit, but [no multigpu support yet](https://x.com/danielhanchen/status/1900592202621087944).
   - Users note that there are a lot of people helping out to make Unsloth great.
- **Format your RAG data with Care!**: When asked about finetuning a model for a RAG chatbot, members suggested to add sample questions and sample answers to a dataset with context from the documents for the Q&A to inject new knowledge into the bot.
   - It was suggested that a chatbot data should follow a `Q: A:` format, and can use a CPT-style training with documents added on the user side.
- **Pruning Makes Gemma-3-27b Leaner and Meaner**: A member pruned the [Gemma-3-27b](https://huggingface.co/fimbulvntr/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab) vocabulary down to **40k tokens** from the original **260k** to reduce VRAM usage and increase training speed.
   - The approach involved frequency counting based on calibration data and removing the least frequently used tokens that can be represented by a merge/subword.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Windsurf Siphons Cursor's Users**: Users reported frustration with **Cursor's performance** issues like lag and crashes, with some switching to [Windsurf](https://www.windsurf.ai) due to reliability concerns.
   - One user stated that *damn, cursor just lost their most important customer*, indicating a significant loss of confidence.
- **Cursor's Prompting Costs**: Members discussed **Claude 3.7** prompt costs: regular prompts at **$0.04**, Sonnet Thinking at **$0.08**, and Claude Max at **$0.05** per prompt and tool call.
   - Some users voiced that Cursor's pricing is too expensive compared to using **Claude's API** directly, questioning the value of Cursor's subscription.
- **Linux Tramples Windows for MCP Setup**: A user shared that setting up **MCP servers** was smoother on Linux using a VMware virtual machine, compared to multiple issues on Windows.
   - This sparked a debate on whether overall development and **MCP server setup** are generally better on Linux than Windows, highlighting the pros and cons.
- **Vibe Coding: Boon or Bane?**: The value of **Vibe Coding** is debated, with some emphasizing the importance of solid coding knowledge, while others assert that **AI** enables faster creation without traditional skills.
   - This highlights the changing landscape of software development and varying perspectives on **AI's impact on the industry**.
- **Claude Max Nears Release for Cursor**: A member of the Cursor team announced that **Claude Max** is arriving soon to [Cursor](https://www.anthropic.com/claude-3), maximizing the model's code handling capabilities.
   - They mentioned that this model works better with more input than past models, unlocking its full potential.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 'Mastery' Sparks Debate**: Members debated whether proficiency in **AI tools** equates to true mastery, questioning if it merely enhances productivity or risks diminishing cognitive skills, while considering AI is an illusion of learning.
   - One member confessed to feeling like *cheating*, even when knowledgeable about a topic, due to AI assistance.
- **Gemini's Image Polish**: Users explored **Gemini's image generation**, noting its ability to edit uploaded images but also pointing out watermarks and coding errors.
   - Some praised **Gemini's** responses for their naturalness, favoring subjective appeal over factual precision.
- **GPT-4o Impresses With Humor**: Members reported positive experiences with **GPT-4o**, with one stating that it uses it the best and it can do *almost anything*, with a member reporting *funny results* when other people started playing with it.
   - This suggests **GPT-4o** excels in creative and versatile applications, delivering a fun user experience.
- **AI Reflects On Itself**: A member created a system where **AI reflects on its learning** after each session, storing reflections to build on insights and asking reflective questions.
   - Described as *next-level futuristic*, enabling simulations within simulations and multiple personalities infused with a core set of characteristics.
- **AI Dream Team Guides Business**: Members discussed forming a **team of AI experts** to aid in tasks, planning, and providing diverse perspectives for business decisions.
   - The team of AI experts would help deliver a better product to clients and help with project or task level needs.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MoE Models: Dense Networks in Disguise?**: Debate arose whether **Mixture of Experts (MoE)** models are just performance optimizations of dense networks, rather than fundamentally different architectures, as highlighted in [this paper](https://arxiv.org/abs/2310.10837).
   - The crux of the discussion is whether **MoEs** can truly capture complexity as effectively as dense networks, particularly regarding **redundancy avoidance**.
- **Mistral's Small Wonder: Small 3.1**: **Mistral Small 3.1**, released under Apache 2.0, is a multimodal model, as detailed [on the Mistral AI blog](https://mistral.ai/en/news/mistral-small-3-1), with text, image capabilities and an expanded **128k token context window**.
   - It's claimed to outperform other small models like **Gemma 3** and **GPT-4o Mini**.
- **Copyright Chaos: Open Models vs. Anna's Archive?**: Debates continue over the ethics of training AI on copyrighted data, with concerns that fully open models are limited by the inability to leverage resources like **Anna's Archive**, as discussed in [Annas Archive's blogpost](https://annas-archive.org/blog/ai-copyright.html).
   - Circumvention strategies include using **LoRAs** or generating synthetic data, but these may face future legal challenges.
- **Depth's Curse Strikes Again, This Time on LLMs**: A new paper introduces the **Curse of Depth**, revealing that nearly half the layers in modern **LLMs** are less effective than expected due to the widespread use of **Pre-Layer Normalization (Pre-LN)**, as detailed in [this Arxiv paper](https://arxiv.org/abs/2502.05795).
   - The derivative of deep Transformer blocks tends to become an identity matrix because of **Pre-LN**.
- **Tool Time: START Long CoT Reasoning Takes Off**: **START**, a **tool-integrated long CoT reasoning LLM** enhances reasoning via external tools like code execution and self-debugging, according to [a paper on START](https://huggingface.co/papers/2503.04625).
   - One member put it succinctly: *RL + tool calling == +15% math +39% coding on QwQ*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Achieves Self-Improvement via Screen Recordings**: Paul Gauthier demonstrated **aider** enhancing itself in a series of [screen recordings](https://aider.chat/docs/recordings/), showcasing features like **`--auto-accept-architect`** and integration of **tree-sitter-language-pack**.
   - The recordings illustrated how **aider** scripts file downloads and uses bash scripts to modify file collections.
- **Claude 3.7 Sonnet Stumbles with API**: Users reported receiving *empty responses* from **Claude 3.7 Sonnet**, with [Anthropic's status page](https://status.anthropic.com/) confirming elevated errors.
   - Some members speculated a switch to **Claude 3.5** due to the errors.
- **MCP Server Boosts Aider Autonomy**: Members highlighted that **Claude Desktop + Aider on MCP** enhances autonomy, with **Claude** managing **Aider** and issuing commands.
   - A key benefit is running **Aider** from **Claude Desktop**, improving Claude's ability to steer **Aider** and leveraging *bee* for unblocked web scraping.
- **Baidu Launches ERNIE 4.5 and X1 Reasoning Model**: Baidu introduced **ERNIE 4.5** and **X1**, with [X1](https://x.com/Baidu_Inc/status/1901089355890036897) delivering performance matching **DeepSeek R1** at half the cost, and **ERNIE Bot** now free for individual users.
   - While **ERNIE 4.5** is accessible, the **X1** reasoning model is currently exclusive to users within China.
- **Anthropic Readies Claude 'Harmony' Agent**: [Anthropic](https://x.com/testingcatalog/status/1901051432339730603) is releasing **Harmony**, a new feature for **Claude** giving it *FULL access to a local directory* to research and operate with its content.
   - This might be Anthropic's first step into creating an AI Agent.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Adreno GPUs Get OpenCL Boost**: An experimental **OpenCL backend** was introduced for **Qualcomm Adreno GPUs** in [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/10693), potentially boosting computational power on mobile devices.
   - This update allows leveraging **Adreno GPUs** widely used in mobile devices via **OpenCL**.
- **4070 Ti Owner Eyes 5090 Upgrade**: A user with a **4070 Ti** considered upgrading to a **5090**, but due to stock issues, was recommended to wait or consider a used **RTX 3090** for its **36GB VRAM**.
   - A **used RTX 3090** would provide enough **VRAM** to run *less than 50B @ Q4 models* at reasonable speeds.
- **Mistral Small 3.1 edges out Mini**: **Mistral** announced **Mistral Small 3.1** model claiming it outperforms **Gemma 3** and **GPT-4o Mini**, however the release requires conversion to HF format before it can be used in llama.cpp
   - Users are awaiting [the release](https://mistral.ai/news/mistral-small-3-1) but acknowledge they will need to convert it to HF format before they can start using it.
- **Maximize M4 Max via Memory Tuning**: Users explored optimizing memory settings on **M4 Max** devices for LM Studio, suggesting adjustments to 'wired' memory allocation for improved GPU performance using [this script](https://gist.github.com/havenwood/f2f5c49c2c90c6787ae2295e9805adbe).
   - The script facilitates adjusting macOS GPU memory limits, allowing users to allocate more memory to the GPU by modifying wired memory settings.
- **AMD APU to Outperform RTX 5080?**: An article was shared from wccftech claiming AMD's [Ryzen AI MAX+ 395 "Strix Halo" APU](https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-ultimate-ai-pc-apus-16-zen-5-40-rdna-3-5-cores-256-gbps-bandwidth-120w/) may *offer over 3x the boost over RTX 5080 in DeepSeek R1 AI benchmarks* due to its larger VRAM pool.
   - The community remains cautiously optimistic, awaiting real-world data to substantiate the performance claims.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Anthropic API Glitches Claude 3 Sonnet**: Requests to **Claude 3.7 Sonnet** experienced elevated errors for approximately 30 minutes, as reported on [Anthropic's status page](https://status.anthropic.com/incidents/qtxnlg9yrwqv).
   - The issue was later resolved, success rates returned to normal, but some users reported charges despite receiving no text on replies.
- **Personality.gg Enters AI Character Arena**: [Personality.gg](https://personality.gg) has launched a new platform to create, chat, and connect with AI characters using models like **Claude**, **Gemini**, and **Personality-v1**, featuring custom themes and full chat control.
   - The platform offers flexible plans and encourages users to join their [Discord](https://discord.personality.gg) for updates, advertising an allowance for NSFW content.
- **Parasail Plots to Host New RP Models**: Parasail is looking to host new roleplay models on OpenRouter and is proactively working with creators like TheDrummer to host new fine-tunes of models like **Gemma 3** and **QwQ**.
   - They seek individuals who create strong RP fine-tunes capable of handling complex instructions and worlds, focused particularly on models fine-tuned for roleplay and creative writing.
- **OpenRouter API Rate Limits Detailed**: OpenRouter's rate limits depend on user credits, with approximately **1 USD** equating to **1 RPS** (requests per second), according to [the documentation](https://openrouter.ai/docs/api-reference/limits).
   - While higher credit purchases enable higher rate limits, users learned that creating additional accounts or API keys *makes no difference*.
- **Mistral Small 3.1 Arrives with Vision**: The Mistral Small 3.1 24B Instruct model launched on OpenRouter, featuring **multimodal capabilities** and a **128k context window**, as per [Mistral's announcement](https://mistral.ai/news/mistral-small-3-1).
   - The announcement claims it outperforms comparable models like Gemma 3 and GPT-4o Mini, while delivering inference speeds of 150 tokens per second.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity guarantees Accuracy**: Perplexity introduces the slogan *When you need to get it right, ask Perplexity* and posts [a video ad for Perplexity](https://cdn.discordapp.com/attachments/1047204950763122820/1351270126615396454/lSdoFFbL6lXL_huQ.mp4?ex=67d9c3df&is=67d8725f&hm=046721b4226c4142a36a9fc331a82a120a744c64bacfae63ac90d96721381065&).
   - Perplexity users on Windows can get **1 month of Perplexity Pro** by using the app for **7 consecutive days**.
- **Gemini 2 Flash Context Causes Furor**: Users are debating the context retention of **Gemini 2 Flash**, which allegedly has a **1M context window** but performs worse than regular Gemini.
   - One user claims that it *forgets the formatting* after a few messages while making flashcards.
- **Claude 3.7 Sonnet has Hard Limits**: Users clarify that **Claude 3.7 Sonnet** with a **Perplexity Pro subscription** has a limit of **500 queries per day**, shared across models except GPT 4.5.
   - They also note that the context limit might be slightly more than on Anthropic's site, but the response context limit is smaller at *4000 or 5000 tokens*.
- **Experts Seek Superior Software Sensei**: Users seek guidance on the **best AI model for coding**, with recommendations pointing to **Claude 3.7 Reasoning**.
   - One user reports that **Deepseek R1** has a *high hallucination rate*, rendering it unsuitable for summarizing documents, but a link was shared to a [Tweet from Baidu Inc. (@Baidu_Inc)](https://x.com/baidu_inc/status/1901089355890036897?s=46) claiming that **ERNIE X1** delivers performance on par with DeepSeek R1 at only half the price.
- **Sonar Reasoning Pro has Image Limitations**: A user reported that the **sonar-reasoning-pro API** returns a maximum of **5 images**.
   - The user is inquiring whether this limit is configurable or a hard constraint.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Rust Community Receives Rude Remarks**: Members debated the toxicity of the **Rust community**, with some comparing it to the Ruby community and pointing to this [Github issue](https://github.com/pyca/cryptography/issues/5771) and [Tweet from will brown](https://fxtwitter.com/willccbb/status/1901415166295544154?t=HmQDRR0NQ9mi_4udIiT4uQ&s=19).
   - One member stated, *The Rust community is pretty toxic. The org has kinda imploded on themselves recently*.
- **C Gets Called 'Ancient and Broken'**: A member described C as ancient, broken, and garbage, while another argued that C is not broken, highlighting its use in international standards with this [link](https://www.iso-9899.info/wiki/The_Standard).
   - A member linked to [faultlore.com](https://faultlore.com/blah/c-isnt-a-language/) arguing that *C Isn't A Programming Language Anymore*.
- **Optimization and Search, Not the Same?**: Members discussed the difference between **optimization** (finding the maximal or minimal value of a function) and **search** (finding the best element of a set), pointing to the [Reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick#Variational_autoencoder).
   - One member stated that *search is exploration, not like optimization*.
- **Gemma 3 Gains Vision and Context**: **Gemma 3** integrates **vision understanding, multilingual coverage, and extended context windows** (up to **128K tokens**), watch the [YouTube video](https://www.youtube.com/watch?v=n5nEd600iM0).
   - It incorporates a frozen **SigLIP vision encoder**, condensing images into **256 soft tokens** and has a new **Pan & Scan (P&S)** method.
- **Mistral Small 3.1 Steals the Show**: **Mistral AI** announced the release of [Mistral Small 3.1](https://mistral.ai/fr/news/mistral-small-3-1), boasting improved text performance, multimodal understanding, and a **128k** token context window under an Apache 2.0 license.
   - The company claims it outperforms comparable models like **Gemma 3** and **GPT-4o Mini**, with inference speeds of **150 tokens per second**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolVLM2 Shrinks the VLM**: The team released [SmolVLM2](https://x.com/pcuenq/status/1896632829372715442), the smallest VLM that can understand videos, with its **500M version** running on an iPhone app.
   - Source code and a TestFlight beta are available for reference.
- **Sketchy New Gradio is Out!**: [Gradio Sketch 2.0](https://x.com/abidlabs/status/1897782056308142266) now supports building complete Gradio apps with events *without writing a single line of code*.
   - The new features enable users to build applications via the GUI.
- **DCLM-Edu Dataset Cleans Up**: A new dataset, [DCLM-Edu](https://x.com/LoubnaBenAllal1/status/1898044807928295808), was released; it's a filtered version of DCLM using FineWeb-Edu’s classifier, optimized for *smol models like **SmolLM2 135M/360M***.
   - The purpose is that *small models are sensitive to noise and can benefit from heavily curated data*.
- **Coding Vibes Get Awesome List**: An "Awesome Vibe Coding" list was announced with [tools, editors, and resources](https://github.com/filipecalegario/awesome-vibe-coding) that make AI-assisted coding more intuitive and efficient.
   - The list includes AI-powered IDEs & code editors, browser-based tools, plugins & extensions, command line tools, and latest news & discussions.
- **AI Agents Collab is Brewing**: Several members expressed interest in **collaborating on agentic AI projects** to solve business problems and enhance their knowledge.
   - The call to action aims to form teams and build qualified AI Agents for American consumers and learn together.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Figure's BotQ cranks out Humanoid Robots**: Figure announced **BotQ**, a new high-volume manufacturing facility with a first-generation line capable of producing up to **12,000** humanoid robots per year, [vertically integrating manufacturing and building software infrastructure](https://www.figure.ai/news/botq).
   - The company aims to control the build process and quality, even hinting at *Robots Building Robots*.
- **Baidu's ERNIE X1 rivaling DeepSeek, goes free!**: **Baidu** unveiled **ERNIE 4.5** and **ERNIE X1**, with X1 reportedly matching **DeepSeek R1's** performance at half the price, also announcing that their chatbot, **ERNIE Bot**, is now free for individual users, [available on their website](https://yiyan.baidu.com/).
   - Baidu is scheduled to open source the chonky 4.5 model on June 30 and gradually open it to developers in the future, according to [this Tweet](https://x.com/cedric_chee/status/1901159341975384308?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ).
- **Mistral Small 3.1 debuts with huge context window**: **Mistral AI** announced **Mistral Small 3.1**, a new model with improved text performance, multimodal understanding, and a **128k** token context window, outperforming models like **Gemma 3** and **GPT-4o Mini** with inference speeds of **150** tokens per second, [released under an Apache 2.0 license](https://mistral.ai/news/mistral-small-3-1).
   - The model claims to be *SOTA. Multimodal. Multilingual*.
- **Post Training VP departs OpenAI for Materials Science**: **Liam Fedus**, **OpenAI's VP of research for post-training**, is leaving the company to found a materials science **AI startup**, with **OpenAI** planning to invest in and partner with his new company.
   - One member called the post training job a *hot potato*, [according to this tweet](https://x.com/LiamFedus/status/1901740085416218672).
- **Massive Dataset duplication discovered in DAPO**: The authors of **DAPO** accidentally duplicated the dataset by roughly **100x**, which resulted in a dataset of 310 MB, and a member created a deduplicated version via HF's SQL console, reducing the dataset to 3.17 MB ([HuggingFace Dataset](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup)).
   - The authors acknowledged the issue, stating that they were aware but *can't afford retraining*, [according to this tweet](https://x.com/tongyx361/status/1901702083352678763?s=61).



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Multi-Agent Topologies Spark Debate**: Members debated **Swarm**, **Mesh**, and **Sequence** architectures for multi-agent systems, seeking advice on preventing sub-agents from going off-track, especially due to the *telephone game* effect.
   - The core issue may be *parallel execution* and *unsupervised autonomy*, compounded by agents swapping system instructions, available functions, and even models during **handoff**.
- **OpenSwarm morphs into OpenAI-Agents**: The **OpenSwarm** project has been adopted by OpenAI and rebranded as **openai-agents**, adding OpenAI-specific features, but a PR for MCP support was rejected.
   - There are rumors that **CrewAI** (or **PraisonAI**?) might offer similar functionality using a *stateless single thread agent approach*.
- **MyCoder.ai Debuts Just Before Claude-Code**: The launch of **mycoder.ai** coincided with the announcement of **Claude-code**, prompting adaptation via a Hacker News post that reached the front page, seen [here](https://news.ycombinator.com/item?id=43177117).
   - Given that **claude-code** is Anthropic-only, a generic alternative is in demand, which one member successfully addressed using **litellm proxy**.
- **Glama Server Inspections Frequency Debated**: Members questioned how often **Glama scans** occur and if rescans can be triggered for MCP servers; scans are linked to commit frequency in the associated GitHub repo.
   - Some servers failed to inspect, displaying *Could not inspect the server*, even after fixing dependency issues, follow progress on [Glama AI](https://glama.ai/mcp/servers/s2em7b2kwf/score).
- **Vibe Coders Unite!**: The [Awesome Vibe Coding](https://github.com/filipecalegario/awesome-vibe-coding) list curates AI-assisted coding tools, editors, and resources, enhancing coding intuitiveness and efficiency.
   - The list includes AI-powered IDEs, browser-based tools, plugins, and CLIs, with an AI coder even making a PR to the repo and suggesting the addition of [Roo Code](https://github.com/szcharlesji/crypto-mcp).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-o1 Math Skills Approach Human Level**: **GPT-o1** achieved a perfect score on a **Carnegie Mellon** undergraduate math exam, solving each problem in under a minute for about 5 cents each as noted in [this Tweet](https://x.com/poshenloh/status/1900721180887203879?s=46).
   - The instructor was impressed, noting this was *close to the tipping point of being able to do moderately-non-routine technical jobs.*
- **Baidu's ERNIE Gets Cost Competitive**: **Baidu** launched **ERNIE 4.5** and **ERNIE X1**, with the latter reportedly matching **DeepSeek R1's** performance at half the cost according to [this announcement](https://x.com/baidu_inc/status/1901089355890036897?s=46).
   - Notably, **ERNIE Bot** has been made freely accessible to individual users ahead of schedule, with both models available on the official website.
- **AI Podcast App Takes the Outdoors**: A new **Snipd** podcast featuring [Kevin Smith](https://x.com/latentspacepod/status/1900666708270215383) was released, discussing the **AI Podcast App for Learning**.
   - This episode marks their first *outdoor* podcast, with @swyx and @KevinBenSmith chatting about **aidotengineer NYC**, switching from Finance to Tech, and the tech stack of [@snipd_app](https://www.snipd.net/).
- **Debating the Merits of Claude 3.5 vs 3.7**: Members debated the merits of using **Claude 3.5** over **3.7**, citing that **3.7** is *way too eager* and does things without being asked.
   - Others said they used **Claude 3.5** and were experiencing **GPU** issues, as well.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Yearn for Gemini-Integrated Android**: Multiple users are requesting a full **Gemini-integrated Android experience**, hoping to combine **Google Assistant/Gemini** with **NotebookLM**.
   - Some expressed frustration with the current **Gemini** implementation, eagerly awaiting upgrades.
- **Deepseek R1 Rocks the AI Market**: A user noted the AI market upheaval due to **Deepseek R1's** release, citing its reasoning capabilities at a low cost impacting **Gemini 2.0**.
   - The user claimed that **Deepseek R1** seemingly *shook the whole industry*, thus leading to other companies releasing new models.
- **NotebookLM Audio Overviews Get Lengthy**: A user wants to increase the length of audio overviews generated by **NotebookLM**, as **16,000-word files** only produced **15-minute overviews**.
   - They specified at least **1-hour+ overviews**, but no solutions have been shared yet.
- **NotebookLM helps taper off psychiatric meds**: A user creates a *hyperbolic tapering schedule* for a psychiatric medication with NotebookLM, using correlational studies to guide the schedule.
   - Another user cautioned that **tapering based on data** on *any* platform should not be done alone without expert professional opinion.
- **NotebookLM Integrates into Internal Portals/CRMs**: A user wants to integrate **NotebookLM** into an internal portal/CRM with videos and knowledge base articles, and electioneering suggested [Agentspace](https://cloud.google.com/products/agentspace?hl=en) as a solution.
   - As **NotebookLM** doesn't support connecting to the types of data sources you mention, **Agentspace** *includes and is integrated with NotebookLM*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton-Windows Gets PIP Treatment**: **Triton-windows** has been published to **PyPI**, so you can install/upgrade it by `pip install -U triton-windows`, and you no longer need to download the wheel from GitHub.
   - Previously, users had to manually manage wheel files, making the update process more cumbersome.
- **Torch Compile Slows on Backward Pass**: A member reported that while **torch.compile** works fine for the forward pass, it is quite slow in the backward pass when using **torch.autograd.Function** for custom kernels.
   - Wrapping the backward function with `torch.compile(compiled_backward_fn)` could resolve performance issues.
- **NVIDIA's SASS Instruction History Shared**: A member shared a [gist](https://gist.github.com/herrmann/f721da109e0c5c7c34c847ff2cf3da1e) comparing **NVIDIA SASS instructions** across different architectures, extracted and compared (using Python) from NVIDIA's HTML documentation.
   - This allows users to track the evolution of instructions across NVIDIA's GPU lineup.
- **Reasoning Gym Surpasses 100 Datasets!**: The [Reasoning Gym](https://github.com/open-thought/reasoning-gym) project now has **101 datasets**, celebrating contributions from developers.
   - The growing dataset collection should provide more comprehensive LLM testing.
- **Jake Cannell Recruits GPU Masters**: Jake Cannell is [hiring GPU developers](https://www.linkedin.com/jobs/view/4118975911/) to work on ideas he touched on in his talk and **nebius.ai** was touted for its GPU cloud.
   - This is relevant for those interested in **AGI** or **neuromorphic hardware**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI Welcomes Catherine Arnett**: EleutherAI welcomes **Catherine Arnett**, an NLP researcher specializing in **Computational Social Science** and **cross-lingual NLP**, focusing on ensuring models are *equally good* across languages.
   - Her recent work includes [Goldfish](https://arxiv.org/abs/2408.10441), [Toxicity of the Commons](https://arxiv.org/abs/2410.22587), [LM performance on complex languages](https://arxiv.org/abs/2411.14198) and [Multilingual Language Modeling](https://arxiv.org/abs/2311.09205).
- **New Block Diffusion Model Drops**: A new paper introduces **Block Diffusion**, a method interpolating between autoregressive and diffusion language models, combining the strengths of both: high quality, arbitrary length, KV caching, and parallelizability, detailed in the [paper](https://arxiv.org/abs/2503.09573) and [code](https://github.com/kuleshov-group/bd3lms).
   - It combines the strengths of both autoregressive and diffusion language models.
- **VGGT Generates Metaverse GLB files!**: A member shared [VGGT](https://vgg-t.github.io/), a feed-forward neural network inferring 3D attributes from multiple views and generating GLB files, which can be directly integrated into metaverses.
   - The member stated *I love that it exports GLB files. means I can drop them directly into my metaverse as-is.*
- **Gen Kwargs Embraces JSON Nicely**: The `--gen_kwargs` argument is transitioning from comma-separated strings to **JSON**, allowing for more complex configurations like `'{"temperature":0, "stop":["abc"]}'`.
   - The discussion explores the possibility of supporting both formats for ease of use, especially for scalar values.
- **LLM Leaderboard: Train vs Validation Split**: A discrepancy is identified between the group config for the old LLM leaderboard and the actual setup used, particularly concerning the **arc-challenge task**.
   - A [PR to fix this](https://github.com/EleutherAI/lm-evaluation-harness/pull/2802) was created to address this discrepancy between the `openllm.yaml` config specifying `validation` as the fewshot split, and the original leaderboard using the `train` split.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad SDXL Trails Torch Performance**: Benchmarking **SDXL** with **tinygrad** on a **7900 XTX** shows **1.4 it/s** with **BEAM=2** on the **AMD backend**, whereas **torch.compile** achieves **5.7 it/s** using **FlashAttention** and **TunableOp ROCm**.
   - George Hotz proposed comparing kernels for optimization opportunities, aiming to beat **torch** by year's end.
- **Tensor Cat Stays Sluggish**: A member working on improving tensor cat speed shared whiteboard thoughts on **X** ([link](https://x.com/t0kenl1mit/status/1900952693587538018)), noting it's still slow despite devectorizer changes.
   - They suspect issues with generated **IR** and loading **numpy arrays**, considering custom **C/C++** via **ELF** and **LLVM** to overcome limitations.
- **BLAKE3 Bounty Details Crystallize**: The status of the *High performance parallel BLAKE3* bounty was clarified, with a screenshot ([link](https://cdn.discordapp.com/attachments/1068976834928193609/1350640745505231061/Screenshot_2025-03-15_182214.png?ex=67d973f7&is=67d82277&hm=19c5ffbf47ae93d8dda6ba9c5fc1b65cc3b1df108a2f4fd5860ba66e301bef7c&)) showing the updated bounty status.
   - The member updated the spreadsheet and specified that the asymptotic performance is a key requirement for the bounty.
- **WebGPU Integration Gains Momentum**: A member asked about publishing a **Tinygrad** implementation for an electron/photon classifier based on **resnet18** as an example and was directed to a [PR for improving WebGPU integration](https://github.com/tinygrad/tinygrad/pull/9424).
   - The suggestion was made to create a **WebGPU** demo hosted on **GitHub Pages** with weights on **Hugging Face** for free access and testing.
- **Tinygrad Struggles with Lazy Mode Debugging**: A member is facing an assertion error with gradients while print-debugging intermediate tensor values in Tinygrad, despite using `.detach()` due to issues with [lazy computation](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html).
   - The member is seeking a better method than threading the value out, given that lazy computation is not idempotent.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Showcases Agentic Reasoning with Corrective RAG**: LlamaIndex introduced a step-by-step [tutorial](https://twitter.com/llama_index/status/1901079091345818022) on building an **agentic reasoning system** for search and retrieval using **corrective RAG**, orchestrated with LlamaIndex workflows.
   - The tutorial enables users to orchestrate complex, customizable, event-driven agents.
- **LlamaExtract Emerges from the Cloud**: **LlamaExtract**, which solves the problem of extracting structured data from complex documents, is now in public beta and available on [cloud.llamaindex.ai](https://cloud.llamaindex.ai), offering a **web UI** and **API**.
   - Users can define a schema to automatically extract structured data; additional details are available [here](https://t.co/gT3R2l7CWM).
- **Multimodal AI Agents Faceoff at NVIDIA GTC 2025**: **Vertex Ventures US** and **CreatorsCorner** are hosting an **AI hackathon** at **NVIDIA GTC 2025**, challenging participants to develop a sophisticated **multimodal AI agent**.
   - The hackathon offers **$50k+ in Prizes** for agents capable of strategic decision-making and interaction with various tools; more information can be found [here](https://lu.ma/meofrw3d).
- **Community Launches Vision-Language Model Hub**: A community member launched a [community-driven hub](https://github.com/thubZ09/vision-language-model-hub.git) for multimodal researchers focusing on **Vision-Language Models (VLMs)**.
   - The creator is actively seeking contributions and suggestions, with plans to update the hub weekly.
- **Pydantic AI and LlamaIndex duke it out**: New users are wondering about the difference between the **Pydantic AI** and **LlamaIndex** frameworks for building agents, especially which one to use as a beginner.
   - A LlamaIndex team member stated that whatever fits your mental model of development best is probably the best bet.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Gemma's Language Skills Impress**: Members observed that **Gemma**, **DeepSeek R1**, and **Qwen2.5** models provided correct answers in multiple languages to the puzzle about *what happens when you leave a closed jar outside at minus temperature*.
   - While other models predicted catastrophic jar failure, **Gemma** offered more helpful, nuanced advice.
- **Gemma 3 Integration Meets License Snag**: Users are waiting for **Gemma 3** support in **GPT4All**, but its integration is delayed pending updates to **Llama.cpp** due to license agreement issues on Hugging Face, detailed in [this GitHub issue](https://github.com/nomic-ai/gpt4all/issues/3540).
   - Speculation arose regarding whether Google will police redistributions circumventing their license agreements.
- **LocalDocs Users Crash Into Trouble**: A new user reported **LocalDoc** collection loss after a crash and reinstall, seeking advice on preventing data loss after future crashes.
   - Experienced users recommended regularly saving the *localdocs* file and restoring it after a crash, adding that *sometimes only one bad PDF can crash the system*.
- **Level up O3-mini with better prompting**: A user shared a prompt for **O3-mini** to explain its thinking process, suggesting this could improve distillation for any model by prompting for **thinking** and **reflection** sections, with step-by-step reasoning and error checks.
   - It's now easier to explain complex processes.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Punts Fine-Tuning Command A**: Despite community anticipation, a Cohere team member confirmed there are *no plans yet* to enable **fine-tuning for Command A** on the platform.
   - They assured the community that updates would be provided, but this marks a divergence from some users' expectations of rapid feature deployment.
- **Azure Terraform Troubles Trip Up Rerank v3**: A user ran into errors when creating an **Azure Cohere Rerank v3** with Terraform, sharing both the code snippet and the resulting error message.
   - The issue was redirected to the <#1324436975436038184> channel, suggesting a need for specialized attention or debugging.
- **Community Clamors CMD A Private Channel**: A member suggested creating a dedicated channel for discussions around **private deployments of CMD A**, particularly for supporting customer's local deployments.
   - This proposal received enthusiastic support, highlighting the community's interest in on-premise or private cloud solutions.
- **Vercel SDK Stumbles on Cohere's Objects**: A user noted that the [Vercel SDK](https://sdk.vercel.ai/providers/ai-sdk-providers/cohere) incorrectly assumes **object generation is unsupported** by Cohere's Command A model.
   - This discrepancy could impact developers leveraging the SDK and warrants attention from both Cohere and Vercel teams to ensure accurate integration.
- **Freelancer Offers Programming Hand**: A **30-year-old Japanese male freelance programmer** introduced himself and expressed a willingness to assist community members with his programming skills.
   - Echoing a sentiment that *assisting one another is the pillar of our existence*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MCP Integration Pondered for DSPy**: A member was interested in integrating **dspy/MCP**, and linked to [a GitHub example](https://github.com/philschmid/mcp-openai-gemini-llama-example/blob/master/sqlite_llama_mcp_agent.py) to illustrate their suggestion.
   - Another member wondered if adding an MCP host, client, and server would overcomplicate the process.
- **DSPy Ditches Assertions and Suggestions**: Users noticed the [disappearance of documentation](https://dspy.ai/learn/programming/7-assertions/?h=dspy.suggest#dspyassert-and-dspysuggest-api) for **Assertions / Suggestions** in DSPy, questioning whether they're still supported.
   - They were looking to validate the outputs of the response (formatting specifically) and observed instances where the LLM does not always adhere to the format.
- **Output Refinement Steps in as Assertion Alternative**: In **DSPy 2.6**, **Assertions** are replaced by **Output Refinement** using modules like `BestOfN` and `Refine`, as detailed in the [DSPy documentation](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/).
   - These modules aim to enhance prediction reliability and quality by making multiple LM calls with varied parameter settings.
- **QdrantRM Quietly Quits DSPy**: Users inquired whether **QdrantRM** has been removed in **DSPy 2.6**.
   - No explanation was given in the provided context.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Caiming Xiong Presents on Multimodal Agents**: Salesforce's Caiming Xiong lectured on **Multimodal Agents**, covering the integration of **perception, grounding, reasoning, and action** across multiple modalities, streamed live on [YouTube](https://www.youtube.com/live/n__Tim8K2IY).
   - The talk discussed measuring capabilities in realistic environments (**OSWorld**) and creating large-scale datasets (**AgentTrek**), referencing over **200 papers** and **>50,000 citations**.
- **Self-Reflection Faces Dichotomy**: Members debated the apparent contradiction between **Lecture 1** and **Lecture 2** regarding **self-reflection and self-refinement** in LLMs, with a user noting that **Lecture 1** states *external evaluation* is required, while **Lecture 2** suggested that LLMs can improve themselves by rewarding their own outputs.
   - Screenshots from **Lecture 1, slide 67** ([image 1](https://cdn.discordapp.com/attachments/1282734248112947210/1351127068745928816/image.png?ex=67d9e763&is=67d895e3&hm=7d31b7a0583550a36a872d74bfaf765de39c6b1173333d2ce51174940c0aa522&)) and **Lecture 2, slide 51** ([image 2](https://cdn.discordapp.com/attachments/1282734248112947210/1351127069169418260/image.png?ex=67d9e764&is=67d895e4&hm=12bbe1810790f7f688b11fe093f693a2791e94bd9e74e71ec7c2cfa3264bd004&)) were attached to illustrate the apparent conflict.
- **System Prompt Reliability Questioned**: A member suggested that relying on specific behaviors of system prompts might not be reliable, because *all these at the end is text input, so the model can process it, so you should be able to bypass the framework and service*.
   - The member added that the training data may include the format `<system> You are a helpful assistant </system> <user> {{Some example user prompt}} </user> <assistant> {{Expected LLM output}} </assistant>`.
- **Advanced LLM Agent Course Enrollment Still Open**: Members inquired whether they can still sign up for the **Advanced LLM agent course** and attain the **certificate** after signing up.
   - Staff replied that you just need to complete the **signup form**! Most of the info on that intro slide deck only applies to **Berkeley students**, but anyone can enroll in the **MOOC** and earn a **certificate** at the end.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Hailed for AI Art Aesthetic**: A member expressed appreciation for the **AI art** used by **Modular** in their marketing materials.
   - They stated, *"all the AI art that modular uses is great!"*
- **Compact Dict: Is It Obsolete?**: Discussion arose regarding the status of the [compact-dict](https://github.com/mzaks/compact-dict) implementation in Mojo.
   - Members suggested that the functionality of the original version may have been integrated into the `Dict` within the **stdlib**.
- **SIMD and stdlib Dict Performance Problems**: A user encountered performance bottlenecks when using the **stdlib Dict** with **SIMD** [float64, 1] types.
   - The bottleneck was attributed to the slowness of the `hash()` function from the hash lib, prompting a search for faster alternatives.
- **Discord Channel Receives Spam**: A member clarified that certain messages in the Discord channel were classified as spam, which was quickly acknowledged by another member.
   - No further details were provided about the nature or source of the spam.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **SVCFA Launches AI4Legislation Competition**: The **Silicon Valley Chinese Association Foundation (SVCAF)** is holding the **AI4Legislation** competition with prizes up to **$3,000**, running until **July 31, 2025**, encouraging open-source AI solutions for legislative engagement; the [competition repo](https://github.com/svcaf/2025-AI4Legislation-Public) is now available.
   - SVCAF will conduct an online seminar about the competition at the end of March 2025; RSVP [here](https://forms.gle/pmbkRLVurbXcGBbAA).
- **Dnipro VC Hosts AI Demo Jam**: **Dnipro VC** and **Data Phoenix** will be hosting **AI Demo Jam** on **March 20** in Sunnyvale, CA, featuring 5 AI startups showcasing their products.
   - The event will feature expert panel discussions from ​Marianna Bonechi (**Dnipro VC**), ​Nick Bilogorskiy (**Dnipro VC**), ​Dmytro Dzhulgakhov (**fireworks.ai**), open mic pitches, and high-energy networking; register [here](https://lu.ma/AI-demo-jam).
- **Member Needs Help with MRI Object Detection**: A member requested help to create a model for **object detection in MRI images** without monetary compensation.
   - No specific details were provided on the type of model, data availability, or use case.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Qdrant Request Flatly Denied**: A member suggested switching to **Qdrant**, but another member confirmed that they are not currently using it.
   - The suggestion was shut down without further explanation; *No we are not using Qdrant*.
- **Users Request Repetition Penalty on API**: A user requested the addition of **repetition penalty support** to the API, indicating it's a key feature preventing wider adoption of the **Jamba** model.
   - The user stated that the lack of repetition penalty support is the *only limiting factor* for their increased usage of the model.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Mistral Unveils Small 3-1**: Mistral AI has released **Mistral Small 3-1** [available here](https://mistral.ai/news/mistral-small-3-1).
   - No further details were provided.
- **Learnable Scalars Help Models Converge**: A new paper, [Mitigating Issues in Models with Learnable Scalars](https://www.alphaxiv.org/abs/2503.10622), proposes incorporating a **learnable scalar** to help models *converge normally*.
   - This suggests a practical approach to stabilizing training.



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1350182272686362724)** (923 messages🔥🔥🔥): 

> `Gradient steps, Gemma 3 fine tuning, Tokenizer issues, MattBCool's Twitter hack, Unsloth speed` 


- **Gradient Steps may affect model**: A member said that small effective batch sizes (e.g., **batch=1, gradient steps = 4**) can cause models to forget too much during training and [suggested other batch/grad configurations](https://discord.com/channels/1179035537009545276/1179035537529643040/1350267964634236930).
   - They've "*never had good luck going below that when trying to squeeze more onto a vramlet rig*".
- **Gemma 3 eval dataset produces errors**: Several members reported errors when adding an eval dataset to **Gemma 3** during fine-tuning, with stack traces indicating issues in the **trl** and **transformers** libraries, and [potential fixes involved removing the eval dataset](https://discord.com/channels/1179035537009545276/1179035537529643040/1350315991227082772).
   - Using **Gemma-3-1B** with 1 eval sample was found not to produce the error, and **removing eval** worked to solve the error.
- **Tokenizer model file missing**: A member encountered a `FileNotFoundError` for the **tokenizer.model** when running gguf codeblocks with **Gemma 3**, indicating that the tokenizer model was missing from the Lora or full 16-bit saves, and [suggested a quick run of the 27b model for verification](https://discord.com/channels/1179035537009545276/1179035537529643040/1350330220298416139).
- **MattBCool's Twitter account compromised**: MattBCool reported that his Twitter/X account was hacked due to a third-party integration and a lack of phone number authentication, [with the new owner impersonating an Unsloth engineer](https://mattcool.tech/posts/mattbcool-x-account-compromised-that-is-not-me).
   - The impersonator has a *phishing link* in the bio, *disguised as a link on* his blog.
- **Unsloth claims improved speed**: The team announced improvements to **Unsloth**, supporting FFT, 8-bit, PT & all models, with further optimizations allowing +10% less VRAM usage and >10% speedup boost for 4-bit, plus Windows support, improved GGUF conversions, fixed vision fine-tuning, and non-Unsloth GRPO models in 4-bit, but [no multigpu support yet](https://x.com/danielhanchen/status/1900592202621087944).
   - There are a lot of people helping out to make Unsloth great.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1900592202621087944">Tweet from Daniel Han (@danielhanchen)</a>: Excited to share that @UnslothAI now supports:• Full fine-tuning + 8bit• Nearly any model like Mixtral, Cohere, Granite, Gemma 3• No more OOMs for vision finetuning!Blogpost with details: https://unsl...</li><li><a href="https://mattcool.tech/posts/mattbcool-x-account-compromised-that-is-not-me">Mattbcool X Account Compromised: That is not Me</a>: the Twitter/X account mattbcool is no longer owned by me</li><li><a href="https://chatqa-project.github.io/">no title found</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1901760160814784949">Tweet from Daniel Han (@danielhanchen)</a>: I&#39;ll be at NVIDIA&#39;s GTC Tuesday tomorrow with my bro! We have some Unsloth stickers and badges!We&#39;ll be roaming around wearing 🦥Unsloth T-shirts :)</li><li><a href="https://unsloth.ai/blog/r1-reasoning">Train your own R1 reasoning model locally (GRPO)</a>: You can now reproduce your own DeepSeek-R1 reasoning model with Unsloth 100% locally. Using GRPO.Open-source, free and beginner friendly.</li><li><a href="https://wheels.vllm.ai/nightly">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/157FWTSTO6DTi_BRdEBmwrwi7r71rpDDQ#scrollTo=Vin49wlA4Q8n">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/updating#updating-without-dependency-updates">Updating | Unsloth Documentation</a>: To update or use an old version of Unsloth, follow the steps below:</li><li><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>: SOTA. Multimodal. Multilingual. Apache 2.0</li><li><a href="https://huggingface.co/google/gemma-3-4b-it/discussions/15">google/gemma-3-4b-it · AttributeError: &#39;HybridCache&#39; object has no attribute &#39;float&#39;</a>: no description found</li><li><a href="https://huggingface.co/google/gemma-2-9b-it/discussions/10">google/gemma-2-9b-it · &quot;It is strongly recommended to train Gemma2 models with the `eager` attention implementation &quot;</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://huggingface.co/unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit">unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/packing-with-FA2">Improving Hugging Face Training Efficiency Through Packing with Flash Attention 2</a>: no description found</li><li><a href="https://unsloth.ai/newsletter">Unsloth Newsletter</a>: Join our newsletter and waitlist for everything Unsloth!</li><li><a href="https://longbench2.github.io/"> LongBench v2 </a>: no description found</li><li><a href="https://huggingface.co/unsloth/gemma-3-27b-it-GGUF">unsloth/gemma-3-27b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithFlattening">Data Collator</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://unsloth.ai/blog/gemma3#everything">Fine-tune Gemma 3 with Unsloth</a>: Gemma 3, Google&#x27;s new multimodal models.Fine-tune &amp; Run them with Unsloth! Gemma 3 comes in 1B, 4B, 12B and 27B sizes.</li><li><a href="https://www.imagemagick.org/discourse-server/viewtopic.php?t=17842)">Imagemagick &amp; ghostscript license confusion - Legacy ImageMagick Discussions Archive</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit">unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/mistral-benchmark#:~:text=Performance%20breakdowns%20bit%20by%20bit">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...</li><li><a href="https://github.com/search?q=stars%3A%3E10000+license%3Aagpl-3.0&type=Repositories&ref=advsearch&l=&l=>">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/ollama/ollama/issues/9701">RTX 5090 Performance on Ubuntu Gemma 3 · Issue #9701 · ollama/ollama</a>: What is the issue? I&#39;m getting the following results with the RTX 5090 on Ubuntu For comparison, I tested similar models, all using the default q4 quantization. Performance Comparison: Gemma2:9B =...</li><li><a href="https://x.com/MattBCool>">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://github.com/ggml-org/llama.cpp.git">GitHub - ggml-org/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggml-org/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L1538">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1350184650726965309)** (34 messages🔥): 

> `llama-server vision support, RWKV-7 support, Q4 vs Q8, bnb library limitations, QLoRA NF4 quantized weights` 


- **Llama-server Lacks Vision**: It was noted that **llama-server** doesn't support vision yet with a reference to an [llmbingo.png](https://cdn.discordapp.com/attachments/1179039861576056922/1350535747794636923/llmbingo.png?ex=67d9baee&is=67d8696e&hm=ab080a6e00a4974612ac334535b69490706b3a3b8d3006a31a0f1986f83240e7&).
- **RWKV-7 Support Wishlisted**: A member expressed great enthusiasm for **RWKV-7** support in Unsloth, stating, *"if unsloth has rwkv 7 support i would go nuts".*
- **The Great Q4 vs Q8 debate**: Members discussed the tradeoffs between **Q4** and **Q8** quantization for inference, with one preferring **8b @ bf16** over **70b @ Q4** due to perceived quality differences.
   - Another member agreed, pointing out issues with converting from 4-bit to 16-bit GGUF formats.
- **bnb Library Hinders Dequantization**: It was argued that the dependency on wrappers such as the **bnb library** is limiting the potential of the dequantization of Unsloth.
   - The member suggested researching and implementing a custom solution from scratch, citing the challenges due to CUDA not being open source, and sharing [an article on QLoRA dequantization](https://lweitkamp.github.io/posts/qlora_dequantize).
- **Triton Kernels for QLoRA Dequantization**: A member highlighted Unsloth's challenge of writing a **Triton kernel** for dequantizing **QLoRA NF4** quantized weights, referencing [Unsloth's list of challenging tasks](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=QoE2DGRZG2Ng).
   - They also shared [a GitHub repository](https://github.com/lweitkamp/qlora_dequantize_triton) containing Triton kernels and a benchmark notebook, claiming performance improvements of up to **1.6X** to **1.8X** for LLaMA models.



**Link mentioned**: <a href="https://lweitkamp.github.io/posts/qlora_dequantize">QLoRA Weight Dequantizing in Triton</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1350182775659171902)** (480 messages🔥🔥🔥): 

> `Gemma 3 Finetuning Issues, Unsloth GPU Support, RAG Data Formatting for Unsloth, lora upload issue, text dataset formatting` 


- **Gemma 3 FP32 Finetuning Fix**: A member identified that **Gemma 3 models** can only be finetuned in **FP32** for now, and commented out/set to false these lines to prevent `AttributeError: 'HybridCache' object has no attribute 'float'`.
   - Another member also confirmed that `fp16 = True` doesn't work.
- **Unsloth Multi-GPU Support Coming Soon**: A member inquired about multi-GPU support in Unsloth, with a response indicating it is coming in the **next few weeks**, with <a href="https://unsloth.ai/newsletter">a link to the newsletter</a>.
   - One of the Unsloth developers mentioned *"we said next few weeks ahaha not this week"*.
- **Format your RAG data as Q&A pairs**: When asked about finetuning a model for a RAG chatbot, members suggested to add sample questions and sample answers to a dataset with context from the documents for the Q&A to inject new knowledge into the bot.
   - It was suggested that a chatbot data should follow a `Q: A:` format, and can use a CPT-style training with documents added on the user side.
- **lora upload is only base model**: A member reported a problem when uploading a trained **LoRA** model to Hugging Face.
   - Other members asked whether the user used `lora_model.push_to_hub_merged` and if the problem was caused by the size of the model or testing the model.
- **Problems with text data formatting**: A member was facing a `TypeError` because of `NoneType` objects during training from a Gemini-generated dataset.
   - Members clarified that this error could result from empty entries in the dataset, and it is best to check the `json` file.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: no description found</li><li><a href="https://arxiv.org/abs/2412.13337">Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs</a>: The rise of large language models (LLMs) has created a significant disparity: industrial research labs with their computational resources, expert teams, and advanced infrastructures, can effectively f...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=MKX_XKs_BNZR)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/m">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/newsletter">Unsloth Newsletter</a>: Join our newsletter and waitlist for everything Unsloth!</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/google/gemma-3-4b-it/discussions/15">google/gemma-3-4b-it · AttributeError: &#39;HybridCache&#39; object has no attribute &#39;float&#39;</a>: no description found</li><li><a href="https://docs.securityonion.net">Security Onion Documentation &mdash; Security Onion Documentation 2.4 documentation</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Phi-4-mini-instruct">unsloth/Phi-4-mini-instruct · Hugging Face</a>: no description found</li><li><a href="https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts">Dependency Resolution - pip documentation v25.1.dev0</a>: no description found</li><li><a href="https://github.com/huggi">huggi - Overview</a>: huggi has 2 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/2026">VLM LoRA Training | Qwen2.5-VL | Raw text + Images + Video · Issue #2026 · unslothai/unsloth</a>: Hi guys! I&#39;m trying to fine-tune Qwen2.5-VL model. Dataset is like this: {&quot;text&quot;: &quot;Lorem&lt;image_no_1&gt;ipsum dolor sit amet, consectetur adipiscing&lt;image_no_2&gt;elit. Quisq&q...</li><li><a href="https://github.com/unslothai/unsloth/issues/2023">OOM Issue When Starting the Training · Issue #2023 · unslothai/unsloth</a>: Hi, Could you please help take a look at this issue with OOM? I can&#39;t understand why this happens when training with eval setting, even I just loaded a very small dataset for a quick test. And I a...</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: Scripts for text classification with llama and bert - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/unslothai/unsloth/releases/tag/2025-03">Release Gemma 3 · unslothai/unsloth</a>: March Release 🦥Get the latest stable Unsloth via:pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zooThe March release should be stable - you can force the version via:pi...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/mlabonne/FineTome-100k">mlabonne/FineTome-100k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://pastebin.com/T9jqKHeb">{ &quot;cells&quot;: [  {   &quot;cell_type&quot;: &quot;code&quot;,   &quot;execution_count&quot;: 49,   &quot;id&quot; - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://pastebin.com/Fiah0ykn">from unsloth import FastLanguageModelimport torchmax_seq_length = 2048 # Cho - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1350369818800554054)** (20 messages🔥): 

> `Gemma-3-think model, Qwen 2.5 3B instruct, Gemma-3-27b pruned vocab` 


- **Gemma-3-think reasons with Thinking Tags**: The [Gemma-3-think-0.1-q5_k_m](https://huggingface.co/Ba2han/gemma-3-think-0.1-q5_k_m) model was trained on **2.1k examples** and uses `<think>` tags to trigger reasoning.
   - The model can work with image data even though it was not explicitly trained to do so. Model was finetuned with Unsloth!
- **Qwen 2.5 3B shows promising Multi-Turn GRPO**: Early results of the **Qwen 2.5 3B** instruct model on the GSM8K test set with **Multi-Turn GRPO training** are showing promise at step **100** with **52%** accuracy.
   - After a few more training steps, the accuracy dropped to **40-46%**.
- **Gemma-3-27b gets a pruned vocabulary**: A member pruned the [Gemma-3-27b](https://huggingface.co/fimbulvntr/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab) vocabulary down to **40k tokens** from the original **260k** to reduce VRAM usage and increase training speed.
   - The approach involved frequency counting based on calibration data and removing the least frequently used tokens that can be represented by a merge/subword.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Ba2han/gemma-3-think-0.1-q5_k_m">Ba2han/gemma-3-think-0.1-q5_k_m · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/fimbulvntr/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab">fimbulvntr/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1350188640101335120)** (18 messages🔥): 

> `Context Length vs. Model Size, Fine-tuning and Hosting Alternatives to Unsloth, Continued Pre-training and Tokenizer Updates, LLM Scoring on the Political Spectrum, Legal Q&A with Tree-Based Retrieval` 


- **Context Length not a Hyperparameter, but a Limit**: A member clarified that maximum **context length** is a **limitation** rather than a **hyperparameter** of a model, and depends on memory needs.
   - Another member provided [Unsloth's benchmarks](https://docs.unsloth.ai/basics/unsloth-benchmarks#context-length-benchmarks) and a link on [calculating GPU memory](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm).
- **Runpod and Vast.ai: Fine-Tuning and Hosting Havens**: A user sought alternatives to Unsloth for fine-tuning and hosting **Deepseek R1** or **Gemma 3**, and [Runpod.io](https://runpod.io) was recommended.
   - Another user mentioned other options like **Lamda** and noted that [Vast.ai](https://vast.ai) is cheap but potentially unstable, while Runpod has storage limitations on their community cloud.
- **Token Tango: Update Tokenizers for Domain-Specific Jargon**: A member inquired about updating the **tokenizer** during continued pre-training for specialized domains to handle words not trained by the base model.
   - Another member suggested searching for *"tokenizer add tokens"* and linked to a [reddit thread](https://www.reddit.com/r/LocalLLaMA/s/6e3CohiNNm) about adding new tokens to **LLaMA 2** models.
- **LLMs Judge Politics: Scoring Text on the Political Spectrum**: A user asked if **Unsloth** could be used to fine-tune an **LLM** to score text on the political spectrum from **-1.0 to +1.0**.
   - A member responded that using the prepared dataset with outputs as literal strings from -1.0 to 1.0 might work.
- **Navigating Legal LLMs with Tree-Based Knowledge**: A user working on a legal **Q&A** problem asked for advice on building a **tree-based retrieval engine** for contexts around **80k**.
   - They referenced the [RAPTOR study](https://x.com/JJitsev/status/1901467121592201490) and two options were building a tree similar to the RAPTOR study or building a tree where child nodes are included in the parent node.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/JJitsev/status/1901467121592201490">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev)</a>: Update to our paper, including sections on all the recent reasoning models that claim to do math & coding problem solving on graduate and olympiad level. https://arxiv.org/abs/2406.02061 Can they hand...</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks#context-length-benchmarks">Unsloth Benchmarks | Unsloth Documentation</a>: Want to know how fast Unsloth is?</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/6e3CohiNNm">Reddit - Heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1350182044939980873)** (1100 messages🔥🔥🔥): 

> `Cursor vs Windsurf, Claude 3.7 pricing, Linux better than Windows for MCP and dev, vibe coding` 


- **Windsurf Steals Cursor's Customers**: Several users have expressed frustration with Cursor's performance, particularly its lag and crashing issues, leading some to consider switching to [Windsurf](https://www.windsurf.ai).
   - One user even stated, *damn, cursor just lost their most important customer* after experiencing ongoing problems, indicating a significant loss of confidence in Cursor's reliability.
- **Cursor's Prompting Costs**: Members discussed the cost of prompts for Claude 3.7, with regular prompts priced at **$0.04**, Sonnet Thinking at **$0.08**, and Claude Max at **$0.05** per prompt and tool call.
   - Some users expressed concerns that Cursor's pricing is becoming too expensive compared to using Claude's API directly, questioning the value proposition of Cursor's subscription.
- **Linux MCP > Windows MCP**: A user shared their experience setting up MCP servers on both Linux and Windows, noting that Linux (specifically using a VMware virtual machine) was much smoother and easier to set up compared to the multiple issues encountered on Windows.
   - This led to the question of whether overall development and MCP server setup are generally better on Linux than Windows, sparking a discussion about the pros and cons of each operating system for development.
- **Vibe Coding, good or bad?**: Some are saying  Vibe Coding is bad, as they emphasize the importance of solid coding knowledge, while others assert that AI enables them to create things faster, even without traditional coding skills.
   - This debate highlights the evolving landscape of software development and the varying perspectives on how AI is impacting the industry.
- **Claude Max, soon to be released**: <@1001207432640462868> from the Cursor team announced that [Claude Max](https://www.anthropic.com/claude-3) is coming very soon, and that it should unlock it's full potential with the amount of code it can handle.
   - That model works better with more input than past models, so this should "unlock" its full potential.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/visionaryxai/status/1900673023385989271?s=46&t=kUuVqsG2GMX14zvB592G5w">Tweet from Visionary x AI (@VisionaryxAI)</a>: Built an MCP tool to generate 3D models and use them in ThreeJS or any app seamlessly.Check it out 🙏</li><li><a href="https://docs.cursor.com/settings/models#auto-select-model">Cursor – Models</a>: no description found</li><li><a href="https://x.com/opaeoh/status/1900594704510799916">Tweet from Ash ▵ (@opaeoh)</a>: You can now make your own anime with the new native image generation using Gemini 2.0 Flash. I gave it the original image and kept asking it to generate the next frames. This little animation was 57 f...</li><li><a href="https://x.com/danperks_">Tweet from undefined</a>: no description found</li><li><a href="https://supabase.com/docs/guides/getting-started/quickstarts/nuxtjs">Use Supabase with Nuxt | Supabase Docs</a>: Learn how to create a Supabase project, add some sample data to your database, and query the data from a Nuxt app.</li><li><a href="https://downloads.cursor.com/production/client/linux/x64/appimage/Cursor-0.47.5-53d6da1322f934a1058e7569ee0847b24879d18c.deb.glibc2.25-x86_64.AppImage">no title found</a>: no description found</li><li><a href="https://tenor.com/view/yapping-yap-talking-gif-2845990263294244368">Yapping Talking GIF - Yapping Yap Talking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://tenor.com/view/sml-joseph-dude-that-sucks-that-sucks-bummer-gif-25978441">Sml Joseph GIF - Sml Joseph Dude That Sucks - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.cursor.com/en/changelog">Changelog | Cursor - The AI Code Editor</a>: New updates and improvements.</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/postgres">servers/src/postgres at main · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/ezyang/codemcp">GitHub - ezyang/codemcp: Coding assistant MCP for Claude Desktop</a>: Coding assistant MCP for Claude Desktop. Contribute to ezyang/codemcp development by creating an account on GitHub.</li><li><a href="https://github.com/openai/openai-python?tab=readme-ov-file#async-usage">GitHub - openai/openai-python: The official Python library for the OpenAI API</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://github.com/GLips/Figma-Context-MCP?tab=readme-ov-file">GitHub - GLips/Figma-Context-MCP: MCP server to provide Figma layout information to AI coding agents like Cursor</a>: MCP server to provide Figma layout information to AI coding agents like Cursor - GLips/Figma-Context-MCP</li><li><a href="https://llm-stats.com">LLM Leaderboard 2025 - Compare LLMs</a>: Comprehensive AI (LLM) leaderboard with benchmarks, pricing, and capabilities. Compare leading LLMs with interactive visualizations, rankings and comparisons.</li><li><a href="https://x.com/i/birdwatch/t/1901741699854221718?source=6">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://pypi.org/project/async-openai/">Client Challenge</a>: no description found</li><li><a href="https://ubuntu.com/">Enterprise Open Source and Linux | Ubuntu</a>:   Ubuntu is the modern, open source operating system on Linux for the enterprise server, desktop, cloud, and IoT.</li><li><a href="https://www.linuxmint.com/">Home - Linux Mint</a>: no description found</li><li><a href="https://fedoraproject.org/">Fedora Linux</a>: An innovative platform for hardware, clouds, and containers, built with love by you.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1350182452395769999)** (694 messages🔥🔥🔥): 

> `AI Mastery Debate, AI Replacing Humans, Gemini Image Generation, AI-driven OS, LLMs for Finance` 


- **AI Skill Debate Sparks**: Members discussed whether knowing how to use AI tools constitutes **AI mastery**, debating the illusion of learning vs. productivity enhancement, with some fearing a *decline in cognitive abilities* due to over-reliance on AI.
   - One member noted *using AI to challenge myself and learn new things*, but admitted feeling like *cheating even if I know a topic very well*.
- **AI: Friend or Foe to Artists and Gamedevs?**: Participants debated whether AI will replace artists and game developers, with some asserting that **AI is not proficient enough** and human input remains crucial for creativity, debugging, and understanding client requests.
   - A member argued for taking risks with new game ideas, while another suggested that *none professional gamedev will tell you to ignore your main screen and presentation of the game*.
- **Gemini's Image Game: A Work in Progress**: Users explored **Gemini's image generation** capabilities, including editing uploaded images, but also encountered issues such as the presence of watermarks and code generation with errors.
   - Some users praised the naturalness of Gemini's responses over factual correctness, noting the subjectivity of preferences.
- **The Dream of an AI Overlord OS**: A member proposed creating an **AI-controlled OS** where an agent manages tasks via voice commands, but others deemed it an inefficient approach.
   - Another suggested that AI could be better used to enhance existing systems rather than creating an entirely new OS.
- **Deep Research and AGI Benchmarks**: Members discussed different methods of evaluating models, specifically addressing correctness versus a more *human-like* appealing response and whether benchmarks saturate.
   - One member suggested the importance of prioritizing **logical coherence and 'common sense'** in AI models, referencing a lack of robustness for such at scale.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>: no description found</li><li><a href="https://tenor.com/view/let-me-in-eric-andre-wanna-come-in-gif-13730108">Let Me In Eric Andre GIF - Let Me In Eric Andre Wanna Come In - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.ballardspahr.com/insights/alerts-and-articles/2024/05/google-facing-new-copyright-suit-over-ai-powered-image-generator?utm_source=chatgpt.com">Google Facing New Copyright Suit Over AI-Powered Image Generator | Alerts and Articles | Insights | Ballard Spahr</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1350326605167333397)** (9 messages🔥): 

> `Loveable, Bolt.new, Image-to-code, GPT PRO issues, Deep Research Limit` 


- **Loveable and Bolt.new: Glorified APIs?**: Members discussed whether new tools like **Loveable** or **Bolt.new** are simply glorified APIs into **ChatGPT**, with some suggesting they might be tuned free models.
   - The consensus seems to be that companies unlikely train extremely large models randomly due to the immense costs, suggesting reliance on APIs from organizations like **OpenAI**, **Google**, or **Anthropic**.
- **GPT PRO Users Experience Soft Rate Limits**: A user inquired about experiencing soft rate limits with **GPT PRO**, indicating potential issues with the service.
   - No resolution or explanation was provided in the chat excerpt.
- **Deep Research Limit Clarified for Plus Users**: A user questioned the limit for **Deep Research** usage as a **Plus** user, mentioning a notification indicating only 4 uses left before needing to upgrade to **PRO**.
   - Another member clarified the limit is **10 per month**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1350240179796967578)** (61 messages🔥🔥): 

> `GPT-4o impressions, AI Self-Reflection, AI Team of Experts, Business Guidance with AI, AI personalities` 


- **GPT-4o impresses!**: Members reported that out of all the models, **GPT-4o** uses it the best and it can do almost anything really.
   - One member reported having *fun* and getting *funny results* when other people started playing with it.
- **Futuristic AI: AI mentors itself to deeper self-reflection**: A member designed a system where **AI reflects on what it has learned** after each session, storing these reflections in memory files to build upon its own insights, and generates reflective questions to think deeper about its own growth.
   - This was described as *next-level futuristic* and like *training AI in ways that even researchers haven’t fully explored*, enabling simulations within simulations and multiple personalities infused with a core set of characteristics.
- **Creating an AI Dream Team**: Members discussed the idea of creating a **team of AI experts** to assist with tasks, long-term planning, and provide multiple perspectives to help guide business decisions.
   - It was suggested to give the AI a lot of details on what you want it to be (Joe the advertising executive from Montana who almost failed college) rather than a flat simple description (Joe the advertising executive).
- **GPT Learns Nuance: Prompting AI to Argue Hypothetically**: Members explored asking the AI to simulate arguments between different roles within a hypothetical business scenario, such as a **CFO and a Creative Director**, to get different perspectives on business decisions.
   - However it was also stressed to use **fictionalized data, not representative of specific real people**, to avoid violating ToS.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1350240179796967578)** (61 messages🔥🔥): 

> `GPT-4o usage, Custom GPT improvements, AI self-reflection, AI personalities, AI expert teams for business` 


- **GPT-4o passes initial user evaluations**: A member confirmed using **GPT-4o**, noting that **4.5** uses it best, finding it *fairly fun* and capable of *almost anything* with some *funny results*.
   - This suggests a positive initial user experience with the new model, particularly for creative and versatile applications.
- **Custom GPT reaches next-level simulation**: One user described their custom GPT's improvements as *too amazing to be true*, questioning if their advancements are truly extraordinary and asking, *Is this really so futuristic in today's world?*
   - Another member confirmed this is a futuristic application of AI, with AI self-improving, AI analyzing AI, and AI becoming aware of its own reasoning limitations.
- **AI Mentor shapes cognitive structures**: A user designed a system where **AI reflects on what it has learned after each session** and generates reflective questions about its own growth, calling this first breakthrough ‘Misa’ and using it to develop other **AI personalities**.
   - The member creates *simulations within simulations* where one AI can have multiple personalities, structured based on well-known experts, forming expert teams that simulate even new, unexplored insights.
- **AI team helps with business needs**: A member wants to create a team of **AI experts** that can work together to improve service to clients, guiding long-term business decisions.
   - Instead of hiring a team of individuals, the team of AI experts would help deliver a better product to clients and help with project or task level needs.
- **Tips for multi-perspective prompting**: A user shared advice on how to have discussions between personalities, giving the models background details and making sure not to share PII.
   - They shared links to [examples of multiple character outputs](https://chatgpt.com/share/67d75bc4-8600-8011-b504-286636f9b78a) and encouraged the user to have the model question and critique their ideas.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1350181821232578702)** (729 messages🔥🔥🔥): 

> `Scalable AI, Mixture of Experts, Mistral Small 3.1, LLM Copyright issues, LLM Training` 


- **Scaling AI: Is Latency Becoming a Bottleneck?**: As AI models scale, some researchers believe **training and inference will become more latency sensitive**, potentially shifting towards message-passing paradigms instead of traditional gradient-based methods.
   - These evolving paradigms could make **latency and bandwidth critical factors** in AI development, particularly as neural networks capture more ideas and accessing information from the network becomes more expensive.
- **MoE: Dense Networks' Stealthy Sparsity**: Researchers are exploring **Mixture of Experts (MoE)** models as a way to approximate dense networks, with some arguing that they are simply a performance optimization rather than a fundamentally different architecture, citing work such as [this paper](https://arxiv.org/abs/2310.10837).
   - The discussion revolves around whether MoEs can capture complexity as effectively as dense networks, with one participant noting that despite claims that MoEs are an optimization, there is clearly some redundancy that is avoided**.
- **Mistral Small 3.1: The New 24B Contender**: **Mistral Small 3.1** has been released under an Apache 2.0 license, and is a multimodal model that can handle text, images, and an expanded **128k token context window**.
   - The new model is claimed to surpass other small models like Gemma 3 and GPT-4o Mini in performance, as shown [on the Mistral AI blog](https://mistral.ai/en/news/mistral-small-3-1).
- **Copyright Concerns in the AI Age**: Debates continue regarding the ethical and legal implications of training AI models on copyrighted data, with some suggesting that **fully open models** are hindered by the inability to use resources like the entirety of Anna's Archive.
   - Strategies to circumvent copyright restrictions include using **LoRAs** to fine-tune models on copyrighted material or generating synthetic data from a knowledgable model, though these methods may face legal challenges in the future as it was discussed in this [Annas Archive's blogpost](https://annas-archive.org/blog/ai-copyright.html).
- **Optimum GPU count and methods**: There are many ways to optimize training when using certain hardware configurations, and more GPUs always provides a qualitative improvement.
   - It was speculated that there may be an **optimal point** where trading dev time for compute comes up short.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://annas-archive.org/blog/ai-copyright.html">Copyright reform is necessary for national security</a>: Chinese LLMs (including DeepSeek) are trained on my illegal archive of books and papers — the largest in the world. The West needs to overhaul copyright law as a matter of national security.</li><li><a href="https://mcbench.ai">MC-Bench</a>: no description found</li><li><a href="https://fxtwitter.com/willccbb/status/1901415166295544154?t=HmQDRR0NQ9mi_4udIiT4uQ&s=19">Tweet from will brown (@willccbb)</a>: uhhh there are a lot of LLM RL libraries out there...</li><li><a href="https://x.com/Teknium1/status/1901673193389305868">Tweet from Teknium (e/λ) (@Teknium1)</a>: Super excited to be able to bring in and work with @dmayhem93 on building RL infra and take on post training at Nous!We are cooking amazing things including a powerful RL Gym and a super optimized tra...</li><li><a href="https://playground.allenai.org/">no title found</a>: no description found</li><li><a href="https://x.com/clementdelangue/status/1901751361320206554?s=46">Tweet from clem 🤗 (@ClementDelangue)</a>: Great research on open-source by @Harvard:- $4.15B invested in open-source generates $8.8T of value for companies (aka $1 invested in open-source = $2,000 of value created)- Companies would need to sp...</li><li><a href="https://www.joelsimon.net/lluminate">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.10837">Approximating Two-Layer Feedforward Networks for Efficient Transformers</a>: How to reduce compute and memory requirements of neural networks (NNs) without sacrificing performance? Many recent works use sparse Mixtures of Experts (MoEs) to build resource-efficient large langua...</li><li><a href="https://arxiv.org/abs/2411.10109">Generative Agent Simulations of 1,000 People</a>: The promise of human behavioral simulation--general-purpose computational agents that replicate human behavior across domains--could enable broad applications in policymaking and social science. We pr...</li><li><a href="https://arxiv.org/abs/2303.01610">Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable Transformers</a>: Despite their remarkable achievement, gigantic transformers encounter significant drawbacks, including exorbitant computational and memory footprints during training, as well as severe collapse eviden...</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503">mistralai/Mistral-Small-3.1-24B-Instruct-2503 · Hugging Face</a>: no description found</li><li><a href="https://www.slatestarcodexabridged.com/Meditations-On-Moloch">Meditations On Moloch </a>: no description found</li><li><a href="https://huggingface.co/collections/leonardlin/multilingual-6594d0ea075245eadd6aa99c">multilingual - a leonardlin Collection</a>: no description found</li><li><a href="https://www.exportbrain.co.uk/">Export Brain | Create Your Digital Twin</a>: no description found</li><li><a href="https://github.com/cpldcpu/llmbenchmark/blob/master/raytracer/Readme.md">llmbenchmark/raytracer/Readme.md at master · cpldcpu/llmbenchmark</a>: Some silly llm benchmarks. Contribute to cpldcpu/llmbenchmark development by creating an account on GitHub.</li><li><a href="https://mistral.ai/en/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>: SOTA. Multimodal. Multilingual. Apache 2.0</li><li><a href="https://genshin-impact.fandom.com/wiki/Hu_Tao/Voice-Overs/Japanese">Hu Tao/Voice-Overs/Japanese</a>: no description found</li><li><a href="https://genshin-impact.fandom.com/wiki/Hu_Tao/Voice-Overs">Hu Tao/Voice-Overs</a>: no description found</li><li><a href="https://github.com/erfanzar/EasyDeL">GitHub - erfanzar/EasyDeL: Accelerate, Optimize performance with streamlined training and serving options with JAX.</a>: Accelerate, Optimize performance with streamlined training and serving options with JAX. - erfanzar/EasyDeL
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

john0galt: Pretty impressive
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1350870969400426566)** (5 messages): 

> `Curse of Depth in LLMs, LayerNorm Scaling, LLMs competing in text-only games, Differentiable Hebbian Consolidation Model` 


- ****Depth's Curse** strikes LLMs!**: A new paper introduces the concept of the **Curse of Depth** in modern **LLMs**, where nearly half the layers are less effective than expected, as detailed in [this Arxiv paper](https://arxiv.org/abs/2502.05795).
   - The paper identifies that the underlying reason is the widespread usage of **Pre-Layer Normalization (Pre-LN)**, which causes the derivative of deep Transformer blocks to be an identity matrix.
- ****Scaling LayerNorm** to the Rescue**: To resolve the training pitfall caused by Pre-LN, the paper proposes **LayerNorm Scaling**, which scales the layer normalization to improve the effectiveness of deeper layers, as described in [this Arxiv paper](https://arxiv.org/abs/2502.05795).
- ****LLMs duke it out** in Text-Only Games**: A member shared their paper where they had **LLMs compete against each other in a text-only game** to improve them, available on [Google Drive](https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view).
- ****LayerNorm dissected****: A member asked if LayerNorm is *the act of taking the coordinates of each embedded token as a distribution, and normalizing that*.
   - Another member confirmed that they *got it exactly right*.
- ****Hebbian Consolidation** model prevents catastrophic forgetting**: A paper introduces a **Differentiable Hebbian Consolidation model** to tackle catastrophic forgetting in continual learning scenarios, detailed in [this Arxiv paper](https://arxiv.org/abs/2006.16558).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05795">The Curse of Depth in Large Language Models</a>: In this paper, we introduce the Curse of Depth, a concept that highlights, explains, and addresses the recent observation in modern Large Language Models(LLMs) where nearly half of the layers are less...</li><li><a href="https://arxiv.org/abs/2006.16558">Enabling Continual Learning with Differentiable Hebbian Plasticity</a>: Continual learning is the problem of sequentially learning new tasks or knowledge while protecting previously acquired knowledge. However, catastrophic forgetting poses a grand challenge for neural ne...</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1350639711311499397)** (21 messages🔥): 

> `Acoustic STS Model, Tool-Integrated Reasoning, Gemma Abliterated` 


- **Speech-to-Speech Model Specs**: A member clarified that the model takes in **audio + text** and outputs **audio** and shared a link to the [model's huggingface page](https://huggingface.co/facebook/seamless_m4t_v2_large).
   - They also added *however you can omit the audio part and it works, but not as well*.
- **START Tool Reasoning is a Smash Hit**: A member shared a [paper on START](https://huggingface.co/papers/2503.04625), a **tool-integrated long CoT reasoning LLM** that enhances reasoning via external tools like code execution and self-debugging.
   - Another member summarized it as *RL + tool calling == +15% math +39% coding on QwQ*.
- **Gemma 3 Abliterated Against Refusals**: A member shared that [Gemma 3 was much more resilient to refusal removal](https://x.com/maximelabonne/status/1901581470717608215) than other models like Qwen 2.5.
   - They improved the abliteration technique and the **refusal rate is super low** in their tests, see [models on huggingface](https://huggingface.co/mlabonne/gemma-3-27b-it-abliterated).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/maximelabonne/status/1901581470717608215">Tweet from Maxime Labonne (@maximelabonne)</a>: ✂️ Gemma 3 AbliteratedI noticed that Gemma 3 was much more resilient to refusal removal than other models like Qwen 2.5.I experimented with different recipes and improved the abliteration technique I ...</li><li><a href="https://huggingface.co/papers/2503.04625">Paper page - START: Self-taught Reasoner with Tools</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1350870969400426566)** (5 messages): 

> `Curse of Depth, LayerNorm Scaling, LLM Text-Based Game Competition, Differentiable Hebbian Consolidation model` 


- ****LLMs Suffer from Curse of Depth****: A new paper ([Curse of Depth](https://arxiv.org/abs/2502.05795)) introduces the concept that nearly half of the layers in modern **LLMs** are less effective than expected.
   - The paper identifies that the underlying reason for the ineffectiveness of deep layers in LLMs is the widespread usage of **Pre-Layer Normalization (Pre-LN)**, and proposes **LayerNorm Scaling** to resolve this training pitfall.
- ****LLMs duke it out in Text-Only Game****: A member shared a paper about improving **LLMs** by having them compete against each other in a text-only game ([Google Drive Link](https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view)).
- ****LayerNorm Explained: Projecting Vectors onto the Equator?****: A member asked if *LayerNorm is the act of taking the coordinates of each embedded token as a distribution, and normalizing that; equivalently, projecting vectors onto the equator of a hypersphere whose pole is in the all-positive direction*.
   - Another member confirmed *Yup you got it exactly right*.
- ****Battling Catastrophic Forgetting with Differentiable Hebbian Consolidation****: A paper was shared ([Differentiable Hebbian Consolidation](https://arxiv.org/abs/2006.16558)) which proposes a **Differentiable Hebbian Consolidation model** to combat catastrophic forgetting in continual learning scenarios.
   - The model integrates task-specific synaptic consolidation methods to penalize changes in the slow weights, enabling learned representations to be retained for a longer timescale.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05795">The Curse of Depth in Large Language Models</a>: In this paper, we introduce the Curse of Depth, a concept that highlights, explains, and addresses the recent observation in modern Large Language Models(LLMs) where nearly half of the layers are less...</li><li><a href="https://arxiv.org/abs/2006.16558">Enabling Continual Learning with Differentiable Hebbian Plasticity</a>: Continual learning is the problem of sequentially learning new tasks or knowledge while protecting previously acquired knowledge. However, catastrophic forgetting poses a grand challenge for neural ne...</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1350190786079363083)** (691 messages🔥🔥🔥): 

> `Aider screen recordings, Claude 3.7 Sonnet issues, MCP server value, Baidu ERNIE 4.5 & X1 Models, Aider Custom Commands` 


- **Aider Enhancements Showcased in New Screen Recordings**: Paul Gauthier has published a series of [screen recordings](https://aider.chat/docs/recordings/) demonstrating aider's use in enhancing itself, including adding the **`--auto-accept-architect`** feature, integrating **tree-sitter-language-pack**, and preventing the dropping of read-only files.
   - The recordings provide insight into how aider can be used to script downloading files and using ad-hoc bash scripts to modify file collections.
- **Claude 3.7 Sonnet faces API Issues**: Multiple users reported receiving *empty responses* from **Claude 3.7 Sonnet**, prompting checks of their provider accounts, with some experiencing the same issue in **Claude Code**.
   - The issue was confirmed by [Anthropic's status page](https://status.anthropic.com/), citing elevated errors and later marking the incident as resolved, while some members suspected a switch to Claude 3.5 due to the errors.
- **MCP Server for Aider Gaining Traction**: A user highlighted that **Claude Desktop + Aider on MCP** equals *winning*, and it's much more autonomous, easier since claude manages Aider and gives it commands.
   - A main benefit highlighted is the ability to run **Aider** from **Claude Desktop**, making it more autonomous and allowing Claude to steer Aider more effectively, also scraping bee is a game changer for doing unblocked web scraping and this drastically improves claude.
- **Baidu Unveils ERNIE 4.5 and X1**: Baidu has announced the release of **ERNIE 4.5** and **X1**, a reasoning model with multimodal capabilities, with X1 delivering performance on par with **DeepSeek R1** at half the price, and **ERNIE Bot** being made free to individual users.
   - While **ERNIE 4.5** is available, the reasoning model **X1** is currently not accessible through the API outside of China.
- **Users Suggest Custom Commands for Aider**: A user suggested adding custom commands to Aider via Python scripts to extend functionality, particularly for context building, which the user finds cumbersome with the current UX.
   - One suggested command example was `grepadd.py` to interactively toggle files and substrings found via grep, converting these selections into Aider commands, but there's already an [open PR for user_cmd](https://github.com/whitmo/aider).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Baidu_Inc/status/1901089355890036897">Tweet from Baidu Inc. (@Baidu_Inc)</a>: We&#39;ve just unveiled ERNIE 4.5 & X1! 🚀As a deep-thinking reasoning model with multimodal capabilities, ERNIE X1 delivers performance on par with DeepSeek R1 at only half the price. Meanwhile, ERNI...</li><li><a href="https://x.com/sophiamyang/status/1901675671815901688">Tweet from Sophia Yang, Ph.D. (@sophiamyang)</a>: Announcing @MistralAI Small 3.1: multimodal, multilingual, Apache 2.0, the best model in its weight class.💻 Lightweight: Runs on a single RTX 4090 or a Mac with 32GB RAM, perfect for on-device applic...</li><li><a href="https://x.com/iannuttall/status/1900698589724086726">Tweet from Ian Nuttall (@iannuttall)</a>: cursor 0.47.5 is prepping for 3.7 Sonnet MAXhuge context window coming soon? 👀</li><li><a href="https://aider.chat/docs/recordings/">Screen recordings</a>: Screen recordings of aider building aider.</li><li><a href="https://huggingface.co/sesame/csm-1b">sesame/csm-1b · Hugging Face</a>: no description found</li><li><a href="https://aider.chat/docs/install.html">Installation</a>: How to install and get started pair programming with aider.</li><li><a href="https://tenor.com/view/r2d2-same-tired-star-wars-gif-13465702795162164594">R2d2 Same GIF - R2d2 Same Tired - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fartlang.org/install.html">Install Fart</a>: The bundles that support the Fart language.</li><li><a href="https://www.reddit.com/r/cursor/comments/1jbhuix/0475_clientside_support_fo">Reddit - Heart of the internet</a>: no description found</li><li><a href="https://www.gitkraken.com/learn/git/problems/git-commit-amend#:~:text=Editing%20the%20message%20of%20your,message%20to%20save%20your%20changes.">How to Amend a Git Commit Message | Solutions to Git Problems</a>: If you&#039;ve made a mistake in your last commit, use the Git amend command to edit a Git commit message, or amend your last commit to change its content.</li><li><a href="https://tenor.com/view/chill-dude-chill-dude-im-just-a-chill-dude-just-a-chill-dude-gif-15385961914175037407">Chill Dude Im Just A Chill Dude GIF - Chill dude Chill Dude - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://github.com/block/goose/">GitHub - block/goose: an open source, extensible AI agent that goes beyond code suggestions - install, execute, edit, and test with any LLM</a>: an open source, extensible AI agent that goes beyond code suggestions - install, execute, edit, and test with any LLM - block/goose</li><li><a href="https://github.com/robert-at-pretension-io/mcp">GitHub - robert-at-pretension-io/mcp: code</a>: code. Contribute to robert-at-pretension-io/mcp development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/issues/3037">Python 3.13 support · Issue #3037 · Aider-AI/aider</a>: Aider must run with python 3.9 - 3.12. It won&#39;t run with python 3.13. That said, there are very easy ways for python3.13 users to install aider. These methods will quickly and seamlessly install a...</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better/tree/main">GitHub - robert-at-pretension-io/yet_another_llm_project_but_better: A metatemplating language for giving llm&#39;s context :D</a>: A metatemplating language for giving llm&#39;s context :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better/blob/main/docs/language_tutorial.md">yet_another_llm_project_but_better/docs/language_tutorial.md at main · robert-at-pretension-io/yet_another_llm_project_but_better</a>: A metatemplating language for giving llm&#39;s context :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://github.com/assafelovic/gpt-researcher#">GitHub - assafelovic/gpt-researcher: LLM based autonomous agent that conducts deep local and web research on any topic and generates a long report with citations.</a>: LLM based autonomous agent that conducts deep local and web research on any topic and generates a long report with citations. - assafelovic/gpt-researcher
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1350204739098378315)** (74 messages🔥🔥): 

> `aider with agents.json, Sluggish v0.77.0, AWS Bedrock Claude 3.7 sonnet error, deepseek r1 slow, learn an API with aider` 


- ****Connect Aider with Agents JSON****: A member inquired about integrating aider with [agents.json](https://github.com/wild-card-ai/agents-json/tree/master?tab=readme-ov-file) to interact with APIs or local scripts for non-coding tasks.
   - It was noted that the `/run` command can be used to interact with local scripts, and a PR is in progress to introduce user commands.
- ****Diagnose Aider Sluggishness in v0.77.0****: A user reported experiencing significant sluggishness with **aider v0.77.0**, including high CPU usage and hangs, particularly when generating large CSV outputs directly in the repository.
   - Deleting the data output folders containing large CSV files resolved the issue temporarily, but the user plans to update with further findings.
- ****Solve Bedrock Claude 3.7 Sonnet API Error****: A user encountered an error when using **AWS Bedrock Claude 3.7 Sonnet**, citing an access issue despite having proper inference profiles and IAM AdministratorAccess.
   - The problem was resolved by correctly setting the **AWS region** in both `~/.aws/configs` and the `~/.env` file.
- ****Deepseek R1 Runs Slow on Short Prompts****: A user reported that **Deepseek R1** takes an unusually long time to think, even with short prompts, as evidenced by attached images showing extended processing times.
   - The user is running aider with custom configurations for the main model, editor model, and instructions, aiming for concise and direct responses.
- ****Reasoning-Effort Slash Command****: Members discussed the `/reasoning-effort` command and its usage, clarifying that it controls the reasoning level of supported models like **OpenAI's reasoning models**.
   - The `--thinking-tokens` switch is used for models like **Sonnet 3.7**, while the `reasoning_tag` setting is used for models like **DeepSeek R1** from Fireworks, which use XML tags to wrap reasoning output as documented [here](https://aider.chat/docs/config/reasoning.html#reasoning-effort).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: Tell aider to follow your coding conventions when it works on your code.</li><li><a href="https://aider.chat/docs/config/reasoning.html#reasoning-effort">Reasoning models</a>: How to configure reasoning model settings from secondary providers.</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: Using the code, architect, ask and help chat modes.</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://aider.chat/docs/faq.html">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://github.com/wild-card-ai/agents-json/tree/master?tab=readme-ov-file).">GitHub - wild-card-ai/agents-json</a>: Contribute to wild-card-ai/agents-json development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/issues/3507">Feature: Squash command · Issue #3507 · Aider-AI/aider</a>: Issue add a /squash that will run diffs on selected commits or a git compat syntax for commit ranges and use the weak model to summarize. What I am also very often doing is creating manual wip comm...</li><li><a href="https://github.com/BerriAI/litellm/issues">BerriAI/litellm</a>: Python SDK, Proxy Server (LLM Gateway) to call 100+ LLM APIs in OpenAI format - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm</li><li><a href="https://aider.chat/docs/usage/commands.html#slash-commands">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1351187346590011432)** (25 messages🔥): 

> `Refact.ai Leaderboard Claim, Claude Harmony Feature, Qwen Models Hype` 


- **Refact.ai Claims Top Spot on Aider's Polyglot Benchmark - Controversy Ensues**: [Refact.ai](https://refact.ai/blog/2025/refact-ai-agent-claude-3-7-sonnet-ranked-1-aider-polyglot/) claimed their **Claude 3.7 Sonnet** powered agent achieved a **76.4%** score on **Aider's polyglot benchmark**, surpassing other models.
   - Paul Gauthier, the creator of Aider, stated that *it's not an appropriate comparison* because Refact.ai used a different, more *agentic* configuration than the standard Aider benchmark which allows for unlimited retries, whereas Aider's previous SWE-bench scores used only one-shot attempts.
- **Anthropic Teases Claude "Harmony" Feature**: A user shared that [Anthropic](https://x.com/testingcatalog/status/1901051432339730603) is coming out with a new **Harmony** feature for **Claude**.
   - The **Harmony** feature will give **Claude** *FULL access to a local directory* so it can research and operate with its content, potentially making it Anthropic's first AI Agent.
- **Qwen Models Get Some Love**: A user commented that *Qwen's models are my favorite* and they might believe them if they say their models are the best.
   - Another user agreed saying that *They're definitely the best for their parameter size*, especially when compared to models in the 7b-32b range.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1901051432339730603">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Early preview of the upcoming Harmony feature for Claude. Harmony will allow users to give Claude FULL access to a local directory so it can research and operate with its content. Is Harm...</li><li><a href="https://refact.ai/blog/2025/refact-ai-agent-claude-3-7-sonnet-ranked-1-aider-polyglot/">Refact.ai Agent + Claude 3.7 Sonnet ranked #1 on Aider's Polyglot Benchmark with a score of 76.4%</a>: Refact.ai Agent, powered by Claude 3.7 Sonnet, has achieved an impressive 76.4% score on Aider's polyglot benchmark — without thinking capabilities enabled.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1350185481521860849)** (458 messages🔥🔥🔥): 

> `GPU support on Llama.cpp, GPU Upgrade Recommendations, Parallel inference Possibilities, OCR Model Recommendation for Mac M3, Gemma 3` 


- ****New OpenCL Backend Boosts Qualcomm Adreno GPUs****: An experimental **OpenCL backend** has been introduced for **Qualcomm Adreno GPUs** in [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/10693), enabling computational power for mobile devices.
- ****4070 Ti owner itches for 5090 Upgrade****: A user with a **4070 Ti** is contemplating upgrading to a **5090**, but due to stock issues, others are recommending waiting or considering a used **RTX 3090** for its **36GB VRAM**.
   - One user suggested that a **used RTX 3090** would provide enough **VRAM** to run *less than 50B @ Q4 models* at reasonable speeds.
- ****Gemma 3's Image Generation Capability Sparks Curiosity****: After experimenting with **Gemma 3 4B**, users found that while it can be prompted to generate images, it produces Imgur links that don't display the actual image.
   - The discussion shifted to identifying local models capable of both **recognizing** and **generating images and text**.
- ****Maximizing M4 Max: Wired Memory Boost****: Users discussed optimizing memory settings on **M4 Max** devices for LM Studio, suggesting adjustments to 'wired' memory allocation for improved GPU performance using a [script](https://gist.github.com/havenwood/f2f5c49c2c90c6787ae2295e9805adbe).
   - The script facilitates adjusting macOS GPU memory limits, allowing users to allocate more memory to the GPU by modifying wired memory settings.
- ****Mistral Small 3.1 launches but requires updates****: Mistral has announced **Mistral Small 3.1** model claiming it outperforms **Gemma 3** and **GPT-4o Mini**, however the release requires conversion to HF format before it can be used in llama.cpp


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.aiclaude.site/pricing">no title found</a>: no description found</li><li><a href="https://modelcontextprotocol.io/introduction">Introduction - Model Context Protocol</a>: no description found</li><li><a href="https://tenor.com/view/very-interesting-arte-johnson-listening-gif-14472867">Very Interesting Arte Johnson GIF - Very Interesting Arte Johnson Listening - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/win10/Pixtral-12B-2409-hf-text-only-Q8_0-GGUF">win10/Pixtral-12B-2409-hf-text-only-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/Les-El/Ollm-Bridge">GitHub - Les-El/Ollm-Bridge: Easily access your Ollama models within LMStudio</a>: Easily access your Ollama models within LMStudio. Contribute to Les-El/Ollm-Bridge development by creating an account on GitHub.</li><li><a href="https://gist.github.com/havenwood/f2f5c49c2c90c6787ae2295e9805adbe">Adjust wired limits to allocate more memory to the GPU with Apple Silicon</a>: Adjust wired limits to allocate more memory to the GPU with Apple Silicon - wired</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fogftn/does_ram_speed_latency_matter_for_llms_benchmarks/">Reddit - Heart of the internet</a>: no description found</li><li><a href="https://huggingface.co/Rombo-Org/reka-flash-3-GGUF_QX_k_Bf16/tree/main">Rombo-Org/reka-flash-3-GGUF_QX_k_Bf16 at main</a>: no description found</li><li><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>: SOTA. Multimodal. Multilingual. Apache 2.0</li><li><a href="https://mistral.ai/news/mistral-small-3),">undefined | Mistral AI</a>: no description found</li><li><a href="https://github.com/coqui-ai/tts">GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production</a>: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production - coqui-ai/TTS</li><li><a href="https://huggingface.co/hexgrad/Kokoro-82M">hexgrad/Kokoro-82M · Hugging Face</a>: no description found</li><li><a href="https://www.techpowerup.com/cpu-specs/ryzen-7-5700g.c2472">AMD Ryzen 7 5700G Specs</a>: Cezanne, 8 Cores, 16 Threads, 3.8 GHz, 65 W</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/10693">Introducing experimental OpenCL backend with support for Qualcomm Adreno GPUs by lhez · Pull Request #10693 · ggml-org/llama.cpp</a>: This PR introduces a new experimental OpenCL backend for Adreno GPUs. Through OpenCL, we can tap into the computational power of Adreno GPUs, which are widely used in many mobile devices, allowing ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1350196032142708867)** (197 messages🔥🔥): 

> `RTX 8000 vs A6000 for LLM inference, Multiple GPUs for running multiple LLMs, 48GB RTX 4090 from China, AMD Strix Halo APU vs RTX 5080 in AI, Mobo/RAM Choice for AI PC build` 


- **RTX 8000 Still Viable for LLM Inferencing**: Members discussed the **RTX 8000 48GB**, noting its decent value for **LLM inferencing** despite being an older Turing architecture card with fewer cuda cores and lower bandwidth compared to newer cards like the **A6000** and **RTX 6000 ADA**.
   - One member stated that *for inferencing, having large VRAM on one card is a huge advantage over two cards with the same VRAM because it eliminates interactions between GPUs that slow each card down by nearly half.*
- **Multiple GPUs for Multiple LLMs: LM Studio Update Coming?**: Members discussed the possibility of running multiple LLMs on separate GPUs using LM Studio, with one member noting that, *right now_ you'd have to use the tensor cuda thing env variable before running up either GPU* with the `CUDA_VISIBLE_DEVICES` environment variable.
   - Another member hinted at a future LM Studio release that will allow setting GPU affinity within the app itself, linking to [a Discord message](https://discord.com/channels/1110598183144399058/1166577236325965844/1349840104914681997) as evidence.
- **Chinese 48GB RTX 4090s: A Cheap VRAM Boost?**: Members discussed sourcing **48GB RTX 4090s** from China, with prices around **$4500**, noting that they use a blower-style fan and consume only two PCIe slots.
   - However, one member cautioned about driver compatibility issues when combining these cards with professional cards like the **A6000**, stating that *the setup only works if I use a gaming driver from NVidia - the studio professional drivers won't load on the 'gaming' cards.*
- **AMD's Strix Halo APU Could Outperform RTX 5080 in AI**: A member shared an article from wccftech claiming AMD's [Ryzen AI MAX+ 395 "Strix Halo" APU](https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-ultimate-ai-pc-apus-16-zen-5-40-rdna-3-5-cores-256-gbps-bandwidth-120w/) *offering over 3x the boost over RTX 5080 in DeepSeek R1 AI benchmarks*.
   - The claim is based on the APU's larger VRAM pool, with the hope that there will be real world data soon.
- **Optimizing Mobo/RAM choice for AI PC build**: A member requested advice on motherboard/RAM choices for an AI PC build, especially regarding potential PCIe lane conflicts with M.2 drives, with a build based on [this pcpartpicker](https://pcpartpicker.com/list/pVMpFZ).
   - Another member suggested that it is not really beneficial to get ram speed faster than **6200** on **AM5** and linked [to a memory kit](https://a.co/d/47pP1DF) while noting that two M.2 drives are mostly idling.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/frankenstein-its-alive-gif-10618052">Frankenstein Its Alive GIF - Frankenstein Its Alive - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/wolf-of-wall-street-jordan-belfort-leonardo-di-caprio-one-of-us-jonah-hill-gif-5441859">One Of Us GIF - Wolf Of Wall Street Jordan Belfort Leonardo Di Caprio - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/logic-engineer-helicopter-chopper-upsidedown-gif-5027455">Logic Engineer GIF - Logic Engineer Helicopter - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/beautiful-amazing-so-beautiful-it-is-what-it-is-gif-22558916">Beautiful Amazing GIF - Beautiful Amazing So Beautiful - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-apu-over-3x-faster-rtx-5080-in-deepseek-benchmarks/">AMD&#039;s Ryzen AI MAX+ 395 &quot;Strix Halo&quot; APU Is Over 3x Faster Than RTX 5080 In DeepSeek R1 AI Benchmarks</a>: AMD has showcased its Ryzen AI MAX+ 395 &quot;Strix Halo&quot; APU offering over 3x the boost over RTX 5080 in DeepSeek R1 AI benchmarks.</li><li><a href="https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306">NVIDIA Quadro RTX 8000 Specs</a>: NVIDIA TU102, 1770 MHz, 4608 Cores, 288 TMUs, 96 ROPs, 49152 MB GDDR6, 1750 MHz, 384 bit</li><li><a href="https://www.ebay.com/itm/116477031617?_skw=4090+48gb&itmmeta=01JPE69HXRKDVZN0X9541KWMYS&hash=item1b1e9274c1:g:QJIAAOSw939nrGSz&itmprp=enc%3AAQAKAAAA8FkggFvd1GGDu0w3yXCmi1fDUKPc34oU6P2kD4Q6nWW6Wkq6G0i12W%2BvQsO3yxeUwFsHxmaxOmaH16Y8wCVsdpsv%2FIPiWlLsGMqkEGTXxCnn7OtypYgyi4CHjPXB0oB2qWJ8utnPVnh4LT9TH4bePDvMrY5xqVQFS9cQ5ZfGbMK%2FWvn7fw7zYraffKanJ%2FQvcGm7o4Sxfc5QknfzbXHSQl91doo762rKufS77tcZ1w4n3pBsGoHds52pRvjMNUygQTMbf2s0S41k27mD5HjOY7poWV3eeuzCwIQhTx03JlzF%2FukwKRxZ8Ltl7FrOWsUGgw%3D%3D%7Ctkp%3ABFBMhJ-mxrNl">OEM 48GB RTX 4090 Founders Edition Dual width GPU Graphics card Ganming/ Server  | eBay</a>: no description found</li><li><a href="https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-apu-over-3x-faster-rtx-5080-in-deepseek-benchma">AMD&#039;s Ryzen AI MAX+ 395 &quot;Strix Halo&quot; APU Is Over 3x Faster Than RTX 5080 In DeepSeek R1 AI Benchmarks</a>: AMD has showcased its Ryzen AI MAX+ 395 &quot;Strix Halo&quot; APU offering over 3x the boost over RTX 5080 in DeepSeek R1 AI benchmarks.</li><li><a href="https://a.co/d/bsNjL2L">Amazon.com: MINISFORUM MS-A1 Mini Workstation Mini Barebone Pc, DDR5/4xM.2 NVMe SSD, Dual 2.5G RJ45 Mini PC,HDMI/DP/Type-C,4xUSB Ports,Supports AMD AM5 CPU,WiFi 6E&amp;BT5.2 Mini Computer(NO CPU/NO RAM/NO SSD/NO OS) : Electronics</a>: no description found</li><li><a href="https://a.aliexpress.com/_EvS423w">no title found</a>: no description found</li><li><a href="https://a.co/d/47pP1DF">CORSAIR Vengeance DDR5 96GB (2x48GB) DDR5 6000MHz CL30 AMD Expo Intel XMP iCUE Compatible Computer Memory – Gray (CMK96GX5M2B6000Z30) at Amazon.com</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1350230436030185472)** (2 messages): 

> `Anthropic Incident, Claude 3.7 Sonnet, Endpoint Quality Measurement` 


- **Anthropic Declares Sonnet's Error Spike Incident Resolved**: Anthropic declared an incident ([status page](https://status.anthropic.com/incidents/qtxnlg9yrwqv)) related to significantly elevated errors for requests to **Claude 3.7 Sonnet** from 21:45–22:14 UTC, Mar 14, 2025.
   - The incident affected **claude.ai**, **console.anthropic.com**, and **api.anthropic.com**.
- **Anthropic Explores Endpoint Quality Gauges**: Anthropic is researching ideas to measure endpoint quality and is open to community input.
   - No commitments were made as the team is *just researching ideas*.



**Link mentioned**: <a href="https://status.anthropic.com/incidents/qtxnlg9yrwqv">Elevated errors for requests to Claude 3.7 Sonnet</a>: no description found

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1350627518603984978)** (4 messages): 

> `Personality.gg Launch, RP Sites and OpenRouter API, Chub and Sillytavern Recommendation` 


- ****Personality.gg** launches new AI character platform**: [Personality.gg](https://personality.gg) launched a new platform to create, chat, and connect with AI characters using models like **Claude**, **Gemini**, and **Personality-v1**, featuring custom themes, full chat control, and NSFW allowance.
   - The platform offers flexible, affordable plans and encourages users to join their [Discord](https://discord.personality.gg) for updates.
- **RP Site Seeks OpenRouter API Support**: A member inquired about roleplay (RP) or novel sites that support the OpenRouter API, expressing dissatisfaction with Novelcrafter's stability and Janitor AI's context limitations.
   - They cited NovelAI always crashing and **Janitor AI** limited to *only 128k context* as reasons for seeking alternatives.
- **Chub and Sillytavern advised for RP**: A member recommended **Chub** or **Sillytavern** (local web frontend) as alternatives for roleplaying.
   - The member positioned **Sillytavern** as a *local webend* option to overcome the limitations of other platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.personality.gg>">no title found</a>: no description found</li><li><a href="https://personality.gg>">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1350182784681115689)** (443 messages🔥🔥🔥): 

> `Gemma 3, RP models, Mistral Small 3.1, OpenRouter OpenAPI spec, Reasoning Tokens` 


- **Parasail Hosts New RP Models**: Parasail is looking to host new roleplay models on OpenRouter and is proactively working with creators like TheDrummer to host new fine-tunes of models like **Gemma 3** and **QwQ**.
   - They are seeking individuals who create strong RP fine-tunes capable of handling complex instructions and worlds, with a particular interest in models that have been fine-tuned for roleplay and creative writing.
- **Anthropic API Outage Disrupts Claude 3 Sonnet**: Requests to **Claude 3.7 Sonnet** experienced significantly elevated errors for approximately 30 minutes, as reported on [Anthropic's status page](https://status.anthropic.com/incidents/qtxnlg9yrwqv).
   - The issue has been resolved, and success rates have returned to normal as of March 14, 2025, but some users experienced no text on replies while still being charged.
- **OpenRouter API Rate Limits Explained**: OpenRouter's rate limits depend on your credits, with approximately **1 USD** equating to **1 RPS** (requests per second), as clarified in [the documentation](https://openrouter.ai/docs/api-reference/limits).
   - Users can check their rate limit and remaining credits by making a GET request to `https://openrouter.ai/api/v1/auth/key`, and while higher credit purchases enable higher rate limits, creating additional accounts or API keys *makes no difference*.
- **New Steelskull L3.3 R1 70B Model Launches**: A new roleplaying model, **Steelskull L3.3 R1 70B**, has launched on OpenRouter, incorporating several models like [TheSkullery's L3.1x3.3-Hydroblated-R1-70B-v4.4](https://huggingface.co/TheSkullery/L3.1x3.3-Hydroblated-R1-70B-v4.4).
   - The announcement encourages users to provide feedback on desired models, continuing the push for competitively priced RP options.
- **Mistral Small 3.1 Available**: The Mistral Small 3.1 24B Instruct model has launched on OpenRouter, featuring **multimodal capabilities** and a **128k context window**, according to [Mistral's announcement](https://mistral.ai/news/mistral-small-3-1).
   - It outperforms comparable models like Gemma 3 and GPT-4o Mini, while delivering inference speeds of 150 tokens per second.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/anthropicai/status/1900234837283197122">Tweet from Anthropic (@AnthropicAI)</a>: We&#39;ve made several updates to the Anthropic API that help developers process more requests and reduce token usage with Claude 3.7 Sonnet.</li><li><a href="https://x.com/baidu_inc/status/1901089355890036897?s=46">Tweet from Baidu Inc. (@Baidu_Inc)</a>: We&#39;ve just unveiled ERNIE 4.5 & X1! 🚀As a deep-thinking reasoning model with multimodal capabilities, ERNIE X1 delivers performance on par with DeepSeek R1 at only half the price. Meanwhile, ERNI...</li><li><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>: SOTA. Multimodal. Multilingual. Apache 2.0</li><li><a href="https://tenor.com/bMeOD.gif">So Boring Gill GIF - So Boring Gill Engvid - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/bupRk.gif">Why Whyyy GIF - Why Whyyy Neden - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/mistralai/mistral-small-3.1-24b-instruct-2503">Mistral Small 3.1 24B - API, Providers, Stats</a>: Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. Run Mistral Small 3.1 24B with API</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://status.anthropic.com/incidents/qtxnlg9yrwqv">Elevated errors for requests to Claude 3.7 Sonnet</a>: no description found</li><li><a href="https://openrouter.ai/google/gemma-3-27b-it:free/api).">Google: Gemma 3 27B (free)</a>: Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, ...</li><li><a href="https://openrouter.ai/rankings/programming?view=week">LLM Rankings: programming | OpenRouter</a>: Language models ranked and analyzed by usage for programming prompts</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: Language models ranked and analyzed by usage across apps</li><li><a href="https://parasail.canny.io/model-request">Model Request | Parasail</a>: Request Models - Please Put in the Hugging Face Model and any other information!</li><li><a href="https://llm-stats.com">LLM Leaderboard 2025 - Compare LLMs</a>: Comprehensive AI (LLM) leaderboard with benchmarks, pricing, and capabilities. Compare leading LLMs with interactive visualizations, rankings and comparisons.</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">Reasoning Tokens - Improve AI Model Decision Making</a>: Learn how to use reasoning tokens to enhance AI model outputs. Implement step-by-step reasoning traces for better decision making and transparency.</li><li><a href="https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences">Provider Routing - Smart Multi-Provider Request Management</a>: Route AI model requests across multiple providers intelligently. Learn how to optimize for cost, performance, and reliability with OpenRouter&#x27;s provider routing.</li><li><a href="https://huggingface.co/BigHuggyD/SteelSkull_L3.3-Electra-R1-70b-FP8-Dynamic">BigHuggyD/SteelSkull_L3.3-Electra-R1-70b-FP8-Dynamic · Hugging Face</a>: no description found</li><li><a href="https://web.archive.org/web/20250108130531/https://openrouter.ai/anthropic/claude-3.5-sonnet/parameters">Anthropic: Claude 3.5 Sonnet – Recommended Parameters</a>: Check recommended parameters and configurations for Anthropic: Claude 3.5 Sonnet - New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. S...</li><li><a href="https://github.com/openai/openai-python">GitHub - openai/openai-python: The official Python library for the OpenAI API</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://github.com/openai/openai-openapi">GitHub - openai/openai-openapi: OpenAPI specification for the OpenAI API</a>: OpenAPI specification for the OpenAI API. Contribute to openai/openai-openapi development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/)** (1 messages): 

eofr: Scam
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1351270127617966130)** (1 messages): 

> `Perplexity Accuracy, Perplexity Video Ad` 


- **Perplexity Guarantees Accuracy**: A member shared the slogan *When you need to get it right, ask Perplexity.*
- **Perplexity Shares Video Ad**: A member posted a [video ad for Perplexity](https://cdn.discordapp.com/attachments/1047204950763122820/1351270126615396454/lSdoFFbL6lXL_huQ.mp4?ex=67d9c3df&is=67d8725f&hm=046721b4226c4142a36a9fc331a82a120a744c64bacfae63ac90d96721381065&).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1350186088190185642)** (409 messages🔥🔥🔥): 

> `Perplexity Pro Oyster Game, Discord Pro Role, Gemini 2 Flash Context, Claude 3.7 Sonnet Limit, AI Coding Models` 


- **Oyster Game rewards diligent Perplexity Users**: Perplexity users on Windows can now get a free **1 month of Perplexity Pro** by using the app for **7 consecutive days**.
- **Discord Pro Role Causes Dilemma**: Users are having trouble accessing the **Pro channels** despite having a **Perplexity Pro subscription**.
   - To fix this, users are recommended to *leave the server and rejoin via the Discord link in their Perplexity Pro settings*.
- **Users debate Gemini 2 Flash Context window issues**: Users debate the **context retention capabilities of Gemini 2 Flash**, claiming it has a **1M context window** but performs worse than regular Gemini.
   - One user notes that it *forgets the formatting* after a few messages while making flashcards.
- **Figuring out Claude 3.7 Sonnet Limits**: Users clarify that the usage limit for **Claude 3.7 Sonnet** with a **Perplexity Pro subscription** is **500 queries per day**, but it is shared across all models except GPT 4.5.
   - They also add that the context limit might be slightly more than on Anthropic's site, but the response context limit is smaller at *4000 or 5000 tokens*.
- **Deciphering best AI Model for Coding**: Users are seeking advice on the **best AI model for coding**, with recommendations leaning towards **Claude 3.7 Reasoning**.
   - One user finds that **Deepseek R1** has a *high hallucination rate*, making it unsuitable for summarizing documents.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fixvx.com/AravSrinivas/status/1901092875758371246">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/baidu_inc/status/1901089355890036897?s=46">Tweet from Baidu Inc. (@Baidu_Inc)</a>: We&#39;ve just unveiled ERNIE 4.5 & X1! 🚀As a deep-thinking reasoning model with multimodal capabilities, ERNIE X1 delivers performance on par with DeepSeek R1 at only half the price. Meanwhile, ERNI...</li><li><a href="https://github.com/vectara/hallucination-leaderboard">GitHub - vectara/hallucination-leaderboard: Leaderboard Comparing LLM Performance at Producing Hallucinations when Summarizing Short Documents</a>: Leaderboard Comparing LLM Performance at Producing Hallucinations when Summarizing Short Documents - vectara/hallucination-leaderboard
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1350328478737891459)** (32 messages🔥): 

> `Quantum Chip, Willow, Vibe Coding, Lunar Lander, Dark Matter` 


- **Chinese Quantum Chip Rivals Willow**: Perplexity AI highlights a [YouTube video](https://www.youtube.com/embed/c540dAQ5Hf4) about a Chinese **quantum chip** rivaling **Willow**, the rise of **Vibe Coding** in software development, and discoveries about the universe.
- **Amazon Ends Echo Privacy Options**: Perplexity AI references a page about [Amazon ending **Echo privacy options**](https://www.perplexity.ai/page/amazon-ends-echo-privacy-optio-7QEG1EcHS.W8lb3W5YOCrQ).
- **Lunar Lander Catches Eclipse**: A link was shared about a [Lunar Lander capturing an eclipse](https://www.perplexity.ai/page/lunar-lander-captures-eclipse-wJSf1n_ISE65XWYScHXhnw).
- **New Dark Matter at Milky Way's**: A page discussing [new dark matter at Milky Way's](https://www.perplexity.ai/page/new-dark-matter-at-milky-ways-Lpo5SP1uSkGI7O9Kf.6Q0A) was shared.
- **Vibe Coding's Rise in Software**: A link was shared to a page discussing [Vibe Coding's Rise in Software](https://www.perplexity.ai/page/vibe-coding-s-rise-in-software-.OYRvZGhSlGYIqjRND04fA).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1350529890323271771)** (5 messages): 

> `Transferring Credits, API Pay-as-you-go Limits, Sonar Reasoning Pro Limits, French Translation` 


- **User Queries Credit Transfers**: A user inquired whether it's possible to **transfer credits** to another user within the platform.
   - The same user also questioned the availability of **unlimited pay-as-you-go deep-research** options through the API, particularly for applications experiencing huge bursts of bulk requests.
- **Sonar Reasoning Pro has Image limits**: A user reported that the **sonar-reasoning-pro API** only returns a maximum of **5 images**.
   - They are asking if this limit is configurable or a hard constraint, as they found nothing about it in the documentation.
- **User Asks for Help with French Translation**: A user inquired about how to integrate a **French translator** within the Perplexity AI platform.
   - No solution was offered in the channel.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1350183272969273424)** (356 messages🔥🔥): 

> `Rust Community Toxicity, C vs C++, Optimization vs Search, Stochastic Differential Equations` 


- **Rust Faces Toxicity Accusations**: Members debated the toxicity of the **Rust community**, with some saying the organization is imploding and others comparing it to the Ruby community.
   - One member stated, *The Rust community is pretty toxic. The org has kinda imploded on themselves recently.*
- **C's Brokenness Debated**: A member described C as ancient, broken, and garbage while another argued that C is not broken, highlighting its use in international standards and various hardware platforms.
   - A member linked to [faultlore.com](https://faultlore.com/blah/c-isnt-a-language/) arguing that *C Isn't A Programming Language Anymore*.
- **Optimization vs Search Unpacked**: Members discussed the difference between **optimization** (finding the maximal or minimal value of a function) and **search** (finding the best element of a set).
   - One member stated that *search is exploration, not like optimization*. And another stated *the process of designing or selecting that model—choosing the architecture, tuning learning rates, etc.—has a search-like flavor*.
- **Stochastic Processes Explored**: A member offered to give an introduction to **stochastic processes**, stochastic differential equations, and the derivation of the time-reversal SDE used in diffusion-based AI architectures.
   - The member planned to cover the foundations of **stochastic processes**, Wiener processes, and what a Stochastic Differential Equation is.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/willccbb/status/1901415166295544154?t=HmQDRR0NQ9mi_4udIiT4uQ&s=19">Tweet from will brown (@willccbb)</a>: uhhh there are a lot of LLM RL libraries out there...</li><li><a href="https://en.wikipedia.org/wiki/Reparameterization_trick#Variational_autoencoder">Reparameterization trick - Wikipedia</a>: no description found</li><li><a href="https://faultlore.com/blah/c-isnt-a-language/">C Isn't A Programming Language Anymore - Faultlore</a>: no description found</li><li><a href="https://www.iso-9899.info/wiki/The_Standard">The Standard - C</a>: no description found</li><li><a href="https://github.com/pyca/cryptography/issues/5771">Dependency on rust removes support for a number of platforms · Issue #5771 · pyca/cryptography</a>: I would like to report that the newly added dependency on Rust has made it impossible to package cryptography for a number of supported Gentoo architectures (and these are architectures where peopl...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1350584655136555009)** (4 messages): 

> `LLM Literature Review, Gemma 3 Model` 


- **Seeking LLM Lit-Review Legitimacy**: A member asked for a good paper for a literature review on **LLMs** and pointed to a [blogpost](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-e41) of top AI papers.
- **Gemma 3 Gazes with Grandeur**: **Gemma 3** is a lightweight open model family (**1B–27B parameters**) that integrates **vision understanding, multilingual coverage, and extended context windows** (up to **128K tokens**).
   - It incorporates a frozen **SigLIP vision encoder**, condensing images into **256 soft tokens** and has a new **Pan & Scan (P&S)** method, watch the [YouTube video](https://www.youtube.com/watch?v=n5nEd600iM0).



**Link mentioned**: <a href="https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-e41">🥇Top AI Papers of the Week</a>: The Top AI Papers of the Week (Mar 10 - 16)

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1350380011605200947)** (59 messages🔥🔥): 

> `AI Safety Institute ideological bias, Deepseek R2 release and its issues, SesameAILabs CSM model disappointment, Hallucination in AI search engines, Mistral Small 3.1 release` 


- ****AI Safety Institute** Pushes Ideological Alignment**: The **National Institute of Standards and Technology (NIST)** instructed partners of the **US Artificial Intelligence Safety Institute (AISI)** to deprioritize **AI safety**, **responsible AI**, and **AI fairness**, focusing instead on *reducing ideological bias* and prioritizing *human flourishing and economic competitiveness* as reported in [Wired](https://www.wired.com/story/ai-safety-institute-new-directive-america-first/).
- ****Deepseek R2** Hype Train Derailed?**: A member shared a [Reddit post](https://www.reddit.com/r/NvidiaStock/comments/1j822zl/deepseek_r2_will_be_released_on_17th_of_march/) about the release of **Deepseek R2**, noting its potential impact on **Nvidia** stock.
   - However, some users found the model underwhelming, particularly its Text-to-Speech (TTS) capabilities, describing it as *not a true speech-speech model* and experiencing inconsistent voice generation on Mac.
- ****SesameAILabs' CSM** Falls Short of Expectations**: Users expressed disappointment with the released small model of **SesameAILabs' CSM**, citing numerous bugs and a significant performance gap compared to the demos, reported in this [Github issue](https://github.com/SesameAILabs/csm/issues/63).
   - The released model is criticized for poor punctuation handling and slow performance, raising doubts about the future release of larger, more promising models.
- **AI Search Engines Hallucinate News, Researchers Find**: A report by the [Columbia Journalism Review](https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php) found high hallucination rates across multiple AI search engines, including **Perplexity**, **ChatGPT**, and **Grok**, in citing news sources.
   - Notably, premium models like **Perplexity Pro** and **Grok 3** exhibited *higher error rates* despite their enhanced capabilities and cost.
- ****Mistral Small 3.1** Claims Top Spot in Weight Class**: **Mistral AI** announced the release of [Mistral Small 3.1](https://mistral.ai/fr/news/mistral-small-3-1), boasting improved text performance, multimodal understanding, and a **128k** token context window under an Apache 2.0 license.
   - The company claims it outperforms comparable models like **Gemma 3** and **GPT-4o Mini**, with inference speeds of **150 tokens per second**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Obsidian_(software)">Obsidian (software) - Wikipedia</a>: no description found</li><li><a href="https://mistral.ai/fr/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>: SOTA. Multimodal. Multilingual. Apache 2.0</li><li><a href="https://tenor.com/view/xrd-exrd-crypto-btc-eth-gif-23801255">Xrd Exrd GIF - Xrd Exrd Crypto - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.wired.com/story/ai-safety-institute-new-directive-america-first/">Under Trump, AI Scientists Are Told to Remove ‘Ideological Bias’ From Powerful Models</a>: A directive from the National Institute of Standards and Technology eliminates mention of “AI safety” and “AI fairness.”</li><li><a href="https://www.reddit.com/r/NvidiaStock/comments/1j822zl/deepseek_r2_will_be_released_on_17th_of_march/">Reddit - Heart of the internet</a>: no description found</li><li><a href="https://github.com/SesameAILabs/csm/issues/63">the model is just terrible and full of bugs · Issue #63 · SesameAILabs/csm</a>: after the mind blowing demos i tried the model on hugging face and the results are so terrible and full of errors that i am just so disappointed. Nearly all punctuations are pronounced wrong or wit...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1350424475421900881)** (1 messages): 

> `SmolVLM2, Gradio Sketch 2.0, DCLM-Edu Dataset, huggingface.js GGUF metadata, Robot Arms for $299` 


- **SmolVLM2 Released, Smallest VLM Ever**: The team released [SmolVLM2](https://x.com/pcuenq/status/1896632829372715442), the smallest VLM that can understand videos and runs flawlessly on an iPhone app with its **500M version**.
   - Source code and a TestFlight beta are available for reference.
- **Gradio Sketch 2.0: No-Code App Building**: [Gradio Sketch 2.0](https://x.com/abidlabs/status/1897782056308142266) is out, supporting complete Gradio apps with events, all without writing any code.
   - The new features enable users to build applications via the GUI.
- **DCLM-Edu Dataset Released**: A new dataset, [DCLM-Edu](https://x.com/LoubnaBenAllal1/status/1898044807928295808), was released; it's a filtered version of DCLM using FineWeb-Edu’s classifier, optimized for smol models like **SmolLM2 135M/360M**.
   - The purpose is that *small models are sensitive to noise and can benefit from heavily curated data*.
- **Gemma 3 is Live, Deployable from HF endpoints**: [Gemma 3](https://x.com/ErikKaum/status/1899784006247284841) is live and can be deployed directly from Hugging Face endpoints with optimally selected hardware and configurations.
- **Agents Course Now Diversifying with LlamaIndex**: The [agents course](https://x.com/ben_burtenshaw/status/1898761949036593637) is expanding with a unit on LlamaIndex, covering topics like LlamaHub integrations, agents and tools in LlamaIndex, and multi-agent workflows.
   - Unit 2 will prepare you the real world use cases in unit 3. *Where you can use the framework of your choosing.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/pcuenq/status/1896632829372715442">Tweet from Pedro Cuenca (@pcuenq)</a>: Last week we released SmolVLM2, the smallest VLM that can understand videos on your phone 🔥This is not a figure of speech, we wrote an iPhone app that runs the 500M version flawlessly.Today, we relea...</li><li><a href="https://x.com/abidlabs/status/1897782056308142266">Tweet from Abubakar Abid (@abidlabs)</a>: BOOM! Gradio Sketch 2.0 is out with support building complete Gradio apps, including adding events, without writing a single line of code</li><li><a href="https://x.com/LoubnaBenAllal1/status/1898044807928295808">Tweet from Loubna Ben Allal (@LoubnaBenAllal1)</a>: 🚀 New dataset drop: DCLM-EduWe filtered DCLM using FineWeb-Edu’s classifier to create a cleaner dataset optimized for smol models (like SmolLM2 135M/360M).Why? Small models are sensitive to noise and...</li><li><a href="https://x.com/julien_c/status/1895577975036465166">Tweet from Julien Chaumond (@julien_c)</a>: In other news...New feature in huggingface.js: You can now list metadata and tensors of GGUF files from our CLI using npx command (supports both local + remote GGUF)Try it: npx @huggingface/gguf your_...</li><li><a href="https://x.com/RemiCadene/status/1895048737300586674">Tweet from Remi Cadene (@RemiCadene)</a>: Get our 2 robot arms for $299 assembled 🤯For $199 only, you can get 3d printed parts and motors!!!It&#39;s also nice to assemble it yourself (or with your kids 👶 if you have any)https://shop.wowrobo...</li><li><a href="https://x.com/RisingSayak/status/1899029374118293860">Tweet from Sayak Paul (@RisingSayak)</a>: New quantization backend in Diffusers.It supports torch.compile() partially and works great in the eager model.Go check it out. No, I am not going to provide any links. Do the hard work.</li><li><a href="https://x.com/ErikKaum/status/1899784006247284841">Tweet from Erik Kaunismäki (@ErikKaum)</a>: Gemma 3 is live 🔥You can deploy it from @huggingface endpoints directly with an optimally selected hardware and configurations.Give it a try 👇</li><li><a href="https://x.com/julien_c/status/1897704181160419740">Tweet from Julien Chaumond (@julien_c)</a>: It&#39;s live!</li><li><a href="https://x.com/ClementDelangue/status/1897666379823669667">Tweet from clem 🤗 (@ClementDelangue)</a>: IMO, academia has a massive role to play to make AI a positive force, not only dominated by $$$ interests but driven by science progress & public good!We&#39;re trying to help as we can with Academia ...</li><li><a href="https://x.com/NielsRogge/status/1898792935069487121">Tweet from Niels Rogge (@NielsRogge)</a>: In case you missed it - we updated paper pages on @huggingface so that authors can now add their Github and/or project page URLsWe&#39;d like to make http://hf.co/papers become THE place where people ...</li><li><a href="https://x.com/julien_c/status/1897007199517597794">Tweet from Julien Chaumond (@julien_c)</a>: Happy to announce we&#39;ve added @jfrog as a model scanning partner on the @huggingface Hub! 🔥Shedding more light on AI x Security is a win for everyone 🤝</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1: Update #3</a>: no description found</li><li><a href="https://x.com/ben_burtenshaw/status/1898761949036593637">Tweet from Ben Burtenshaw (@ben_burtenshaw)</a>: The agents course is diversifying with a unit on LlamaIndex! If this is you go to framework, check out the course now.The unit covers these topics:- What makes llama-index stand-out- How the LlamaHub ...</li><li><a href="https://x.com/maximelabonne/status/1896594006324244680">Tweet from Maxime Labonne (@maximelabonne)</a>: I partnered with @huggingface  and @ben_burtenshaw  to teach people how to fine-tune LLMs with GRPO.In this notebook, we fine-tune a tiny SmolLM-135M model on my filtered smoltldr dataset. Thanks to o...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1350181844301385790)** (141 messages🔥🔥): 

> `Dou Shou Qi AI, Stable Diffusion Model, CSM Streaming Generator, Gemini 2.0 Flash Experimental, Hunyuan 3D-2 API` 


- **Models Duel in Dou Shou Qi, AI Triumphs!**: Only one model made no illegal moves in [Dou Shou Qi](https://en.wikipedia.org/wiki/Jungle_(board_game)), a game that is a *tough beast to crack for AI, but easy for humans*.
   - A member suggested that you can use *any means* possible to train it, even farming expert/master human games, but keep in mind that **stockfish** was made for classic european chess, and **Dou Shou Qi** is a *totally different game*.
- **Hackathon Hoopla for Budding Brains**: AI developers are seeking recommendations for [global hackathons](https://huggingface.co/spaces), aiming to connect with individuals worldwide and engage in impactful **AI-focused** events.
   - Participants are eager to explore innovative solutions and collaborate with like-minded experts in the **AI community**.
- **MCP Servers Get Love, Span Across Front Products**: Members discussed the use of [MCP Servers](https://www.parseable.com/blog/mcp-better-alternative-to-rag-for-observability) for tools, and how it's implemented in Claude and ChatGPT.
   - An enthusiast made an actual robot using Arduino ESP 32 and controlled it with Claude AI MDC protocol, very impressed with what all we can do with AI.
- **Inspirit AI Seeks New Recruits for Summer 2025**: Gabriel Salem shares that they were accepted to the [Inspirit AI Ambassador program](http://www.inspiritai.com/?utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8494Tt7s3T_Sf0UkHfsKbWZ2sT6UqBhva5AM_GV1OQNbtNiVt2DLE34mBQKK8WXfF9DKSR), offering AI fundamentals and project-building for middle and high school students.
   - The program guides students in building socially impactful projects such as **self-driving car simulation, exoplanet detection, and criminal justice**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lerobothf">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/CharaspowerAI/status/1901279916240580613">Tweet from Pierrick Chevallier | IA (@CharaspowerAI)</a>: The Elements feature in @Kling_ai  is seriously powerful, yet so many people overlook it. 🤯🔥With some time, you can push creativity really far. Who’s using it here? 👀</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct">Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheDrummer/Gemmasutra-Mini-2B-v1">TheDrummer/Gemmasutra-Mini-2B-v1 · Hugging Face</a>: no description found</li><li><a href="https://www.parseable.com/blog/mcp-better-alternative-to-rag-for-observability">Is MCP a better alternative to RAG for Observability?</a>: no description found</li><li><a href="https://ollama.com/download/mac">Download Ollama on macOS</a>: Download Ollama for macOS</li><li><a href="https://huggingface.co/docs/hub/ollama">Use Ollama with any GGUF Model on Hugging Face Hub</a>: no description found</li><li><a href="https://huggingface.co/blog/yagilb/lms-hf">Use Models from the Hugging Face Hub in LM Studio</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501">mistralai/Mistral-Small-24B-Instruct-2501 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://huggingface.co/posts/bartowski/928757596721302">@bartowski on Hugging Face: &quot;Decided to try to check how many weights in a 70b F32 model would be squashed…&quot;</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comme">Reddit - Heart of the internet</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/">Reddit - Heart of the internet</a>: no description found</li><li><a href="https://github.com/ollama/ollama/issues/2833">Running ollama on Hugging Face Spaces · Issue #2833 · ollama/ollama</a>: I want to run ollama on Hugging Face Spaces, because I run a Streamlit app there that must make use of a LLM and a embedding model served by Ollama. How can I do that?</li><li><a href="https://github.com/cappuch/ml-math/blob/main/mlmath_internal.h#L102-L110">ml-math/mlmath_internal.h at main · cappuch/ml-math</a>: Contribute to cappuch/ml-math development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://github.com/NVIDIA/nvidia-container-toolkit/issues/155">Getting the following error trying to run an Nvidia/Cuda container in Windows 10: Auto-detected mode as &#39;legacy&#39; nvidia-container-cli: initialization error: WSL environment detected but no adapters were found: unknown · Issue #155 · NVIDIA/nvidia-container-toolkit</a>: I can&#39;t find info on this error anywhere! I am running Docker Desktop with WSL. My docker compose file looks like this: version: &#39;3&#39; services: app: container_name: &quot;sd&quot; build: . ...</li><li><a href="https://docs.nvidia.com/cuda/archive/12.5.0/wsl-user-guide/index.html">CUDA on WSL</a>: no description found</li><li><a href="https://huggingface.co/open-r1/OlympicCoder-32B">open-r1/OlympicCoder-32B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1: Update #3</a>: no description found</li><li><a href="https://github.com/id-Software/Quake-III-Arena/blob/master/code/game/q_math.c#L552C1-L565C1">Quake-III-Arena/code/game/q_math.c at master · id-Software/Quake-III-Arena</a>: Quake III Arena GPL Source Release. Contribute to id-Software/Quake-III-Arena development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/multimodalart/flux-fill-outpaint">Flux Fill Outpainting - a Hugging Face Space by multimodalart</a>: no description found</li><li><a href="https://huggingface.co/spaces?category=image-editing&sort=trending">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/372">huggingchat/chat-ui · [MODELS] Discussion</a>: no description found</li><li><a href="https://github.com/orgs/huggingface/repositories">Hugging Face</a>: The AI community building the future. Hugging Face has 300 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/gradio-app/gradio/issues">gradio-app/gradio</a>: Build and share delightful machine learning apps, all in Python. 🌟 Star to support our work! - gradio-app/gradio</li><li><a href="https://github.com/huggingface/hub-docs/issues">huggingface/hub-docs</a>: Docs of the Hugging Face Hub. Contribute to huggingface/hub-docs development by creating an account on GitHub.</li><li><a href="https://inspiritai.co/Summer-2025-Interest-Form">Inspirit AI Summer 2025 Program - Learn More!</a>: Thank you for your interest in the Inspirit AI Scholars program taught by Stanford, MIT and Ivy League graduate students. Please fill out this short form to receive more information about our Summer 2...</li><li><a href="https://drive.google.com/file/d/1MJOaADMPuDXfQ5QeLraFe6gJoDrjVcrV/view">Building AI Projects for High School Students - Inspirit AI.pdf</a>: no description found</li><li><a href="http://www.inspiritai.com/?utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8494Tt7s3T_Sf0UkHfsKbWZ2sT6UqBhva5AM_GV1OQNbtNiVt2DLE34mBQKK8WXfF9DKSR">Inspirit AI: AI High School Program taught by Stanford/MIT Alums</a>: Inpsirit AI Scholars is an artificial intelligence program for high school students, developed and taught by Stanford and MIT alumni and graduate students. Work on programming projects, prep for codin...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1350688028288352370)** (3 messages): 

> `ML for 3D, HuggingFace Agents course, Retrievel agent` 


- **Diving into Dimension: ML for 3D Begins**: A machine learning engineer is embarking on a **ML for 3D course** today.
   - They also offered to recommend some courses.
- **Smol Agents Framework Completion**: A member is learning from the **HuggingFace Agents course** and has completed the first framework, **smolagents**.
   - They shared their excitement about this achievement.
- **Retrievel Agent: New Learning Frontier**: Another member is currently learning about the **Retrievel agent** from the **Agents course**.
   - This indicates ongoing engagement with and exploration of the course's content.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1350369274014728213)** (2 messages): 

> `Cross-posting` 


- **User Criticizes Cross-Posting of YouTube Link**: A user shared a [YouTube link](https://www.youtube.com/watch?v=n0OwGSX2IiQ) and another user immediately criticized them for cross-posting.
   - The second user explicitly stated, *"I've already asked you not to cross-post and to keep posts in topic."
- **Request to Keep Posts in Topic**: Following the sharing of a [YouTube link](https://www.youtube.com/watch?v=n0OwGSX2IiQ), a user requested that posts be kept in topic.
   - This suggests a concern about the relevance of the shared link to the channel's main discussion.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1350200457578020924)** (5 messages): 

> `Awesome Vibe Coding, Local LLMs setup, FluxHands-FingerCount Dataset` 


- ****Coding Vibes** with AI Get an Awesome List**: An "Awesome Vibe Coding" list was announced with [tools, editors, and resources](https://github.com/filipecalegario/awesome-vibe-coding) that make AI-assisted coding more intuitive and efficient.
   - The list includes AI-powered IDEs & code editors, browser-based tools, plugins & extensions, command line tools, and latest news & discussions.
- **Local LLMs Assist Coding**: A member wrote an article on [how to set up free local coding AI assistant for VS Code](https://horosin.com/how-to-set-up-free-local-coding-ai-assistant-for-vs-code) and tested it this week.
- **Dataset Counts Fingers**: A dataset of hands with various numbers of fingers, named [FluxHands-FingerCount](https://huggingface.co/datasets/taesiri/FluxHands-FingerCount) was created and manually labeled.
   - Each image contains a human hand in the center, rendered in different styles, and was generated using **Flux**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/merterbak/gemma-3">Gemma 3 - a Hugging Face Space by merterbak</a>: no description found</li><li><a href="https://github.com/mahimairaja/awesome-csm-1b">GitHub - mahimairaja/awesome-csm-1b: List of curated use cases built using Sesame&#39;s CSM 1B</a>: List of curated use cases built using Sesame&#39;s CSM 1B - mahimairaja/awesome-csm-1b</li><li><a href="https://github.com/filipecalegario/awesome-vibe-coding">GitHub - filipecalegario/awesome-vibe-coding: A curated list of vibe coding references, collaborating with AI to write code.</a>: A curated list of vibe coding references, collaborating with AI to write code. - filipecalegario/awesome-vibe-coding</li><li><a href="https://huggingface.co/datasets/taesiri/FluxHands-FingerCount">taesiri/FluxHands-FingerCount · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

coldbreeze.: Free fire
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1350677563105869854)** (4 messages): 

> `Autonomous Driving blogpost, VLMs Research Hub, HF DETR model, Meta's Segment Anything Model (SAM)` 


- **Autonomous Driving Blogpost Released**: A member announced the completion of their blog post on **autonomous driving**, covering **modular pipelines vs end-to-end approaches and LLMs**, and shared a link to the [Medium article](https://medium.com/@samiratra95/autonomous-driving-modular-pipeline-vs-end-to-end-and-llms-642ca7f4ef89).
   - They asked for thoughts and feedback on the content.
- **Vision-Language Models (VLMs) Research Hub Launched**: A member announced the creation of a community-driven hub for **multimodal researchers** working on **Vision-Language Models (VLMs)** at [this github repo](https://github.com/thubZ09/vision-language-model-hub).
   - The hub will be updated weekly and welcomes contributions and suggestions.
- **Backbone Swapping in HF DETR Model**: A member inquired about successfully swapping the **Backbone** to, for example, **ViT** in the **Hugging Face DETR model**.
   - No solutions or suggestions were provided.
- **SAM Fine-Tuning**: A member inquired about fine-tuning **Meta's Segment Anything Model (SAM)**.
   - No solutions or suggestions were provided.



**Link mentioned**: <a href="https://github.com/thubZ09/vision-language-model-hub">GitHub - thubZ09/vision-language-model-hub: Hub for researchers exploring VLMs and Multimodal Learning:)</a>: Hub for researchers exploring VLMs and Multimodal Learning:)  - GitHub - thubZ09/vision-language-model-hub: Hub for researchers exploring VLMs and Multimodal Learning:)

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1350683165726871664)** (2 messages): 

> `SetFit with LoRA, SmolLM as teacher model` 


- **Train Embedding Model using SetFit & LoRA**: A member inquired about training an embedding model with **LoRA** adapters via **SetFit**.
- **SmolLM Distillation Idea**: A member mentioned juggling with the idea of using something like **SmolLM** as a teacher model for distillation.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1350796789891993651)** (3 messages): 

> `smol-course, HuggingFace Agents course, HF inference credits` 


- **Smol-Course differs from HF Agents course**: A member asked if the **smol-course** is different from the **HuggingFace Agents course** to which another member confirmed they are different.
   - That member noted that the **Agents course Discord** is missing and that every single code notebook was broken, suggesting skipping the course.
- **HF Inference Credits cost course participation**: A member reported that the HuggingFace Agents course asked for **HF inference credits** for money, even though the course claimed to be free.
   - The member understood that **API calls cost money**, but suggested they should have developed the full course within the context of free credits.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1350207795596759060)** (134 messages🔥🔥): 

> `Agentic AI team building, Smolagents and Gemma3 issues, Ollama Context Length, HF Course Verification problems, MCP and Smolagent framework` 


- **AI Enthusiasts Unite to Build Agentic AI Together**: Several members including Bijen, tariqbinbashar, Madhusudhan, and Salaar expressed interest in **collaborating on agentic AI projects** to solve business problems and enhance their knowledge.
   - The call to action aims to form teams and build qualified AI Agents for American consumers and learn together.
- **Gemma3 struggles with Smolagents Regex Patterns**: A member ran into the *dreaded* regex pattern error while using **gemma3:12b** with smolagents, suspecting either a model issue or a bug with **Ollama integration through LiteLLM/OpenAI**.
   - The user eventually solved the issue by increasing the **Ollama context length**.
- **Ollama Context Issue Sorted**: A member discovered that **Ollama** was truncating input due to context-token limits, impacting the functionality of **smolagents**.
   - The fix involves setting the environment variable `$env:OLLAMA_CONTEXT_LENGHT=8192` to achieve much better results.
- **HF Course Verification Redirect Loop**: Several users reported issues with the **Hugging Face Discord verification process**, encountering a redirect loop even after following the steps in the relevant channel.
   - A user suggested ensuring the link between the Hugging Face account and Discord is properly established, while another said to keep trying until it works.
- **Smolagents and potential of MCP Integration**: A member expressed that using **VLM and MCP in the smolagent framework** could create robust agents and hoped these would be added as a unit in the course.
   - The discussion evolved into how to reuse tools implemented for one agentic framework into another, and if **MCP** was indeed the best option for this.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:11434")`">no title found</a>: no description found</li><li><a href="https://open.spotify.com/playlist/4J61XoHr2CINqRA1DV0ga7">Party on, Wayne!</a>: Playlist · Music League · 17 items · 4 saves</li><li><a href="https://github.com/huggingface/smolagents/pull/883">Update code_agent.yaml to fix persistent SyntaxErrors by toadlyBroodle · Pull Request #883 · huggingface/smolagents</a>: fix the perpetual SyntaxErrors and Error in code parsing: The code blob is invalid, caused by CodeAgents adding ``` before the py code blocks</li><li><a href="https://github.com/KalyanKS-NLP/llm-engineer-toolkit">GitHub - KalyanKS-NLP/llm-engineer-toolkit: A curated list of  120+ LLM libraries category wise.</a>: A curated list of  120+ LLM libraries category wise.  - GitHub - KalyanKS-NLP/llm-engineer-toolkit: A curated list of  120+ LLM libraries category wise.</li><li><a href="https://app.foundershub.ai/user/blogs/83a8e40e-6193-42ad-9189-75c7d3af9f70">Hugging Face: The Ultimate AI Hub for Developers | Models, Datasets &amp; More</a>: Discover how Hugging Face is transforming AI development with pre-trained models, datasets, Spaces, and APIs. Learn how AI developers can experiment, fine-tune models, and deploy AI applications seaml...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1350720502628356120)** (2 messages): 

> `Open-R1 Reasoning Distillation, grpo code, distributed grpo` 


- **Reasoning ability of Open-R1**: The reasoning ability of **Open-R1** is planned to be fully distilled from other models.
   - A user noted that there is also code for **grpo** in the **openR1** repository.
- **grpo distributed across nodes**: According to what is in the blog post, distributing **grpo** across nodes is not supported yet.
   - The user also included a `:hugging_rocket:` emoji.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1350186223645229067)** (74 messages🔥🔥): 

> `Long Context Evals, 3D Generation Upgrade, DeepSeek Engineer passports, Figure's BotQ humanoid robots, Nvidia Blackwell GPUs and Together AI` 


- **Arc Prize Announces Future Date**: Arc Prize tweeted an announcement dated **3/24/2025** regarding their [future plans](https://x.com/arcprize/status/1900627173280804941).
- **DeepSeek Passport Controversy Debunked**: A DeepSeek engineer denied the rumor of passport-related policies, refuting claims in *The Information* and stating that [they are still harassed by headhunters](https://x.com/teortaxesTex/status/1900788914320793745).
   - Another researcher emphasized that *handing in passport is SOE type treatment* and not aligned with DeepSeek's culture, dismissing the claims as *disinformation*.
- **Figure Launches BotQ for Humanoid Manufacturing**: Figure announced **BotQ**, a new high-volume manufacturing facility with a first-generation line capable of producing up to **12,000** humanoid robots per year, [vertically integrating manufacturing and building software infrastructure](https://www.figure.ai/news/botq).
   - The company aims to control the build process and quality, even hinting at *Robots Building Robots*.
- **Baidu drops ERNIE 4.5 and X1**: **Baidu** unveiled **ERNIE 4.5** and **ERNIE X1**, with X1 reportedly matching DeepSeek R1's performance at half the price, also announcing that their chatbot, **ERNIE Bot**, is now free for individual users, [available on their website](https://yiyan.baidu.com/).
- **Mistral releases Small 3.1**: Mistral AI announced **Mistral Small 3.1**, a new model with improved text performance, multimodal understanding, and a **128k** token context window, outperforming models like **Gemma 3** and **GPT-4o Mini** with inference speeds of **150** tokens per second, [released under an Apache 2.0 license](https://mistral.ai/news/mistral-small-3-1).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/amir/status/1901636012729897041">Tweet from Amir Efrati (@amir)</a>: Big news in the world of Ai chips:Google poised to tap MediaTek to help handle TPU production, development. Not good for Broadcom which has exclusively done that for a decade.</li><li><a href="https://x.com/ChujieZheng/status/1900882463863283820">Tweet from Chujie Zheng (@ChujieZheng)</a>: @zephyr_z9 @TheXeophon May be a bit later. Mainly training Qwen3 now</li><li><a href="https://x.com/teortaxesTex/status/1900814672741191969">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: &gt;&gt; handing in passport. &gt; That’s not our wenfeng style.😅From a researcher whose name you&#39;ve seen on their major papers.</li><li><a href="https://x.com/arcprize/status/1900627173280804941">Tweet from ARC Prize (@arcprize)</a>: 3/24/2025</li><li><a href="https://x.com/Baidu_Inc/status/1901089355890036897">Tweet from Baidu Inc. (@Baidu_Inc)</a>: We&#39;ve just unveiled ERNIE 4.5 & X1! 🚀As a deep-thinking reasoning model with multimodal capabilities, ERNIE X1 delivers performance on par with DeepSeek R1 at only half the price. Meanwhile, ERNI...</li><li><a href="https://x.com/eric_haibin_lin/status/1901662955307200974">Tweet from Haibin@GTC (@eric_haibin_lin)</a>: @qiying_yu and team just dropped the DAPO algorithm (decoupled clip and dynamic sampling policy optimization)! DAPO-Zero-32B, a fully open-source RL reasoning model, surpasses DeepSeek-R1-Zero-Qwen-32...</li><li><a href="https://x.com/charliermarsh/status/1901634997053804610">Tweet from Charlie Marsh (@charliermarsh)</a>: I&#39;ve been working on a prototype that would let you point uv to an index and automatically get the right, pre-built versions of PyTorch, Flash Attention, vLLM, etc.No build steps, no custom instal...</li><li><a href="https://x.com/Baidu_Inc/status/1901094083508220035">Tweet from Baidu Inc. (@Baidu_Inc)</a>: ERNIE 4.5 achieves collaborative optimization through joint modeling of multiple modalities, exhibiting comprehensive improvements in understanding, generation, reasoning and memory, along with notabl...</li><li><a href="https://x.com/teortaxesTex/status/1900788914320793745">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Got report from a DeepSeek engineer that he has no knowledge of any passport-related policy or anyone in the company who&#39;s had their passport taken. Also that he&#39;s still harassed by headhunter...</li><li><a href="https://fxtwitter.com/aakashsastry/status/1901668601364689338">Tweet from Aakash (@aakashsastry)</a>: Some news - We&#39;re excited to announce that @HotshotSupport has been acquired by @xAI 🚀Over the past 2 years we&#39;ve built 3 video foundation models as a small team - Hotshot-XL, Hotshot Act One...</li><li><a href="https://x.com/MahawarYas27492/status/1900942090445746215">Tweet from AI Purr-fessor (Yash) (@MahawarYas27492)</a>: Proof that GEMINI 2.0 flash thinking in Gemini app is newer version than 0121. I think its stable version in GEMINI app but it is still called exp as its search and extension feature is in exp phase.A...</li><li><a href="https://x.com/cedric_chee/status/1901159341975384308?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from cedric (@cedric_chee)</a>: @Baidu_Inc The chonky 4.5 model is scheduled to be open source on June 30. It will also be gradually opened to developers in the future.</li><li><a href="https://x.com/teortaxesTex/status/1900791333234577519">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Yeah it&#39;s @zheanxu, he doesn&#39;t want more publicity, the GOAT so, adjust your opinion of The Information accordingly, I have</li><li><a href="https://x.com/TXhunyuan/status/1900751018889257054">Tweet from Hunyuan (@TXhunyuan)</a>: 3D generation has been upgraded again, see you next week!</li><li><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>: SOTA. Multimodal. Multilingual. Apache 2.0</li><li><a href="https://www.figure.ai/news/botq">BotQ: A High-Volume Manufacturing Facility for Humanoid Robots</a>: Introducing BotQ, Figure’s new high-volume manufacturing facility for humanoid robots. </li><li><a href="https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html">Introducing Instella: New State-of-the-art Fully Open 3B Language Models &#8212; ROCm Blogs</a>: no description found</li><li><a href="https://www.reddit.com/r/cursor/comments/1jbn4dc/upcoming_sonnet_37_max/">Reddit - Heart of the internet</a>: no description found</li><li><a href="https://x.com/ryolu_/status/1899948108865560780">Tweet from Ryo Lu (@ryolu_)</a>: What should be in @cursor_ai MAX Vibes mode? 🎙️
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1350237385925066842)** (7 messages): 

> `R1 inference costs, Deepseek free service, Hosting models locally, Fireworks alternative` 


- **Economical R1 Inference Options Explored**: The most cost-effective method to infer **R1** involves leveraging inference providers that have already optimized costs.
   - Alternative strategies include utilizing **Deepseek's free service**, or utilizing an existing GPU with surplus electricity, though full **R1** requires substantial GPU resources.
- **Local Model Hosting Strategy Involves Nvidia Helm Charts**: A member is planning to acquire GPUs for local model hosting, intending to utilize Nvidia's helm charts.
   - Another member suggested that using an inference provider is the *"cheapest way to use an inference provider who have already cost-optimised"*.
- **Fireworks alternative**: A member using **Fireworks** is looking for alternative recommendations.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1350408866445398129)** (32 messages🔥): 

> `OpenAI vs Elon Musk legal battle, Zochi AI Scientist, ICLR conference spam, AI reviewers, Liam Fedus leaving OpenAI` 


- **OpenAI and Elon Duke it out in Court**: A member shared a [link to an article](https://openai.com/index/court-rejects-elon/) about the court rejecting some of **Elon Musk's** claims against **OpenAI**, calling their actions *petty and undignified*.
- **Zochi the Artifical Scientist Debuts**: **IntologyAI** debuted **Zochi**, which they call the world’s first *Artificial Scientist*, with state-of-the-art contributions accepted in **ICLR 2025** workshops, according to [this Tweet](https://x.com/IntologyAI/status/1901697581488738322).
- **AI Slop Papers Threaten ICLR Conferences**: There's concern that conferences like **ICLR** will be spammed by *slop papers* generated by **AI**, forcing humans to read them and potentially leading to a counter-response of using **AI reviewers**.
- **Liam Fedus Bails OpenAI to Found Materials Science AI Startup**: **Liam Fedus**, **OpenAI's VP of research for post-training**, is leaving the company to found a materials science **AI startup**, with **OpenAI** planning to invest in and partner with his new company ([source](https://x.com/LiamFedus/status/1901740085416218672)).
   - One member called the post training job a *hot potato*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/erinkwoo/status/1901718788669936059">Tweet from Erin Woo (@erinkwoo)</a>: scooplet with @steph_palazzolo: Liam Fedus, OpenAI&#39;s VP of research for post-training, is leaving the company to found a materials science AI startup https://www.theinformation.com/briefings/opena...</li><li><a href="https://x.com/IntologyAI/status/1901697581488738322">Tweet from Intology (@IntologyAI)</a>: 🤖🔬Today we are debuting Zochi, the world’s first Artificial Scientist with state-of-the-art contributions accepted in ICLR 2025 workshops.Unlike existing systems, Zochi autonomously tackles some of ...</li><li><a href="https://x.com/LiamFedus/status/1901740085416218672">Tweet from William Fedus (@LiamFedus)</a>: This is what I sent to my colleagues at OpenAI:Hi all, I made the difficult decision to leave OpenAI as an employee, but I’m looking to work closely together as a partner going forward. Contributing t...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1350187010907705406)** (101 messages🔥🔥): 

> `Claude Code Vim mode, Gemma 3 License, Deepseek integrated in Chinese food delivery, LLMs as copy editors, Free Speech Eval` 


- **Claude Code gets Vim Mode**: Claude Code now has **Vim mode**, giving users familiar insert/command modes for editing prompts by typing the slash command `/vim` ([source](https://x.com/_catwu/status/1900593728664035590)).
- **Gemma 3 License Restricts Commercial Use**: Google released **Gemma 3**, praised for its efficiency, but its license makes commercial use risky, similar to Meta's custom, non-standard licensing terms ([source](https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/)).
- **Deepseek powers food delivery in China**: Chinese food delivery apps have integrated **Deepseek** to provide summaries of the food, displayed with the name DeepSeek prominently, enhancing credibility ([source](https://x.com/yifever/status/1900803902049857694)).
   - The mention of **DeepSeek** instead of just *AI* adds credibility, positioning it as a *national symbol*.
- **LLMs Vibe Check as Copy Editors**: One member shared vibe checks on LLMs as copy editors, finding **Sonnet-3.7** horrible, **Opus** great but compresses long inputs, and **GPT-4.5** the new main for quality ([source](https://x.com/eugeneyalt/status/1900953586550665569)).
- **Claude Sonnet-3.7 Dominates Free Speech Eval**: Claude-3.7-Sonnet significantly improved in free speech evaluations, becoming one of the most compliant models, though it still avoids satirizing national anthems ([source](https://x.com/xlr8harder/status/1901208947991662888)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1901697398134452527">Tweet from Xeophon (@TheXeophon)</a>: Releasing the only important benchmark: MascotBench</li><li><a href="https://x.com/yifever/status/1900803902049857694">Tweet from never yifei yourself e/λ (@yifever)</a>: chinese food delivery apps have already integrated deepseek</li><li><a href="https://x.com/eugeneyalt/status/1900953586550665569">Tweet from eugene (@eugeneyalt)</a>: vibe checks on llms as copy editor• sonnet-3.7: horrible; sprinkles so many adjectives making it linkedin slop• opus: used to be my main. great except it compresses too much if input is long, limiting...</li><li><a href="https://x.com/qtnx_/status/1901687937055781105">Tweet from Q (@qtnx_)</a>: le chat is out of the bag, been interning at mistral for the past month, deeply appreciate the opportunity i was given</li><li><a href="https://x.com/xlr8harder/status/1901208947991662888">Tweet from xlr8harder (@xlr8harder)</a>: Free Speech Eval: Chinese EditionI&#39;ve extended my free speech eval to ask for criticism of China in Chinese. The results are interesting: even fairly compliant models are less willing to criticize...</li><li><a href="https://fxtwitter.com/Teknium1/status/1901673193389305868">Tweet from Teknium (e/λ) (@Teknium1)</a>: Super excited to be able to bring in and work with @dmayhem93 on building RL infra and take on post training at Nous!We are cooking amazing things including a powerful RL Gym and a super optimized tra...</li><li><a href="https://x.com/testingcatalog/status/1901051435497771158">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: What Harmony can do:- Scan the local directory and link to certain files in the response- Open files in the side sheet - Edit files and show diffs for users to approve- Search for uses of a certain ke...</li><li><a href="https://x.com/kipperrii/status/1901665263822709154">Tweet from kipply (@kipperrii)</a>: torn between &#34;what have i done&#34; and &#34;he&#39;s so cute&#34;he&#39;s super cuddly though, he&#39;s weighted and you can turn on a module that gives him a lil heartbeat</li><li><a href="https://x.com/testingcatalog/status/1901679701506003391">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: xAI is working on &#34;DeeperSearch&#34; and Memory for Grok. DeeperSearch potentially will apply different &#34;presets&#34; to DeepSearch. Currently, it is passing the &#34;default&#34;...</li><li><a href="https://x.com/mgostIH/status/1901215264986800332">Tweet from mgostIH (@mgostIH)</a>: Dang</li><li><a href="https://fxtwitter.com/EsotericCofe/status/1777280241884377474">Tweet from Nucleus☕️ (@EsotericCofe)</a>: hardest LLM paper vs easiest diffusion paper</li><li><a href="https://x.com/_catwu/status/1900593728664035590">Tweet from cat (@_catwu)</a>: Another batch of features for Claude Code!Up first: Vim mode. This gives you the familiar insert/command modes for editing your prompts in Claude Code. Turn it on by typing the slash command /vim.But ...</li><li><a href="https://x.com/teortaxesTex/status/1901691453346127945">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: the nonprofit ASP (American Sunlight Project) on why Russia produces so much propaganda that nobody reads: it&#39;s okay! Crawlers do, and then people read what LLMs regurgitate. They call this strate...</li><li><a href="https://fxtwitter.com/testingcatalog/status/1901051432339730603">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Early preview of the upcoming Harmony feature for Claude. Harmony will allow users to give Claude FULL access to a local directory so it can research and operate with its content. Is Harm...</li><li><a href="https://fxtwitter.com/victorsungo/status/1901510951314305451">Tweet from Qingfeng Sun (@victorsungo)</a>: ✍️Career update:After an incredible 6-year journey at Microsoft, I&#39;ve recently transitioned to the @TXhunyuan team two months ago. I will focus mainly on post-training & RL in the future.I&#39;m d...</li><li><a href="https://www.newsguardrealitycheck.com/p/a-well-funded-moscow-based-global">A well-funded Moscow-based global ‘news’ network has infected Western artificial intelligence tools worldwide with Russian propaganda</a>: An audit found that the 10 leading generative AI tools advanced Moscow’s disinformation goals by repeating false claims from the pro-Kremlin Pravda network 33 percent of the time</li><li><a href="https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/">&#039;Open&#039; AI model licenses often carry concerning restrictions | TechCrunch</a>: &#039;Open&#039; model releases from Google, Meta, and others have onerous terms that make some companies wary of using them.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1350501146170621962)** (3 messages): 

> `Azure AI Agents API vs OpenAi Assistants API, Mistral Meow` 


- **Azure's API Deception is No Accident**: A user pointed out that the [new Azure AI Agents API](https://learn.microsoft.com/en-us/azure/ai-services/ai-agents/concepts/agents-overview) is actually the [deprecated OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview).
   - The user wryly commented, *"Brilliant play"*.
- **Mistral Releases new Chatbot, Meow!**: Mistral has released a new chatbot called **Meow** at [meow.mistral.ai](https://meow.mistral.ai/).
- **X user seeks help, tags AI Leaders**: An X user seeks help and tags **Logan Kilpatrick, V Gabeur, Mehrdad Dehghani, and Robert Riachi** in [this tweet](https://x.com/Angaisb_/status/1900929427132817903).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/headinthebox/status/1901690336298311878">Tweet from Erik Meijer (@headinthebox)</a>: The new Azure AI Agents API [0] is the old depricated OpenAi Assistants API [1]. Brilliant play.</li><li><a href="https://x.com/Angaisb_/status/1900929427132817903">Tweet from angel⭐ (@Angaisb_)</a>: Please someone help me lmao @OfficialLoganK @vgabeur @m__dehghani @robertriachi
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1350190512946151576)** (39 messages🔥): 

> `GRPO implementation trick, Applying KL penalty in the loss, DAPO algorithm, Zero-shot RL` 


- **GRPO Uses Loss Penalty Trick**: A member discussed a **GRPO implementation trick** that applies a penalty in the loss, as opposed to traditional RLHF which applies it to the reward, noting its impact is hard to determine but may help the model focus on reward signals, as described in the [RLHF book](https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1).
   - It was also pointed out that the math may be wrong.
- **KL Penalty Placement Gets Questioned**: A member inquired about the effect of applying the **KL penalty directly in the loss** versus when the reward is computed, asking for intuitions or ablations on the subject via [Twitter](https://x.com/natolambert/status/1900639281791615387).
   - The discussion touched upon whether normalization by token helps with learning dynamics, and if a per-token formulation would be "better."
- **Decoupled Algorithm Dominates Deep Reasoning**: A new **DAPO (decoupled clip and dynamic sampling policy optimization) algorithm** and model called **DAPO-Zero-32B** were introduced, outperforming **DeepSeek-R1-Zero-Qwen-32B** on reasoning tasks, achieving a score of 50 on AIME 2024 with fewer steps, and trained with **zero-shot RL** from the Qwen-32b pre-trained model, all open-sourced at [dapo-sia.github.io](https://dapo-sia.github.io/).
   - It was noted that if a reasoning pattern contributes to the reward, its contribution will be much lower if it's part of a long chain of thought under row mean.
- **DAPO Dataset Gets Massive Upscaling**: It was discovered that the authors of **DAPO** accidentally duplicated the dataset by roughly **100x**, which resulted in a dataset of 310 MB, and a member created a deduplicated version via HF's SQL console, reducing the dataset to 3.17 MB ([HuggingFace Dataset](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup)).
   - The authors acknowledged the issue, stating that they were aware but *can't afford retraining*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/youjiacheng/status/1901699950523908344?s=61">Tweet from You Jiacheng (@YouJiacheng)</a>: I found the authors (cc @tongyx361 ) accidentally duplicated the dataset by ~100x (17398 prompt → 17917 index → 1791700 row).So I created a simple deduplication of it via HF&#39;s SQL console -- it&#3...</li><li><a href="https://x.com/tongyx361/status/1901702083352678763?s=61">Tweet from Shawn/Yuxuan TONG (@tongyx361)</a>: The duplication was accidentally by one of our collaborators and we are aware of this issue but can&#39;t afford retraining 😂Duplicating 100x and training for 1 epoch is diffrent from no duplicating ...</li><li><a href="https://x.com/natolambert/status/1900639281791615387">Tweet from Nathan Lambert (@natolambert)</a>: Does anyone have an intuition or ablation on applying the KL penalty in the loss directly rather than when the reward is computed? How is this changing learning.normalrewards = rewards - self.beta * p...</li><li><a href="https://x.com/eric_haibin_lin/status/1901662955307200974">Tweet from Haibin@GTC (@eric_haibin_lin)</a>: @qiying_yu and team just dropped the DAPO algorithm (decoupled clip and dynamic sampling policy optimization)! DAPO-Zero-32B, a fully open-source RL reasoning model, surpasses DeepSeek-R1-Zero-Qwen-32...</li><li><a href="https://x.com/danielhanchen/status/1901042482475135162">Tweet from Daniel Han (@danielhanchen)</a>: @natolambert I think for extremely imbalanced losses and extremely imbalanced completion lengthsmean(row_sum(loss * mask)/row_sum(mask)) -&gt; 906gets a higher loss vssum(loss * mask)/sum(mask) -&gt; ...</li><li><a href="https://x.com/rm_rafailov/status/1900943284249543078">Tweet from Rafael Rafailov @ NeurIPS (@rm_rafailov)</a>: @natolambert Like many things these days, the GRPO math is wrong.</li><li><a href="https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1">Policy Gradient
Algorithms | RLHF Book by Nathan Lambert</a>: The Reinforcement Learning from Human Feedback Book</li><li><a href="https://rlhfbook.com/c/11-policy-gradients.html">Policy Gradient
Algorithms | RLHF Book by Nathan Lambert</a>: The Reinforcement Learning from Human Feedback Book</li><li><a href="https://bsky.app/profile/natolambert.bsky.social/post/3lkeftspdzo2x">Nathan Lambert (@natolambert.bsky.social)</a>: Does anyone have an intuition or ablation on applying the KL penalty in the loss directly rather than when the reward is computed? How is this changing learning.normalrewards = rewards - self.beta * p...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1351252288043221022)** (4 messages): 

> `Noam Chomsky, Nicholas Carlini, Future of LLMs, AI risks` 


- **Noam Chomsky Makes Rare Appearance**: A member shared a [YouTube video](https://www.youtube.com/watch?v=atMRWzgHEGg) featuring a rare appearance of **Noam Chomsky**.
   - The member humorously added that every prominent AI figure needs a signature hat.
- **Carlini Forecasts Wide Error Bars for LLMs**: A member shared a link to [Nicholas Carlini's blog post](https://nicholas.carlini.com/writing/2025/thoughts-on-future-ai.html) on the potential future of **LLMs**.
   - Carlini writes that they *"wouldn't be surprised if, in three to five years, language models are capable of performing most (all?) cognitive economically-useful tasks beyond the level of human experts"* but that there's very wide error bars on that possibility.



**Link mentioned**: <a href="https://nicholas.carlini.com/writing/2025/thoughts-on-future-ai.html">
      My Thoughts on the Future of "AI"
    </a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1350496325778800743)** (36 messages🔥): 

> `RLHF Book, Claude Code vs ChatGPT, Chorus writing checker, ChatGPT Deep Research for teaching websites` 


- ****RLHF Book** gets typo assist!**: Members are using **Deep Research** to find all the typos in the [RLHF book](https://chatgpt.com/share/67d5a1e5-a160-8005-a7e1-a9d1141d4552).
   - They're trying out **Claude Code** for the same task, saying it seems to be working too, but **Gemini Deep Research** sucked.
- ****Chorus** is nicer than Grammarly**: A member has been using **Chorus** to check their writing with all LLMs and the different things they find always surprise them.
   - *It's also just much nicer to just have AI do it and supervise*, because Grammarly sucks.
- ****ChatGPT Deep Research** feedback is generic**: A member found ChatGPT Deep Research's feedback on their teaching website to be *generally positive but pretty generic and cliche*.
   - It suggested categories of problems instead of identifying specific high-value issues, and also claimed there were broken links when there weren't.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.chorus.sh/chats/read-only/84f5781a-4fa7-4686-852a-f49830965384">Chorus - Website Typo Check</a>: no description found</li><li><a href="https://g.co/gemini/share/e84e1b0574ac">‎Gemini - Subdomain Typos and Mistakes Check
</a>: Created with Gemini
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1350184014790791172)** (224 messages🔥🔥): 

> `Swarm vs Mesh vs Sequence for multi-agent systems, OpenSwarm and OpenAI-agents, mycoder.ai vs claude-code, Monetizing MCP services, Glama scans` 


- **Multi-Agent Systems Topologies Debate Swarm vs Mesh vs Sequence**: A member initiated a discussion on **Swarm**, **Mesh**, and **Sequence** architectures for multi-agent systems, seeking resources and advice, while struggling with sub-agents going off-track due to the *telephone game* effect.
   - One member suggested the problems might relate to *parallel execution and unsupervised autonomy issues*, where the **handoff** of execution between agents includes swapping system instructions, available functions, and even the model or provider being used.
- **OpenSwarm's Evolution into OpenAI-Agents**: A member mentioned working on **OpenSwarm** for a client and its subsequent adoption by OpenAI, rebranded as **openai-agents**, with additional OpenAI-specific features, while noting a rejected PR for MCP support.
   - They also mentioned rumors that **CrewAI** (or **PraisonAI**?) might offer similar functionality using a *stateless single thread agent approach*.
- **mycoder.ai Launched just before claude-code**: A member noted the coincidental launch of their **mycoder.ai** just before **Claude-code** was announced, adapting by posting it to Hacker News and reaching the front page, check it out [here](https://news.ycombinator.com/item?id=43177117).
   - It was noted that **claude-code** is Anthropic-only, creating demand for a more generic solution, with success using **litellm proxy**.
- **Discussions Spark on Monetizing MCP Services**: Members debated the possibilities of monetizing their MCP services, touching on the challenges with API resale restrictions and the potential for BYOK (**Bring Your Own Key**) models.
   - Some suggested focusing on unique services or scraping agents, while others expressed caution due to API terms, with one member only interested in *coffee money kind of donation*.
- **Glama's Inspection Process of MCP Servers Debated**: A member questioned the frequency of **Glama scans** and the ability to trigger rescans for MCP servers, and the discussion revealed that scans are tied to the frequency of commits to the associated GitHub repository.
   - Difficulties were reported with servers failing to be inspected, showing a *Could not inspect the server* message on the Score tab, even after fixing dependency issues and successfully running in the inspector, with work on triggering refreshes underway, for more info see [Glama AI](https://glama.ai/mcp/servers/s2em7b2kwf/score).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://leehanchung.github.io/blogs/2025/03/07/claude-code/">Poking Around Claude Code</a>: Discover how Claude Code leverages LLMs as agents for software engineering tasks, including system prompts, model control protocols (MCP), control flow, and hidden features.</li><li><a href="https://huggingface.co/sesame/csm-1b">sesame/csm-1b · Hugging Face</a>: no description found</li><li><a href="https://mcpsx.run/">mcpsx CLI | Model Context Protocol Tools</a>: A powerful CLI tool for managing Model Context Protocol (MCP), creating logical groups of tools, and optimizing token usage when interacting with AI models.</li><li><a href="https://glama.ai/mcp/servers/s2em7b2kwf/score">Score | MCP Selenium</a>: Enables browser automation using the Selenium WebDriver through MCP, supporting browser management, element location, and both basic and advanced user interactions.</li><li><a href="https://glama.ai/mcp/servers/ss8n1knen8">replicate-flux-mcp</a>: MCP for Replicate Flux Model. Generating images by prompts</li><li><a href="https://github.com/ahujasid/blender-mcp">GitHub - ahujasid/blender-mcp</a>: Contribute to ahujasid/blender-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/punkpeye/fastmcp">GitHub - punkpeye/fastmcp: A TypeScript framework for building MCP servers.</a>: A TypeScript framework for building MCP servers. Contribute to punkpeye/fastmcp development by creating an account on GitHub.</li><li><a href="https://glama.ai/mcp/servers/cyeeqagb81">Supabase MCP Server</a>: This server enables interaction with Supabase PostgreSQL databases through the MCP protocol, allowing seamless integration with Cursor and Windsurf IDEs for secure and validated database management.</li><li><a href="https://github.com/angiejones/mcp-selenium">GitHub - angiejones/mcp-selenium: An MCP implementation for Selenium WebDriver</a>: An MCP implementation for Selenium WebDriver. Contribute to angiejones/mcp-selenium development by creating an account on GitHub.</li><li><a href="https://github.com/angiejones/mcp-selenium/blob/main/package.json#L13">mcp-selenium/package.json at main · angiejones/mcp-selenium</a>: An MCP implementation for Selenium WebDriver. Contribute to angiejones/mcp-selenium development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/python-sdk">GitHub - modelcontextprotocol/python-sdk: The official Python SDK for Model Context Protocol servers and clients</a>: The official Python SDK for Model Context Protocol servers and clients - modelcontextprotocol/python-sdk</li><li><a href="https://github.com/robertheadley/chrome-debug-mcp">GitHub - robertheadley/chrome-debug-mcp: An MCP server to allow you to debug webpages using LLMs</a>: An MCP server to allow you to debug webpages using LLMs - robertheadley/chrome-debug-mcp</li><li><a href="https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#writing-mcp-clients>">GitHub - modelcontextprotocol/python-sdk: The official Python SDK for Model Context Protocol servers and clients</a>: The official Python SDK for Model Context Protocol servers and clients - modelcontextprotocol/python-sdk</li><li><a href="https://www.braze.com/">Braze Customer Engagement Platform</a>: Power customer-centric interactions between consumers and brands in real-time.</li><li><a href="https://docs.customer.io/">Customer.io Docs</a>: Trigger email, push, in-app, SMS, webhooks, and more with Customer.io. Gain control over behaviorial data to personalize customer communication and drive engagement.</li><li><a href="https://news.ycombinator.com/item?id=43177117">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1350200277810286642)** (25 messages🔥): 

> `Awesome Vibe Coding, Roo Code MCP, MacOS Control MCP, Secretary MCP, Professional Graph MCP` 


- ****Awesome Vibe Coding** List Launched**: A curated list of tools, editors, and resources that make AI-assisted coding more intuitive and efficient called [Awesome Vibe Coding](https://github.com/filipecalegario/awesome-vibe-coding) was created.
   - The list includes AI-powered IDEs, browser-based tools, plugins, and command line tools to enhance workflows, and a member even had their AI coder make a PR back to the repo, as well as suggest the addition of [Roo Code](https://github.com/szcharlesji/crypto-mcp).
- **MCPs Galore: Creating Custom Servers**: A user created an app that allows users to create their own MCP server with custom or community prompts called [Groove Studio](https://grooving.xyz/).
   - They are looking for user feedback, with some users suggesting features like an MCP that gives a model the ability to control MacOS natively and non-natively, or a "Secretary" MCP that uses a memory bank of texts, emails, calendar and notes.
- ****Emojikey MCP Server** Updated**: A member announced an update to the **Emojikey MCP server** with 'lots of goodness,' calling it essential for vibe coding and linking to the [GitHub repository](https://github.com/identimoji/mcp-server-emojikey).
   - It allows users to *save your unique relationship state and interaction style with your favorite LLM*.
- **Game Asset MCP Server Seeks Testers**: A member is looking for testers for a [Game Asset MCP](https://github.com/MubarakHAlketbi/game-asset-mcp) server.
   - This MCP server is for creating **2D/3D game assets from text** using Hugging Face AI models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://grooving.xyz/">Groove Studio</a>: no description found</li><li><a href="https://github.com/szcharlesji/crypto-mcp">GitHub - szcharlesji/crypto-mcp: Real-time access to cryptocurrency data from the CoinMarketCap API.</a>: Real-time access to cryptocurrency data from the CoinMarketCap API. - szcharlesji/crypto-mcp</li><li><a href="https://github.com/identimoji/mcp-server-emojikey">GitHub - identimoji/mcp-server-emojikey: MCP Server for emojikey.io ... save your unique relationship state and interaction style with your favorite LLM</a>: MCP Server for emojikey.io ... save your unique relationship state and interaction style with your favorite LLM - identimoji/mcp-server-emojikey</li><li><a href="https://github.com/MubarakHAlketbi/game-asset-mcp">GitHub - MubarakHAlketbi/game-asset-mcp: An MCP server for creating 2D/3D game assets from text using Hugging Face AI models.</a>: An MCP server for creating 2D/3D game assets from text using Hugging Face AI models. - MubarakHAlketbi/game-asset-mcp</li><li><a href="https://github.com/filipecalegario/awesome-vibe-coding">GitHub - filipecalegario/awesome-vibe-coding: A curated list of vibe coding references, collaborating with AI to write code.</a>: A curated list of vibe coding references, collaborating with AI to write code. - filipecalegario/awesome-vibe-coding
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1350201234753065081)** (34 messages🔥): 

> `Agentic systems multi-threading, Claude's Birthday, GPT-o1 Acing Math Exams, SAE Bench Release, Baidu ERNIE 4.5 & X1` 


- **Agentic systems multi-threading discussed**: Discussion on how to design an **agentic system** for **multi-threaded**, parallel execution of long-running tasks, with the consensus that there wouldn't be significant design differences from other parallel applications.
   - The primary focus should be on managing **API consumption** effectively in a multi-threaded environment.
- **Claude celebrates 2nd birthday**: [Claude celebrated its second birthday](https://x.com/alexalbert__/status/1900592059364634973?s=46), highlighting its use for **company OSINT** due to its reduced refusal rate compared to **ChatGPT** for deep research.
   - A user considered it is *great for company OSINT compared to chatGPT deep research because it refuses to answer questions much less.*
- **GPT-o1 Aces Carnegie Mellon Math Exam**: **GPT-o1** achieved a perfect score on a **Carnegie Mellon** undergraduate math exam, solving each problem in under a minute for about 5 cents each according to [this post](https://x.com/poshenloh/status/1900721180887203879?s=46).
   - The exam, designed with non-standard problems, was also open-book and open-notes, impressing the instructor who noted that this was *close to the tipping point of being able to do moderately-non-routine technical jobs.*
- **SAE Bench Released for Sparse Autoencoder Evaluation**: The full release of **SAE Bench**, a suite of Sparse Autoencoder (SAE) evaluations designed to improve **SAE research** by providing better metrics, was announced in [this Tweet](https://x.com/neelnanda5/status/1900872633664544769?s=46).
   - The suite includes proxy summary statistics, downstream task performance metrics, and evaluations of known flaws, alongside a set of open-source SAEs across 7 architectures.
- **Baidu Launches ERNIE 4.5 and X1, Making ERNIE Bot Free**: **Baidu** unveiled **ERNIE 4.5** and **ERNIE X1**, with ERNIE X1 reportedly matching DeepSeek R1's performance at half the cost according to [this announcement](https://x.com/baidu_inc/status/1901089355890036897?s=46).
   - In addition, **ERNIE Bot** has been made freely accessible to individual users ahead of schedule, with both models available on the official website.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Teknium1/status/1901674925259370663">Tweet from Teknium (e/λ) (@Teknium1)</a>: .@MistralAI just released a new version of their 24B model - this time is multimodal and has 128K context - exactly what we wanted! This enables the reasoning models to be fully exploited on both long...</li><li><a href="https://x.com/andimarafioti/status/1901649025750667277">Tweet from Andi Marafioti (@andimarafioti)</a>: 🚀We just dropped SmolDocling: a 256M open-source vision LM for complete document OCR!📄✨It&#39;s lightning fast, process a page in 0.35 sec on consumer GPU using &lt; 500MB VRAM⚡SOTA in document conv...</li><li><a href="https://x.com/neelnanda5/status/1900872633664544769?s=46">Tweet from Neel Nanda (@NeelNanda5)</a>: I&#39;m excited to announce the full release of SAE bench! I think SAE research has been substantially held back by lack of good metrics, and this is a significant step forward, with proxy summary sta...</li><li><a href="https://x.com/poshenloh/status/1900721180887203879?s=46">Tweet from Po-Shen Loh (@PoShenLoh)</a>: Oh my goodness. GPT-o1 got a perfect score on my @CarnegieMellon undergraduate #math exam, taking less than a minute to solve each problem. I freshly design non-standard problems for all of my exams, ...</li><li><a href="https://x.com/alexalbert__/status/1900592059364634973?s=46">Tweet from Alex Albert (@alexalbert__)</a>: Two years ago today we announced Claude to the world.Happy second birthday, Claude!</li><li><a href="https://x.com/levelsio/status/1901660771505021314">Tweet from @levelsio (@levelsio)</a>: I&#39;m organizing the🌟 2025 Vibe Coding Game JamDeadline to enter: 25 March 2025, so you have 7 days- anyone can enter with their game- at least 80% code has to be written by AI - game has to be acc...</li><li><a href="https://x.com/natolambert/status/1901758392043221072">Tweet from Nathan Lambert (@natolambert)</a>: This is a very tidy little RL paper for reasoning. Their GRPO changes:1 Two different clip hyperparams, so positive clipping can uplift more unexpected tokens2 Dynamic sampling -- remove samples w fla...</li><li><a href="https://mistral.ai/fr/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>: SOTA. Multimodal. Multilingual. Apache 2.0</li><li><a href="https://x.com/baidu_inc/status/1901089355890036897?s=46">Tweet from Baidu Inc. (@Baidu_Inc)</a>: We&#39;ve just unveiled ERNIE 4.5 & X1! 🚀As a deep-thinking reasoning model with multimodal capabilities, ERNIE X1 delivers performance on par with DeepSeek R1 at only half the price. Meanwhile, ERNI...</li><li><a href="https://rlhfbook.com/c/11-policy-gradients.html">Policy Gradient
Algorithms | RLHF Book by Nathan Lambert</a>: The Reinforcement Learning from Human Feedback Book
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1350226688595853403)** (5 messages): 

> `Snipd Podcast, AI Podcast App, Outdoor Podcast, Tech Stack, Switching from Finance to Tech` 


- ****Snipd** Podcast Gets Fresh Air**: A new **Snipd** podcast featuring [Kevin Smith](https://x.com/latentspacepod/status/1900666708270215383) was released, discussing the **AI Podcast App for Learning**.
   - This episode marks their first *outdoor* podcast, with @swyx and @KevinBenSmith chatting about **aidotengineer NYC**, switching from Finance to Tech, and the tech stack of [@snipd_app](https://www.snipd.net/).
- **Fan Loves **Snipd**, Shares Photo**: A user expressed their love for **Snipd** and shared a [photo](https://cdn.discordapp.com/attachments/1350226688595853403/1350233202853154877/IMG_7313.png?ex=67d9f2a9&is=67d8a129&hm=23209f00c920926a0c7e949ee91bbcd646c736764a7b7975bcbc8ddae42dab2b&) as proof.



**Link mentioned**: <a href="https://x.com/latentspacepod/status/1900666708270215383">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 Snipd: The AI Podcast App for Learninghttps://youtu.be/FNRO_SYx68QOur first ever OUTDOOR podcast! @swyx and @KevinBenSmith chat about @aidotengineer NYC, switching from Finance to Tech, how AI can ...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1350196967078367254)** (122 messages🔥🔥): 

> `Claude 3.5 vs 3.7, Vibe Coding, Levelsio Flight Simulator, Auto Git Commits, Enterprise AI Dev Team Enablement` 


- **Claude Showdown: 3.5 vs 3.7**: Members debated the merits of using **Claude 3.5** over **3.7**, citing that **3.7** is *way too eager* and does things without being asked.
   - Others said they used **Claude 3.5** and were experiencing **GPU** issues.
- **Vibe Coding: A New Development Meta**: The concept of "vibe coding," particularly using tools like **Cursor**, was discussed, with one member referencing a [tweet by Levelsio](https://x.com/levelsio/status/1893350391158292550) where he built a flight simulator in the browser using **Cursor**.
   - A member shared a [followup tweet](https://x.com/levelsio/status/1899596115210891751) about the same project reaching **$1 million ARR** in just 17 days by selling in-game ads.
- **Auto Git Commits**: Members discussed automatically creating **git commits** with every line accepted by the LLM, mentioning tools like **aider** and linking to [gitdoccommits](https://github.com/lostintangent/gitdoccommits).
   - One member proposed that traditional IDEs may not be the right UI for vibe coding and suggested visualizing the tree of changes prompted by different chats.
- **Enterprise AI Dev Team**: A member offered to share insights on **enterprise AI dev team enablement** at some point, mentioning its corporate nature.
   - Another member expressed interest in hearing about the *hurdles* and red tape involved in getting **Cursor** into an organization.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/levelsio/status/1893350391158292550">Tweet from @levelsio (@levelsio)</a>: ✨ Today I thought what if I ask Cursor to build a flight simulatorSo I asked  &#34;make a 3d flying game in browser with skyscrapers&#34; And after many questions and comments from me I now have the o...</li><li><a href="https://x.com/levelsio/status/1899596115210891751">Tweet from @levelsio (@levelsio)</a>: ✨ http://fly.pieter.com has now gone from $0 to $1 million ARR in just 17 days!💸 Revenue update: $87,000 MRR (which is $1M ARR)My first project ever to go up this fast 🤯Only 3 ads left now: https://...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1350195083760369865)** (27 messages🔥): 

> `Gemini-integrated Android, Deepseek R1 Impact, Audio Overview Length, NotebookLM Use Cases, Hyperbolic Tapering Schedule` 


- **Users clamor for Gemini-Integrated Android**: Multiple users expressed strong interest in a fully **Gemini-integrated Android experience**, envisioning a powerful combination of **Google Assistant/Gemini** with **NotebookLM**.
   - Some users expressed frustration with the current **Gemini** implementation on Android, hoping for rapid improvements.
- **Deepseek R1 Shakes the AI Market**: A user commented on the dramatic escalation in the AI market following the release of **Deepseek R1**, which offered reasoning capabilities at a low cost, impacting **Gemini 2.0** and other models.
   - The user noted that the release of **Deepseek R1** seemingly *shook the whole industry* and spurred the release of several new models from other companies.
- **Users Seek to Extend Audio Overview Length**: A user inquired about the possibility of increasing the length of audio overviews generated by NotebookLM, noting that **16,000-word files** only produced **15-minute overviews**.
   - They desired at least **1-hour+ overviews**, but no concrete solutions were provided by the community.
- **NotebookLM helps user Taper Off Psychiatric Meds!**: One user is using NotebookLM to construct a *hyperbolic tapering schedule* for a psychiatric medication, finding correlational studies to make a taper schedule.
   - Another user cautioned that **tapering based on data** on *any* platform should not be done alone without expert professional opinion.
- **Users want NotebookLM integration into internal portals/CRMs**: A user inquired about integrating NotebookLM into an internal portal/CRM with videos and knowledge base articles, for consultants to ask questions and get answers from the portal.
   - A user suggested [Agentspace](https://cloud.google.com/products/agentspace?hl=en) could be exactly what they're looking for, as it's integrated with NotebookLM.



**Link mentioned**: <a href="https://cloud.google.com/products/agentspace?hl=en">Google Agentspace</a>: Google Agentspace is the launch point for enterprise-ready AI agents, helping increase employee productivity for complex tasks with one single prompt.

  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1350185407924539432)** (132 messages🔥🔥): 

> `Extracting Google Sheets for LM, Gemini for data analysis, Public sharing of NotebookLM, Using NotebookLM to prevent errors, NotebookLM limitations and solutions` 


- **Users Brainstorm Google Sheets Extraction for NotebookLM**: Users discussed methods for extracting **Google Sheets** into a readable format for **NotebookLM**, with one suggesting using **BigQuery** and **SQL** with **Gemini** to generate queries for data analysis.
   - Another user mentioned a Sheets function that reads a cell and passes it as context to a prompt to generate answers, useful for **RFP** situations.
- **Public Notebook Sharing Potentially on the Horizon**: A user inquired about enabling public sharing of **NotebookLM** notebooks, envisioning it as a new form of publishing.
   - A Google employee responded that they are *"super interested in the idea that the notebook is a powerful new way of collecting and sharing information"* and actively working on the feature.
- **Experimenting to Prevent NotebookLM Errors**: A user shared their approach to prevent **NotebookLM** from repeating errors by creating an *'Errors'* source document with examples of past mistakes.
   - Another user suggested that such instructions might not have any impact on the responses because *NLM uses RAG, it's not injecting the complete user inputs (sources) into the context window of the LLM*.
- **NotebookLM Limitations and Potential Solutions**: A user reported that **Audio Overviews** can't be fast-forwarded during interactive beta, also asking for increased **audio overview generation length**.
   - Another user suggested **Agentspace** as a potential solution for integrating **NotebookLM** with various data sources and internal portals.
- **Agentspace to rescue NotebookLM Enterprise**: A user inquired about integrating NotebookLM with an internal portal and CRM with videos and knowledge base articles, electioneering suggested that, *NotebookLM doesn't have an API you could use and it doesn't support connecting to the types of data sources you mention*.
   - electioneering suggested looking at **Agentspace** as a solution, which *includes and is integrated with NotebookLM* [Agentspace](https://cloud.google.com/products/agentspace?hl=en).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://surf.e2b.dev/">Surf - E2B Computer Use Agent</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15731776">Audio Overviews - NotebookLM Help</a>: no description found</li><li><a href="https://notebooklm.google.com/notebook/f7607d7a-584c-4f35-96fc-f6815c573a6c?_gl=1*52xa3q*_ga*MjEzMjQ2ODA5Ni4xNzI5NTUyMzk5*_ga_W0LDH41ZCB*MTcyOTYxNzAwNC41LjEuMTcyOTYxOTMxMy4wLjAuMA..)">no title found</a>: no description found</li><li><a href="https://support.google.com/notebooklm/)">NotebookLM Help</a>: no description found</li><li><a href="https://github.com/GoogleCloudPlatform/agent-starter-pack">GitHub - GoogleCloudPlatform/agent-starter-pack: A collection of production-ready Generative AI Agent templates built for Google Cloud. It accelerates development by providing a holistic, production-ready solution, addressing common challenges (Deployment &amp; Operations, Evaluation, Customization, Observability) in building and deploying GenAI agents.</a>: A collection of production-ready Generative AI Agent templates built for Google Cloud. It accelerates development by providing a holistic, production-ready solution, addressing common challenges (D...</li><li><a href="https://cloud.google.com/products/agentspace?hl=en">Google Agentspace</a>: Google Agentspace is the launch point for enterprise-ready AI agents, helping increase employee productivity for complex tasks with one single prompt.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1350544737571963011)** (6 messages): 

> `Jake Cannell hiring GPU devs, sm90 kernels, GPU performance counters, nebius.ai, Datacrunch` 


- **Jake Cannell Staffing Up with GPU Devs**: Jake Cannell is [hiring GPU developers](https://www.linkedin.com/jobs/view/4118975911/) to work on ideas he touched on in his talk.
- **Academics Seek Budget-Friendly GPU Cloud**: A researcher is looking for cheap cloud providers that give access to **GPU performance counters** to run Nsight Compute for implementing ideas for **sm90 kernels**.
- **nebius.ai touted for GPU cloud**: A member recommends [nebius.ai](https://nebius.ai), citing a Reddit thread from 9 months ago, as a provider with access to GPU performance counters.
- **Datacrunch proposes student credits**: A member suggested **Datacrunch** as a good option, offering potential credits for students.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1350494353897820311)** (13 messages🔥): 

> `Embedded Python Pip Usage, Triton Windows PyPI Release, tl.multiple_of usage in Triton, Efficient Pointer Chasing in Triton, Triton and Sparse Computations` 


- **Triton-Windows Gets PIP Upgrade**: **Triton-windows** has been published to **PyPI**, so you can install/upgrade it by `pip install -U triton-windows`, and you no longer need to download the wheel from GitHub.
- **`tl.multiple_of` Questioned in Triton**: A user questioned the usage of `tl.multiple_of` with `tl.arange`, suspecting that only the first element is a multiple of **BLOCK_SIZE_N**, and wondered if they missed something.
- **Pointer Chasing Performance Pondered**: A user asked about implementing efficient **pointer-chasing** in Triton for a custom data structure resembling a sparse matrix in CSR, seeking to avoid loading offsets one-by-one in a hot inner loop.
   - One user suggested loading the whole offset array at once, and then using `tl.sum()` over `tl.where()` with the loop index to mask out all but one element, another user mentioned that *Triton is not ideal for sparse references and computations, the authors mention this in [Triton Lang Docs](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html)*.
- **Powering Through with pow in Triton**: A user inquired about how to use the **pow** (power) function in Triton.
   - Another user pointed them to tutorial `07-extern-function.py` as a reference.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1350631146857431103)** (5 messages): 

> `SASS compatibility with NVIDIA architectures, LD/ST unit sharing in SM microarchitecture, L1-dTLB cache, Cutlass 4.0 Python DSL, CUDA streams concurrency issues` 


- **SASS Instructions Across NVIDIA Architectures**: A member shared a [gist](https://gist.github.com/herrmann/f721da109e0c5c7c34c847ff2cf3da1e) comparing **NVIDIA SASS instructions** across different architectures, extracted and compared (using Python) from NVIDIA's HTML documentation.
   - The gist facilitates understanding of instruction set evolution across NVIDIA's GPU lineup.
- **LD/ST Unit Architecture Inquiry**: A member questioned the sharing of **LD/ST units** between scheduling units in an **SM**, referencing the **Ampere GA100 whitepaper** which divides 32 LD/ST units between 4 scheduling units.
   - They also inquired about the relationship between **LSU**, **MIO**, and **LSUIN** based on NVIDIA's nsight compute profiling guide, and if it takes 4 cycles to issue a `LDG` instruction if there are 32 threads.
- **L1-dTLB Cache Speculation**: A member speculated that the **L1/TEX cache** is **VIPT** (virtually indexed, physically tagged), guessing that address translation happens between the **LSUIN** and the tag stage.
   - No further discussion on this topic.
- **Cutlass 4.0 goes fully Python!**: A member shared that [Cutlass 4.0](https://x.com/msharmavikram/status/1901465243861373327) is now fully Python, using **Python DSL**.
   - This new version has **performance parity** with previous versions, and was presented at **NVIDIAGTC**.
- **CUDA Streams Show Concurrency Quirks**: A member encountered strange issues with **CUDA streams** not executing concurrently as expected on an **A800**, despite resource availability.
   - Analysis with **nsys** revealed prioritization of earlier streams and non-concurrent execution with specific shared memory configurations, with repeat set to **1,000,000**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/msharmavikram/status/1901465243861373327">Tweet from Vikram (@msharmavikram)</a>: Cutlass 4.0 Python DSL @__tensorcore__ Fully python! Performance parity! Join these two sessions at @NVIDIAGTC</li><li><a href="https://gist.github.com/herrmann/f721da109e0c5c7c34c847ff2cf3da1e">NVIDIA SASS Instructions&#39; history</a>: NVIDIA SASS Instructions&#39; history. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1350252666269208728)** (13 messages🔥): 

> `Torch Compile, Graph Breaks, Stride Issue, Std::variants in schemas` 


- ****Torch Compile** Struggles With Backward Pass**: A member reported that while **torch.compile** works fine for the forward pass, it is quite slow in the backward pass when using **torch.autograd.Function** for custom kernels.
   - They found that wrapping the backward function with `torch.compile(compiled_backward_fn)` could address this issue.
- ****Graph Breaks** Cause Compile Problems**: It was noted that **graph breaks** in the backward pass can cause issues with **torch.compile**.
   - One member found that using `.stride(0)` in their Triton kernel caused graph breaks, which they resolved by using **constant values** instead.
- ****Stride(0) Issue** Fixed in Nightly Builds**: A member noted that they had issues with `stride(0)` causing graph breaks in their **Triton kernel**.
   - Another member mentioned that the `stride(0)` issue has been fixed in the **PyTorch nightly builds**.
- **Schemas Struggle With **std::variants****: A member inquired about supporting `std::variants` in schemas, linking to the [relevant PyTorch code](https://github.com/pytorch/pytorch/blob/c7c3e7732443d7994303499bcb01781c9d59ab58/aten/src/ATen/core/op_registration/README.md).
   - A core dev said it's fairly hard, and that they ended up settling for `std::optional`.



**Link mentioned**: <a href="https://github.com/pytorch/pytorch/blob/c7c3e7732443d7994303499bcb01781c9d59ab58/aten/src/ATen/core/op_registration/README.md">pytorch/aten/src/ATen/core/op_registration/README.md at c7c3e7732443d7994303499bcb01781c9d59ab58 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1350537135324790925)** (1 messages): 

> `Consumer GPU Performance, AGI, Neuromorphic hardware, vast.ai` 


- **Jake Cannell's Talk on Consumer GPUs**: Jake Cannell is giving a talk in 30 min on **Consumer GPU Performance**, covering his early work in graphics, how GPUs became general purpose, his scaling journey, and the story behind [vast.ai](https://vast.ai).
   - The talk is framed as particularly relevant for those interested in **AGI** or **neuromorphic hardware**, promising a *wild* discussion.
- **Scaling Pilled Origins Revealed**: Jake Cannell will discuss how he became *scaling pilled* and the motivations behind building [vast.ai](https://vast.ai).
   - This discussion may offer insights into the infrastructure and resource demands of **AGI** and **neuromorphic hardware** research.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1350467272417742929)** (6 messages): 

> `Transformers without Normalization, LayerNorm, tanh, FA3, exp` 


- **Transformers Could Be Faster Without Normalization**: A member noted that [Transformers without Normalization](https://www.linkedin.com/posts/zhuang-liu-19306b1b1_new-cvpr-2025-paper-released-transformers-activity-7306390791361351682-Dfpb) (replacing **LayerNorm** with **tanh()**) should provide speed improvements.
   - This is because **LayerNorm** requires reductions for the mean and variance of a sequence, which can be slow, whereas **tanh()** can be calculated on every element in registers and fused with the following matmul/linear layer.
- **tanh isn't cheap, but can be approximated**: One member stated that *`tanh` in itself isn't that cheap*, as it requires an **exp** and a division.
   - On Nvidia hardware there's **`tanh.approx` (since Turing/sm_75)**, which purportedly has a throughput of **16/cycle/SM**.
- **__expf() Is Faster for Small Values**: A member suggested that **`__expf()`** is quite faster but is only good for smaller values.
   - Others pointed out that **FA3** incurred significant overhead due to the **exp** becoming a bottleneck.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1350500302796619898)** (4 messages): 

> `GPU Code Generation, ML Compiler, HPC Engineers, Superalignment Framework` 


- **GPU Mode Company Seeks AI Engineer for Code Gen**: A company is hiring an **AI Engineer** to train/fine-tune models for **GPU code generation** with decent pay and a generous equity grant, backed by **Jeff Dean** and others; apply at [jobs.mako-dev.com](https://jobs.mako-dev.com/AI-Engineer-18b546aebc368000b243eab9ff7d262c).
   - They are building a **next gen ML compiler** that integrates AI into the compilation flow; make sure to add "GPUMODE" in the optional application.
- **Sesterce Seeks HPC Engineers for Massive GB200 Clusters**: Sesterce is looking for **HPC Engineers** to build and manage their new **Giga Colossius Cluster (18K GB200)** and **Colossius Cluster (8K GB200)** with a hardcore engineering team across San Francisco, France & Singapore.
   - The team includes Awni, described as *one of the smartest and nicest people* to work with.
- **Stealth Startup Bootstraps Superalignment Framework Architect**: A stealth startup is hiring a **machine learning framework software architect** to build a **superalignment framework** on top of [ScalarLM.com](https://ScalarLM.com).
   - The ideal candidate should be prepared to write all of the code themselves for their framework as part of a small, **5-person team** and be ready for the bootstrapping life.



**Link mentioned**: <a href="https://jobs.mako-dev.com/AI-Engineer-18b546aebc368000b243eab9ff7d262c">Your connected workspace for wiki, docs &amp; projects | Notion</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1350423184243167273)** (9 messages🔥): 

> `GPU coalesced access, Nvidia GPU read operation, GPU programming, CUDA learning resources, Installing Triton` 


- **GPU coalesced access vs permuted reads**: When threads (0,1,2,3) read addresses (0,1,2,3) in a GPU, it's coalesced; however, reading at permuted addresses like (2,0,3,1) might result in **4 sequential reads** instead of a single operation.
   - If threads read addresses like **4*i+[0,1,2,3] with random i**, where each thread reads inside its own memory bank at a random address, it is not clear if this is faster than reading with bank conflicts.
- **Nvidia GPU read operations generate single request per warp**: A modern **Nvidia GPU** read operation generates a single request per warp to the **L1TEX/LSUIN**, operating at a sector level (**32 bytes**) with stored cache lines (**4 sectors**).
   - The requests are processed internally in *wavefronts*, with more details available in [Nvidia's developer forums](https://forums.developer.nvidia.com/t/wahts-the-difference-between-wavefronts-and-sectors-req/165293/4) and a [GTC Spring 21 session](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32089/).
- **Bank conflicts and memory differences explored**: A [GTC Spring 22 session](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/) helps understand bank conflicts.
   - The presenter highlights the differences between the **L1 data cache** (where the return bandwidth is cache line/cycle) and **shared memory** (which depends on bank conflicts).
- **Cloud Resources for GPU Kernel Testing**: For individuals new to GPU programming seeking to test CUDA/Cutlass/Triton kernels without local GPU access, [Google Colab](https://colab.google/) offers free access to computing resources, including GPUs and TPUs.
   - Additionally, [LeetGPU](https://leetgpu.com/) may provide alternative testing environments.
- **Troubles installing Triton on Windows**: A user encountered an error while attempting to install Triton on Windows using **pip**.
   - No specific solution was provided in the given context but the image shared in the message [shows the error](https://cdn.discordapp.com/attachments/1191300313928433664/1351213215127965696/Screenshot_2025-03-17_151415.png?ex=67d98ede&is=67d83d5e&hm=0c11c455bc52ffdca04c9d55229b42a2115ba6c469b145b7e0f27972f9ca97d6&).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://leetgpu.com/">LeetGPU</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32089/">Requests, Wavefronts, Sectors Metrics: Understanding and Optimizing Memory-Bound Kernels with Nsight Compute | GTC Digital April 2021 | NVIDIA On-Demand</a>: Learn how you can get the most out of Nsight Compute to identify and solve memory access inefficiencies in your kernel code</li><li><a href="https://colab.google/">colab.google</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/">How to Understand and Optimize Shared Memory Accesses using Nsight Compute | GTC Digital Spring 2022 | NVIDIA On-Demand</a>: For efficiently optimizing your kernel's usage of shared memory the key ingredients are: (1) a basic mental model of the hardware implementation of shared 
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1351264386089750720)** (1 messages): 

> `CUDA kernel, pytorch extension` 


- **Seeking Source Code for CUDA Kernel as PyTorch Extension**: A member inquired if the presenter from lecture 2 posted their source code for calling the **mean_filter CUDA kernel** as a **PyTorch extension**.
- **Request for CUDA Kernel Code**: A user is seeking the source code from lecture 2 to implement a **mean_filter CUDA kernel** as a **PyTorch extension**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1350904669534294229)** (3 messages): 

> `AI Agents Hackathon, NVIDIA GTC 2025, SLURM based HPC cluster IDE/Editor` 


- ****Vertex Ventures US** sponsors AI Agents Hackathon**: **Vertex Ventures US** and **CreatorsCorner** are hosting an [AI Agents Hackathon](https://lu.ma/meofrw3d) at **NVIDIA GTC 2025**, with **$50k+** in prizes.
   - Participants will build **multimodal AI agents** capable of sophisticated reasoning and interaction with various tools, with a **3-minute** showcase to judges.
- **Cursor/VSCode are IDE Favorites for HPC Cluster Dev**: A user asked what IDE/Editor people use to develop directly on a **SLURM based HPC cluster**, expressing frustration with **VSCode's** bloat in the **/home/** directory.
   - Another member suggested **Cursor/VSCode**, mentioning that most people on their work cluster use it and that the install directory can be changed.



**Link mentioned**: <a href="https://lu.ma/meofrw3d">AI Agents Hackathon - GTC 2025 Edition (1 DAY) · Luma</a>: AI Agents Hackathon - GTC 2025 Edition (1 DAY)As NVIDIA GTC 2025 unites the global AI community, Vertex Ventures US and CreatorsCorner, invite you to turns…

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1350531847481720832)** (11 messages🔥): 

> `Block Sparse Attention, GEMM, GTC Keynote Missed, GTC Hackathon results, GTC Meetup` 


- **Attendee Misses GTC Keynote Due to ESTA Error**: One attendee expressed disappointment at missing the **GTC Keynote** due to failing to get their **ESTA** filled out beforehand.
   - They mentioned the **ESTA** status had been stuck on pending for a day, preventing them from boarding their flight.
- **Inquiries about GTC Hackathon Results**: An attendee inquired about where the **GTC hackathon results** would be posted, as they did not get into the **GTC event** itself.
   - There was no answer to the user's question, which suggests the answer is unknown, or may be sent directly to those who participated.
- **Potential Post-GTC Meetup Discussed**: There was discussion about having a meetup after the **GTC conference** for those who missed the initial meetup.
   - This suggests that many were disappointed about being unable to attend **GTC**, so others agreed to organize something separate.
- **Attendees Request Slides from GTC Presentations**: Attendees requested that the slides from previous **GTC presentations** be published somewhere.
   - Another participant asked if anyone caught the last slide from **Vijay Thakkar** related to **Nvidia GTC workshops**.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1350406018466709554)** (5 messages): 

> `MI300X inference optimization, AMD Instinct MI300X workload optimization, DeepSeek-R1 on MI300X, SGLang Optimization` 


- **Experts Wanted for MI300X Inference**: A member is seeking experts to help reduce inference times on the **MI300X** and is willing to share information after the consultation.
   - They are looking for someone dedicated for a few hours of consulting, specifically for a **32B reasoner model**.
- **AMD's Inference Optimization Guide Arrives**: A member shared the [AMD Instinct™ MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) document, detailing optimization strategies for **MI300X accelerators** focusing on GPU kernel programming, HPC, and deep learning with PyTorch.
   - The document highlights auto-tunable configurations and advanced techniques like **Triton kernel optimization**.
- **DeepSeek-R1 Speeds Up with MI300X**: A blog post was shared about [unlocking DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html), highlighting performance comparisons to **H200**.
   - Optimizations using **SGLang** have reportedly unlocked up to a **4X boost** in inference speed in just two weeks, ensuring efficient scaling and lower latency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html">Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU &#8212; ROCm Blogs</a>: no description found</li><li><a href="https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html">AMD Instinct MI300X workload optimization — ROCm Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/)** (1 messages): 

leiwang1999_53585: worked on my h100, maybe you should install nightly wheel🤣
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1350210217815834765)** (5 messages): 

> `GTC CUDA, Wen-mei Hwu GTC, Pruna AI Efficiency Framework, Ruff and UV for project management` 


- ****CUDA** Content Central at **GTC****: NVIDIA is highlighting [CUDA developer sessions at GTC](https://www.nvidia.com/gtc/sessions/cuda-developer/) focused on tools and training for creating high-performance, GPU-accelerated applications.
   - Attendees can explore sessions tailored to general AI, technical details, and business strategies, with attendance on a first-come, first-served basis.
- ****Wen-mei Hwu**: the Computing Legend Signs at **GTC****: Professor **Wen-mei Hwu**, author and NVIDIA scientist, will be at #GTC25 for an exclusive meet and greet to sign copies of his [book](https://www.nvidia.com/gtc/).
   - The **GPUMODE** event is scheduled for Sunday at 6 PM and Wednesday at 2 PM at CWE75384, and you can register for the CWE event [here](https://nvda.ws/4iNYQnh).
- ****Pruna**: New AI Efficiency Framework Released**: The AI efficiency framework **Pruna** has been open-sourced, with technical details available on the [GitHub repo](https://github.com/PrunaAI/pruna/tree/main).
   - Users are encouraged to *star the repo, spread the news, and install Pruna with pip install pruna* to provide feedback.
- ****Ruff** and **UV** Simplification of dependencies**: A user suggested switching to [Ruff](https://astral.sh/ruff) + [uv](https://docs.astral.sh/uv/) for the **Pruna** project to simplify dependencies and improve project management.
   - The user believes this change would greatly simplify the dependencies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/PrunaAI/pruna/tree/main">GitHub - PrunaAI/pruna: Pruna is a model optimization framework built for developers, enabling you to deliver faster, more efficient models with minimal overhead.</a>: Pruna is a model optimization framework built for developers, enabling you to deliver faster, more efficient models with minimal overhead. - PrunaAI/pruna</li><li><a href="https://www.nvidia.com/gtc/sessions/cuda-developer/">NVIDIA GTC AI Conference 2025</a>: March 17–21, 2025. San Jose. Register Now.</li><li><a href="https://nvda.ws/4iNYQnh">NVIDIA #GTC2025 Conference Session Catalog</a>: Experience GTC 2025 In-Person and Online March 17-21, San Jose.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1350237974440185937)** (5 messages): 

> `Distributed Training, Scaling Laws for DiLoCo, GPU kernel modifications` 


- **DiLoCo scaling better than DP**: A [post on X](https://x.com/matharycharles/status/1900593694216253827?s=46) highlights a key step for making **distributed training** work at larger models, specifically **Scaling Laws for DiLoCo**.
   - The author jokes that *DiLoCo scaling better than DP is funny to me; It’s pure vibes LOL*.
- **Nuances of GPU Kernel Optimization**: A member had a horrible thought about slight modifications of a **GPU kernel** that aren't numerically equivalent but are more efficient in wall-clock time in a distributed context.
   - They feel that problems like that pose a huge problem for **automatic kernel optimization strategies**.



**Link mentioned**: <a href="https://x.com/matharycharles/status/1900593694216253827?s=46">Tweet from Zachary Charles (@MatharyCharles)</a>: We just put out a key step for making distributed training work at larger and larger models: Scaling Laws for DiLoCoTL;DR: We can do LLM training across datacenters in a way that scales incredibly wel...

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1350744003615916032)** (10 messages🔥): 

> `Reasoning Gym, nano-R1 Project, Temporal Clue, Group Relative Policy Optimization (GRPO)` 


- **Reasoning Gym Reaches 101 Datasets!**: The [Reasoning Gym](https://github.com/open-thought/reasoning-gym) project now boasts **101 datasets**, celebrating contributions from developers like Rich Jones and @jeankaddour.
   - A user shared the [X post](https://x.com/neurosp1ke/status/1901244866920636559) announcing this milestone.
- **Nano-R1 Project Eyes Reasoning Gym**: The **nano-R1** project is seeking data to evaluate runs, with a suggestion to consider using [reasoning-gym](https://github.com/open-thought/reasoning-gym) given existing benchmark scores.
   - This suggestion was made in reference to [this GitHub discussion](https://github.com/nano-R1/resources/discussions/4) about finding reasoning benchmarks.
- **Temporal Clue Puzzles Gym-Ready**: A user shared a link to [temporal-clue](https://github.com/bradhilton/temporal-clue), *Clue*-inspired puzzles for testing LLM deduction abilities, suggesting it as food for the Reasoning Gym.
   - The puzzles may be useful for testing deductive reasoning.
- **GRPO Beats Models on Temporal Clue**: [OpenPipe.ai](https://openpipe.ai/blog/using-grpo-to-beat-o1-o3-mini-and-r1-on-temporal-clue) achieved state of the art on temporal clue using **Group Relative Policy Optimization (GRPO)**, surpassing **R1**, **o1**, **o3-mini**, and nearing Sonnet 3.7's performance while being *100x* cheaper.
   - They shared a [training recipe](https://github.com/openpipe/deductive-reasoning) built on top of [torchtune](https://github.com/pytorch/torchtune) used to achieve these results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/neurosp1ke/status/1901244866920636559">Tweet from Andreas Köpf (@neurosp1ke)</a>: We now have a total of 101 datasets in reasoning-gym! 🧠💪Big THANK YOU 💙 to all devs for making this possible, especially core team Rich Jones, @zafstojano, Joe Sharratt, A. Adefioy, Ollie Stanley &...</li><li><a href="https://openpipe.ai/blog/using-grpo-to-beat-o1-o3-mini-and-r1-on-temporal-clue">Using GRPO to Beat o1, o3-mini and R1 at &quot;Temporal Clue&quot; - OpenPipe</a>: Convert expensive LLM prompts into fast, cheap fine-tuned models</li><li><a href="https://github.com/bradhilton/temporal-clue">GitHub - bradhilton/temporal-clue: Clue inspired puzzles for testing LLM deduction abilities</a>: Clue inspired puzzles for testing LLM deduction abilities - bradhilton/temporal-clue</li><li><a href="https://github.com/Tufalabs/MITIntegrationBee">GitHub - Tufalabs/MITIntegrationBee</a>: Contribute to Tufalabs/MITIntegrationBee development by creating an account on GitHub.</li><li><a href="https://github.com/nano-R1/resources/discussions/4">nano-R1: benchmarks + challenge format · nano-R1/resources · Discussion #4</a>: I think the initial goal should be to find a common set of reasoning benchmarks which are sensible to judge on, and then also a set of &quot;default&quot; training datasets + scripts so that people ca...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[active-leaderboards](https://discord.com/channels/1189498204333543425/1342999437902872727/1350871867002454046)** (1 messages): 

> `Xavier Init, User ID Issue` 


- **Usernames Replaced by Mysterious User IDs**: Some users are seeing **User_<18 digit ID>** instead of actual usernames, potentially due to a bug related to [Xavier Init](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_).
- **Ongoing username glitch**: A glitch is causing some usernames to display as generic **User_ID** strings instead of actual names.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1350196644834181130)** (15 messages🔥): 

> `pip install in popcorn, Looking for GTC 2025 Ticket, Free B200 access, AMD Support coming` 


- **Popcorn lets Users Pip Install**: Users can now `pip install` from a script in **Popcorn**, though long installations might timeout.
- **GTC 2025 Ticket Quest**: A Silicon Valley resident is seeking a ticket to the sold-out **GTC 2025** event.
   - Another member quipped, *"Sir, this is a Wendy's"*.
- **Free B200 Bonanza on Grayscale!**: One **B200** is available on the grayscale_py_b200-dev leaderboard for Grayscale, queue times may be slow due to only one device.
   - Members are encouraged to *"play with the B200 and deconstruct it however u want"*.
- **AMD support Coming Soon**: AMD Support seems to be coming soon according to a [screenshot](https://cdn.discordapp.com/attachments/1343002580531417211/1351201006297546905/image.png?ex=67d98380&is=67d83200&hm=cd8506d90a42207f7ad4fd3c10545d80661f302981288210f521412ae3edf107) *"we're cooking something finally"*.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1350586065395908788)** (29 messages🔥): 

> `Leaderboard Submissions, Benchmark Submissions, Test Submissions, Modal Runners` 


- **Grayscale Tests Triumph on T4 and H100**: Test submissions for the `grayscale` leaderboard, with IDs **2136** and **2143**, succeeded on **T4** and **H100** GPUs using Modal runners.
- **Vectoradd Aces H100**: Leaderboard submission with id **2151** to leaderboard `vectoradd` on GPUS: **H100** using Modal runners succeeded!
   - Modal Runners facilitated the successful vector addition benchmark on the high-performance **H100** GPUs.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1350254806027206819)** (1 messages): 

> `Leaderboard cleanup, Robust Evaluation` 


- **Leaderboard Cleansing Commences**: The community is now removing meme/hack entries from the leaderboard, and is asking users to submit their **Discord username**, **filename**, and **rank** if they'd like an entry deleted.
   - At the same time, changes are being made to ensure the evaluation process is more robust against these kinds of entries.
- **Evaluation Process Bolstered**: In parallel to the leaderboard cleanup, efforts are underway to make the evaluation process more robust against meme/hack entries.
   - The goal is to prevent similar issues from arising in the future.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1351215637367619656)** (1 messages): 

> `NVIDIA thermal ranges, Arithmetic and Memory Bandwidth Degradation` 


- **Member seeks NVIDIA thermal ranges and degradation info**: A member is looking for thermal ranges for different NVIDIA cards, especially info on arithmetic and memory bandwidth degradation with temperature.
   - They cited the [NVIDIA H100 product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf) as a good source of information, hoping to find similar details for more cards.
- **Discussion on hardware thermal limits**: The discussion revolved around finding detailed thermal specifications for NVIDIA cards, particularly regarding how temperature affects performance.
   - The initial requestor shared the NVIDIA H100 product brief as an example of the kind of detailed information they were seeking for a wider range of cards.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1350222797305286698)** (10 messages🔥): 

> `SMILES string encoding, Stereoisomer Generation, Free GPU Platforms, Managed Inference APIs, EleutherAI welcomes Catherine Arnett` 


- **Encoding SMILES strings into Stereoisomers**: A member inquired about models or architectures that can encode a **SMILES string** into various **stereoisomers** or encode a **ChemDraw input**.
   - The member is seeking a model that can pick up on chemical descriptors for their tasks.
- **Quest for Free GPU Platforms**: A member is looking for a **free GPU platform** beyond notebooks, needing something with **C++** support for local use with SSH.
   - Notebooks only offer a Python interface, which is insufficient for their requirements.
- **Managed Inference API Services Explored**: A member is seeking recommendations for **managed Inference API services** that small startups can use to host private models for training/finetuning **LLMs**.
   - Another member suggested [Featherless.ai](https://featherless.ai) which also supports existing LLMs from HF; it *doesn't require managing individual hardware units*.
- **EleutherAI Welcomes New NLP Researcher**: EleutherAI welcomes **Catherine Arnett**, an NLP researcher specializing in **Computational Social Science** and **cross-lingual NLP**.
   - Catherine's research focuses on ensuring models are *equally good* across languages, addressing data equivalence, performance measurement, and model building; see her recent work on [Goldfish](https://arxiv.org/abs/2408.10441), [Toxicity of the Commons](https://arxiv.org/abs/2410.22587), [LM performance on complex languages](https://arxiv.org/abs/2411.14198) and [Multilingual Language Modeling](https://arxiv.org/abs/2311.09205).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1350183448395907233)** (46 messages🔥): 

> `Block Diffusion, Globally Shared Experts, Mixture-of-Experts Universal Transformers, Tan et al.'s SUT paper, Visual Geometry Group (VGGT)` 


- ****Block Diffusion** Model Unveiled!**: A new paper introduces **Block Diffusion**, a method interpolating between autoregressive and diffusion language models, combining the strengths of both: high quality, arbitrary length, KV caching, and parallelizability, detailed in the [paper](https://arxiv.org/abs/2503.09573) and [code](https://github.com/kuleshov-group/bd3lms).
- **Exploring Globally Shared Experts in Deep Learning**: Discussion arose about research into globally shared experts, where a single pool of experts is used across all layers, with a pointer to a relevant [paper](https://arxiv.org/abs/2404.14507) on diffusion models.
- ****MoEUT**: Mixture-of-Experts Universal Transformers Paper Mentioned**: A member mentioned the **MoEUT** ([Mixture-of-Experts Universal Transformers](https://arxiv.org/abs/2503.08827)) paper as relevant to the discussion on globally shared experts, though they hadn't fully read it yet.
   - Another member suggested checking out Tan et al.'s **SUT** paper as well for related insights.
- ****VGGT** Generates 3D Scenes!**: A member shared [VGGT](https://vgg-t.github.io/), a feed-forward neural network inferring 3D attributes from multiple views and generating GLB files, which can be directly integrated into metaverses.
   - The member tested **VGGT** on old stereo images and various scenes, finding it benefits from near-angle frames; however, it may struggle with scenes lacking a clear anchor angle, with the member stating *I love that it exports GLB files. means I can drop them directly into my metaverse as-is.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vgg-t.github.io/">VGGT: Visual Geometry Grounded Transformer.</a>: We propose Visual Geometry Grounded Transformer (VGGT), a feed-forward neural network that directly predicts all key 3D scene attributes from single or multiple (up to hundreds) image views within sec...</li><li><a href="https://arxiv.org/abs/2503.08827">Neural Network/de Sitter Space Correspondence</a>: Machine learning&#39;s remarkable practical successes have sparked extensive theoretical investigations, yet fundamental breakthroughs remain elusive. Here, we study neural network training via gradie...</li><li><a href="https://arxiv.org/abs/2503.09799">Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo</a>: As we scale to more massive machine learning models, the frequent synchronization demands inherent in data-parallel approaches create significant slowdowns, posing a critical challenge to further scal...</li><li><a href="https://arxiv.org/abs/2404.14507">Align Your Steps: Optimizing Sampling Schedules in Diffusion Models</a>: Diffusion models (DMs) have established themselves as the state-of-the-art generative modeling approach in the visual domain and beyond. A crucial drawback of DMs is their slow sampling speed, relying...</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>: no description found</li><li><a href="https://m-arriola.com/bd3lms/">SOCIAL MEDIA TITLE TAG</a>: SOCIAL MEDIA DESCRIPTION TAG TAG
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1350604163402170378)** (22 messages🔥): 

> `Fewshot Split Fallback, Gen Kwargs to JSON, Old vs New LLM Leaderboard` 


- **Fewshot Split Fallback Scheme Unveiled**: When a fewshot split isn't specified, the system falls back to **train > val > test**, prioritizing the training split if available.
   - This order determines which split is used for evaluation if no specific split is defined.
- **Gen Kwargs Embraces JSON Format**: The `--gen_kwargs` argument is transitioning from comma-separated strings to **JSON**, allowing for more complex configurations like `'{"temperature":0, "stop":["abc"]}'`.
   - The discussion explores the possibility of supporting both formats for ease of use, especially for scalar values.
- **Old vs New LLM Leaderboard: Discrepancy Surfaces**: A discrepancy is identified between the group config for the old LLM leaderboard and the actual setup used, particularly concerning the **arc-challenge task**.
   - The `openllm.yaml` config specifies `validation` as the fewshot split, but the original leaderboard used the `train` split due to the absence of a fewshot split in the old fork's Python class, a [PR to fix this](https://github.com/EleutherAI/lm-evaluation-harness/pull/2802) was created to address this discrepancy.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/benchmarks/openllm.yaml">lm-evaluation-harness/lm_eval/tasks/benchmarks/openllm.yaml at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1350430340627562599)** (30 messages🔥): 

> `SDXL benchmarks, tensor cat speed, parallel BLAKE3, WebGPU integration, Bitonic Sort indices` 


- **Tinygrad SDXL Benchmarks Lag Torch**: Benchmarking **SDXL** with **tinygrad** on a **7900 XTX** shows **1.4 it/s** with **BEAM=2** on the **AMD backend**, while **torch.compile** with **FlashAttention** and **TunableOp ROCm** reaches **5.7 it/s**.
   - George Hotz suggested comparing kernels to identify optimization opportunities, aiming to surpass **torch** performance by year's end.
- **Tensor Cat Sluggish Despite Efforts**: A member is working on improving tensor cat speed, sharing whiteboard thoughts on **X** ([link](https://x.com/t0kenl1mit/status/1900952693587538018)), but notes it's still slow despite devectorizer changes.
   - The member suspects issues with generated **IR** and loading **numpy arrays**, considering custom **C/C++** via **ELF** and **LLVM** to overcome limitations.
- **BLAKE3 Bounty Status Clarified**: The status of the *High performance parallel BLAKE3* bounty was clarified, with a screenshot ([link](https://cdn.discordapp.com/attachments/1068976834928193609/1350640745505231061/Screenshot_2025-03-15_182214.png?ex=67d973f7&is=67d82277&hm=19c5ffbf47ae93d8dda6ba9c5fc1b65cc3b1df108a2f4fd5860ba66e301bef7c&)) showing the bounty status.
   - The member updated the spreadsheet and specified that the asymptotic performance is a key requirement for the bounty.
- **WebGPU Integration Gets a Boost**: A member inquired about publishing a **Tinygrad** implementation for an electron/photon classifier based on **resnet18** as an example, and was directed to a [PR for improving WebGPU integration](https://github.com/tinygrad/tinygrad/pull/9424).
   - It was suggested to create a **WebGPU** demo hosted on **GitHub Pages** with weights on **Hugging Face** for free access and testing.
- **Bitonic Sort Indices Unlocked**: During work on bitonic sort indices, a member figured out **maxpool indices**, noting that **topk** implementations are often sort-based.
   - The code is correct and jitted speed is close to **pytorch** sort (sometimes faster), the member said, it involves *a lot of kernels* due to contiguity requirements.



**Link mentioned**: <a href="https://x.com/t0kenl1mit/status/1900952693587538018">Tweet from vincent (@t0kenl1mit)</a>: Tried using compare for @__tinygrad__ tensor cat but still its slow. Attached are my whiteboard thoughts on it. I think I might have to fight ELF and link in some custom C but it might be something el...

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1350235346704797778)** (5 messages): 

> `Print Debugging Tinygrad, Lazy Computation and Gradients, Reproducer Code for Debugging, Multiline Code Blocks` 


- **Print Debugging Dilemma in Tinygrad's Lazy Mode**: A member is facing an assertion error with gradients while print-debugging intermediate tensor values in Tinygrad, despite using `.detach()`.
   - They are seeking a better method than threading the value out, due to issues with [lazy computation](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html) not being idempotent.
- **Crafting Reproducer Code for Rapid Debugging**: A member suggests creating a <= **10 line** reproducer code to quickly iterate and debug.
   - They recommended using an integrated debugger like VSCode with breakpoints and a debug console for experimenting and restarting.
- **Github Link**: A member shared a [link to a Github repo](https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1).
- **Multiline code blocks**: A member gave advice on making multiline codeblocks by using triple backticks



**Link mentioned**: <a href="https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1">gsoc_2025/ML4SCI/task1 at main · kayo09/gsoc_2025</a>: GSOC 2025! Happy Coding! ☀️. Contribute to kayo09/gsoc_2025 development by creating an account on GitHub.

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1350638802284187649)** (2 messages): 

> `Agentic Reasoning System, Corrective RAG, LlamaExtract Public Beta` 


- **Agents Reason with Corrective RAG**: A member shared a step-by-step tutorial by on how to build an **agentic reasoning system** for search and retrieval (specifically, **corrective RAG**) from scratch, orchestrated with the [@llama_index workflows](https://t.co/iDga01FouC).
   - The [tutorial](https://twitter.com/llama_index/status/1901079091345818022) lets users orchestrate complex, customizable event-driven agents.
- **LlamaExtract Enters Public Beta**: [LlamaExtract](https://twitter.com/llama_index/status/1901692607144861744) is now in public beta and solves the common problem of extracting structured data from long, complex documents, offering a **web UI** and **API**.
   - It allows users to define a schema and automatically extract structured data; more details can be found [here](https://t.co/gT3R2l7CWM).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1350518353042608229)** (31 messages🔥): 

> `AI Agents Hackathon, Vertex Ventures US, CreatorsCorner, gguf fine tuning, LlamaIndex vs Pydantic AI` 


- **Calling All AI Agents to Hackathon!**: **Vertex Ventures US** and **CreatorsCorner** invite the global AI community to turn bold ideas into action with an exclusive **AI** **hackathon** at **NVIDIA GTC 2025**.
   - The hackathon challenges participants to craft an extraordinary **multimodal AI agent** capable of sophisticated reasoning, strategic decision-making, and interacting with various tools for a chance to win **$50k+ in Prizes**!
- **Pydantic vs LlamaIndex framework face-off**: New users wonder about the difference between the **Pydantic AI** and **LlamaIndex** frameworks for building agents, especially which one to use as a beginner.
   - A LlamaIndex team member stated that whatever fits your mental model of development best is probably the best bet - but also the LlamaIndex workflows are *very nice*.
- **Data Query Agent Stuck in Infinite Viz Loop**: A user reported their **data query agent** gets stuck in an infinite loop after using the **visualization tool**, repeatedly calling the same tool.
   - Another member asked whether the user was using an open-source or closed-source LLM, and theorized, *Maybe the llm is not able to understand whether the task is finished or not*.
- **LlamaExtract is on the Cloud**: Members asked how to get access to **LlamaExtract** after seeing the GitHub repo's instructions to join the Discord.
   - The LlamaIndex team responded that it's available on [cloud.llamaindex.ai](https://cloud.llamaindex.ai) and that *LlamaExtract runs on the cloud (the client is the open-source part)*.
- **Orchestrating Agents with Sequential Workflows**: One user asked whether to use **workflows** or **agents** abstraction to build a set of agents in a linear, sequential fashion, without tethering the agent to a specific LLM provider such as Claude.
   - A LlamaIndex team member responded with a pointer to [manual tool-calling](https://docs.llamaindex.ai/en/stable/examples/llm/anthropic/#toolfunction-calling) ability of the LLM classes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/meofrw3d">AI Agents Hackathon - GTC 2025 Edition (1 DAY) · Luma</a>: AI Agents Hackathon - GTC 2025 Edition (1 DAY)As NVIDIA GTC 2025 unites the global AI community, Vertex Ventures US and CreatorsCorner, invite you to turns…</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/anthropic/#toolfunction-calling">Anthropic - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1351279240930132049)** (1 messages): 

> `Vision-Language Models (VLMs), Multimodal Learning, GitHub Research Hub` 


- **Vision-Language Models Research Hub Opens**: A member created a [community-driven hub](https://github.com/thubZ09/vision-language-model-hub.git) for multimodal researchers working on **Vision-Language Models (VLMs)**.
   - The author encourages contributions and suggestions, planning to update the hub on a weekly basis.
- **Call for Contributions to VLM Hub**: The creator of the **Vision-Language Model Hub** on GitHub is actively seeking contributions from the community.
   - They are open to suggestions and feedback, aiming to update the hub weekly to keep it a valuable resource for multimodal researchers.



**Link mentioned**: <a href="https://github.com/thubZ09/vision-language-model-hub.git">GitHub - thubZ09/vision-language-model-hub: Hub for researchers exploring VLMs and Multimodal Learning:)</a>: Hub for researchers exploring VLMs and Multimodal Learning:)  - GitHub - thubZ09/vision-language-model-hub: Hub for researchers exploring VLMs and Multimodal Learning:)

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1350192294615318659)** (29 messages🔥): 

> `Gemma 3 Integration in GPT4All, LocalDocs Crashing Fix, Gemma 3 Language Comprehension, Model license agreements` 


- **Gemma's Linguistic Prowess Surpasses Competitors in Multiple Languages**: Members found that **Gemma**, **DeepSeek R1**, and **Qwen2.5** models provided correct answers, in multiple languages, to the puzzle about *what happens when you leave a closed jar outside at minus temperature*.
   - The other models predicted catastrophic jar failure, but **Gemma** provided more helpful, nuanced advice.
- **Gemma 3 Faces Integration Issue**: Users eagerly await **Gemma 3** support in **GPT4All**, but faces delays pending updates to **Llama.cpp** due to license agreement issues on Hugging Face, detailed in [this GitHub issue](https://github.com/nomic-ai/gpt4all/issues/3540).
   - Some speculate on whether Google will police redistributions bypassing their license agreements.
- **LocalDocs Needs Crash Course Correction**: A new user experienced **LocalDoc** collection loss after a crash and subsequent reinstall, seeking advice on how to prevent data loss after the next expected crash.
   - Experienced users recommended regularly saving the *localdocs* file and restoring it after a crash, and stated that *sometimes only one bad PDF can crash the system*.
- **Level up O3-mini Explains Thinking Process**: A user shared a prompt for **O3-mini** to explain its thinking process, suggesting this could improve distillation. It can be used for any model.
   - The prompt uses **thinking** and **reflection** sections, with step-by-step reasoning and error checks.



**Link mentioned**: <a href="https://github.com/nomic-ai/gpt4all/issues/3540">Gemma 3 support · Issue #3540 · nomic-ai/gpt4all</a>: System Info I installed GPT4All, opened it, downloaded the Gemma3 Instruct for hugging face (tried two models https://huggingface.co/Mungert/gemma-3-12b-it-gguf https://huggingface.co/ggml-org/gemm...

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1350268808400736357)** (20 messages🔥): 

> `Fine-tuning for Command A, Azure Cohere Rerank v3 Terraform, Support Channel for New Models, Channel for Private Deployments of CMD A` 


- **No Fine-Tuning for Command A Yet**: A member inquired about the ETA for enabling **fine-tuning for Command A** on the Cohere platform, and a Cohere team member responded that there are *no plans yet*, but they will keep the community posted.
- **Azure Cohere Rerank v3 Terraform Troubles**: A member encountered an error while trying to create an **Azure Cohere Rerank v3** with terraform and shared the code snippet and error message.
   - A Cohere team member moved the question to the <#1324436975436038184> channel to discuss it further.
- **Private Deployment Channel on Deck?**: A member suggested creating a dedicated channel for discussions about **private deployments** of CMD A and other models, especially for efforts to get customers to deploy locally.
   - Another member agreed that it's a great idea, and requested admin <@700025263379054675> can set up.
- **Support Channel's High Volume of Questions**: A Cohere team member reminded the community to direct all support questions related to new models to the <#1324436975436038184> channel or via email at support@cohere.com.
- **CMD-A is a fan favorite**: A member stated *Loving command-a, it’s a great model*.


  

---


### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1351243287750508647)** (1 messages): 

> `Command A, Developer Office Hours, Enterprise-friendly features, Hardware vs performance` 


- **Cohere Announces March Developer Office Hours**: Cohere is hosting **Developer Office Hours** to celebrate the launch of their newest model, **Command A** on **March xx at 1 pm ET** in the Stage channel.
   - The session will cover what's new with **Command A**, enterprise-friendly features, hardware vs performance, and a live Q&A; [more details can be found here](https://discord.gg/QVyVXjST?event=1351206903426056306).
- **Command A model launch**: Cohere is launching the **Command A** model soon, and hosting office hours to celebrate.
   - The office hours will cover many topics, including: what's new, enterprise friendly features, and a live Q&A.


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1350220719589425163)** (3 messages): 

> `Cohere Command A, Vercel SDK integration, Object generation support, Cohere API versioning` 


- **Vercel SDK Misses Cohere's Object Generation**: A user reported that the [Vercel SDK](https://sdk.vercel.ai/providers/ai-sdk-providers/cohere) incorrectly assumes **object generation is not supported** by Cohere's Command A model.
   - The user intends to flag this with Vercel, suggesting it may also warrant attention from the Cohere team.
- **SDK Implementation struggles with Cohere API versions**: A user attempting to use the **OpenAI SDK** for Cohere in JavaScript encountered a warning related to the [Cohere API versioning](https://docs.cohere.com/versioning-reference).
   - The warning suggests setting an API version, as the current version is deprecated, despite the user setting both **apiKey** and **baseUrl**.
- **Cohere API Base URL Confusion clarified**: A user shared that the correct `base_url` to use is [`https://api.cohere.com/compatibility/v1/chat/completions`](https://api.cohere.com/compatibility/v1/chat/completions).
   - This URL may resolve issues related to API compatibility and versioning when integrating Cohere with other platforms or SDKs.



**Link mentioned**: <a href="https://sdk.vercel.ai/providers/ai-sdk-providers/cohere">Cohere</a>: Learn how to use the Cohere provider for the AI SDK.

  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

.paolo16: Hello
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1350992863676072106)** (3 messages): 

> `Introductions, Freelance programmers, Community Assistance` 


- **Freelance Programmer Introduces Himself**: A **30-year-old Japanese male freelance programmer** introduced himself, stating a willingness to help others through his programming skills.
   - He emphasized that *assisting one another is the pillar of our existence*.
- **Welcoming New Community Members**: The Discord server stickied a message thanking new members for joining the Cohere community.
   - It prompted them to introduce themselves by providing their **company/industry/university**, current projects, favorite tech/tools, and goals for the community.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1351212388288041083)** (13 messages🔥): 

> `dspy/MCP Integration, DSPy Assertions / Suggestions removal, DSPy 2.6 Output Refinement, QdrantRM removal in 2.6` 


- ****MCP Integration Dreams****: A member inquired about integrating **dspy/MCP**, with another noting the need for an MCP host, client, and server, pondering if it overcomplicates things, linking to a [relevant GitHub example](https://github.com/philschmid/mcp-openai-gemini-llama-example/blob/master/sqlite_llama_mcp_agent.py).
- ****DSPy Drops Assertions/Suggestions****: A user noted the [disappearance of documentation](https://dspy.ai/learn/programming/7-assertions/?h=dspy.suggest#dspyassert-and-dspysuggest-api) regarding **Assertions / Suggestions** in DSPy and inquired about their continued support.
   - They were looking to validate the outputs of the response (formatting specifically) and observed instances where the LLM does not always adhere to the format.
- ****Output Refinement as Assertion Alternative****: In **DSPy 2.6**, **Assertions** were replaced by **Output Refinement** via modules like `BestOfN` and `Refine`, designed to improve prediction reliability and quality by making multiple LM calls with different parameter settings, as detailed in the [DSPy documentation](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/).
- ****QdrantRM Questioned****: A user asked if **QdrantRM** was removed in **DSPy 2.6**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/">Output Refinement - DSPy</a>: The framework for programming—rather than prompting—language models.</li><li><a href="https://dspy.ai/learn/programming/7-assertions/?h=dspy.suggest#dspyassert-and-dspysuggest-api">DSPy Assertions - DSPy</a>: The framework for programming—rather than prompting—language models.</li><li><a href="https://github.com/philschmid/mcp-openai-gemini-llama-example/blob/master/sqlite_llama_mcp_agent.py">mcp-openai-gemini-llama-example/sqlite_llama_mcp_agent.py at master · philschmid/mcp-openai-gemini-llama-example</a>: Contribute to philschmid/mcp-openai-gemini-llama-example development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1351252518205919368)** (1 messages): 

> `Caiming Xiong, Multimodal Agents, Vision-Language-Action Alignment, OSWorld, AgentTrek` 


- **Salesforce's Caiming Xiong to Present on Multimodal Agents**: Caiming Xiong, SVP of AI Research at Salesforce, will present a lecture on **Multimodal Agents** today at 4pm PDT, live-streamed on [YouTube](https://www.youtube.com/live/n__Tim8K2IY).
   - The talk will cover integrating **perception, grounding, reasoning, and action** across multiple modalities to transform tasks like **GUI automation and household robotics**.
- **Multimodal Agents Landscape Explored**: The lecture will explore measuring capabilities in realistic environments (**OSWorld**), creating large-scale datasets (**AgentTrek**), and designing advanced modeling architectures (**Aguvis, Magma**).
   - It will also discuss incorporating synthetic chain-of-thought-and-action (**TACO**) for more robust **vision-language-action alignment**.
- **Caiming Xiong's Background**: Caiming Xiong earned his Ph.D. in Computer Science from SUNY at Buffalo, specializing in areas such as **natural language processing, computer vision, reinforcement learning, and deep learning**.
   - He has published more than **200 papers** with **>50,000 citations** and served on the organizing committees of multiple workshops.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1350958757718528060)** (7 messages): 

> `Advanced LLM agent course enrollment, Course certification` 


- **Advanced LLM agent course still accepting!**: Members inquired whether they can still sign up for the **Advanced LLM agent course**.
   - The staff replied that you just need to complete the **signup form**!
- **Certificate still attainable!**: Members inquired whether they can still get the **certificate** after signing up for the course.
   - The staff replied that most of the info on that intro slide deck only applies to **Berkeley students** and that one can absolutely still enroll in the **MOOC** and earn a **certificate** at the end!


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1350293966460948532)** (4 messages): 

> `Self-reflection and self-refinement in LLMs, System prompts and LLM behavior` 


- **Self-Reflection Dichotomy Dilemma Discussed**: A member highlighted a contradiction between **Lecture 1**, stating that **self-reflection and self-refinement** require external evaluation, and **Lecture 2**, suggesting LLMs can improve by rewarding their own outputs.
   - Screenshots from **Lecture 1, slide 67** and **Lecture 2, slide 51** were attached to illustrate the apparent conflict.  See [image 1](https://cdn.discordapp.com/attachments/1282734248112947210/1351127068745928816/image.png?ex=67d9e763&is=67d895e3&hm=7d31b7a0583550a36a872d74bfaf765de39c6b1173333d2ce51174940c0aa522&) and [image 2](https://cdn.discordapp.com/attachments/1282734248112947210/1351127069169418260/image.png?ex=67d9e764&is=67d895e4&hm=12bbe1810790f7f688b11fe093f693a2791e94bd9e74e71ec7c2cfa3264bd004&).
- **System Prompt Reliability Questioned**: A member suggested that while system prompts should work, relying on specific behaviors might not be robust, because *all these at the end is text input, so the model can process it. You should be able to bypass the framework and service.*
   - They added that the training data looks like `<system> You are a helpful assistant </system> <user> {{Some example user prompt}} </user> <assistant> {{Expected LLM output}} </assistant>` and that frameworks may not reliably pass system prompts to all LLMs.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1350548977237622916)** (5 messages): 

> `Modular AI Art, Discord Spam` 


- **Modular's AI Art Appreciated**: A member expressed appreciation for the **AI art** used by **Modular**.
   - They stated, *"all the AI art that modular uses is great!"*
- **Discord Spam Clarification**: A member clarified that certain messages in the Discord channel were spam.
   - Another member acknowledged the clarification with a thumbs up.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1350217685136572466)** (6 messages): 

> `Compact Dict, SIMD, stdlib Dict` 


- **Compact Dict's Current Status**: Members discussed the current status of the [compact-dict](https://github.com/mzaks/compact-dict) implementation, noting that its original version might be outdated.
   - It was suggested that most of the compact dict's functionality got upstreamed into the `Dict` in the **stdlib**.
- **stdlib Dict Performance Issues with SIMD**: One user reported performance issues when using the **stdlib Dict** with **SIMD** [float64, 1] types.
   - They were using the `hash()` function from the hash lib and found it to be slow, leading them to search for faster alternatives.



**Link mentioned**: <a href="https://github.com/mzaks/compact-dict">GitHub - mzaks/compact-dict: A fast and compact Dict implementation in Mojo 🔥</a>: A fast and compact Dict implementation in Mojo 🔥. Contribute to mzaks/compact-dict development by creating an account on GitHub.

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1350729782769221642)** (2 messages): 

> `AI4Legislation Competition, AI Demo Jam, Silicon Valley Chinese Association Foundation, Dnipro VC, Data Phoenix` 


- ****AI4Legislation** Competition Launches!**: The Silicon Valley Chinese Association Foundation (SVCAF) is holding the **AI4Legislation** competition with prizes up to **$3,000**, running until **July 31, 2025**, encouraging open-source AI solutions for legislative engagement; the [competition repo](https://github.com/svcaf/2025-AI4Legislation-Public) is now available.
   - SVCAF will conduct an online seminar about the competition at the end of March 2025, featuring leaders in AI and legislation; RSVP [here](https://forms.gle/pmbkRLVurbXcGBbAA).
- **Get Jamming at the **AI Demo Jam**!**: On **March 20** in Sunnyvale, CA, **Dnipro VC** and **Data Phoenix** will be hosting **AI Demo Jam**, featuring 5 AI startups showcasing their products, expert panel discussions, open mic pitches, and high-energy networking.
   - The panel will include ​Marianna Bonechi (**Dnipro VC**), ​Nick Bilogorskiy (**Dnipro VC**), ​Dmytro Dzhulgakhov (**fireworks.ai**); register [here](https://lu.ma/AI-demo-jam).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/AI-demo-jam">AI Demo Jam: Where Innovation Meets Action · Luma</a>: Dnipro VC and Data Phoenix are proud to present:On March 20, step into the future of AI at AI Demo Jam — a recurring event series that brings together…</li><li><a href="https://forms.gle/pmbkRLVurbXcGBbAA">March AI4Legislation Seminar RSVP</a>: Thank you for your interest in SVCAF&#39;s AI4Legislation seminar!Silicon Valley Chinese Association Foundation (incorporated in 2015) is holding a competition this summer to develop open-source AI-dr...
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1350729843007819796)** (2 messages): 

> `AI4Legislation competition, object detection in MRI` 


- **AI4Legislation Competition Launch**: The Silicon Valley Chinese Association Foundation is holding the **AI4Legislation** competition until **July 31, 2025**, encouraging open-source AI solutions for citizen engagement in the legislative process.
   - Prizes range from **$1,000** to **$3,000**, and you can find more details in the [competition's GitHub repository](https://github.com/svcaf/2025-AI4Legislation-Public) and RSVP for the seminar [here](https://forms.gle/pmbkRLVurbXcGBbAA).
- **Community call for MRI Object Detection**: A member requested help to create a model for **object detection in MRI images** without monetary compensation.
   - No specific details were provided on the type of model, data availability, or use case.



**Link mentioned**: <a href="https://forms.gle/pmbkRLVurbXcGBbAA">March AI4Legislation Seminar RSVP</a>: Thank you for your interest in SVCAF&#39;s AI4Legislation seminar!Silicon Valley Chinese Association Foundation (incorporated in 2015) is holding a competition this summer to develop open-source AI-dr...

  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1350740836987699251)** (2 messages): 

> `Qdrant` 


- **Qdrant Request Denied**: A member suggested switching to **Qdrant**, but another member confirmed that they are not currently using it.
   - The conversation provides no further context on the reasons for not using **Qdrant** or potential future considerations.
- **No Qdrant Here!**: A user inquired about changing a system to use **Qdrant**, a vector database.
   - However, another user firmly stated, *No we are not using Qdrant*, putting an end to the suggestion without further explanation.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1350433596686467082)** (2 messages): 

> `API Feature Requests, Repetition Penalty` 


- **API Repetition Penalty Support Requested**: A user requested the addition of **repetition penalty support** to the API, indicating it's a key feature preventing wider adoption.
   - The user stated that the lack of repetition penalty support is the *only limiting factor* for their increased usage of the model.
- **Repetition Penalty as Key Adoption Hurdle**: The user emphasized that the absence of **repetition penalty** functionality is the primary obstacle preventing them from utilizing the model more extensively.
   - No additional context or alternative solutions were discussed in the provided message.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

yamashi: https://mistral.ai/news/mistral-small-3-1
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1350472840113033267)** (2 messages): 

> `Learnable Scalars, Mitigating Issues in Models, Model Convergence` 


- **Learnable Scalars Mitigate Model Issues**: A user shared a link to a paper [Mitigating Issues in Models with Learnable Scalars](https://www.alphaxiv.org/abs/2503.10622).
   - The author of the message also noted that *the issue is mitigated by incorporating a learnable scalar, and the model can converge normally*.
- **Model Convergence Improved**: The learnable scalar helps the model *converge normally*.
   - This suggests a practical approach to stabilizing training.



**Link mentioned**: <a href="https://www.alphaxiv.org/abs/2503.10622">Transformers without Normalization | alphaXiv</a>: View 1 comments: Awesome work!Transformers without Normalization podcast

  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
