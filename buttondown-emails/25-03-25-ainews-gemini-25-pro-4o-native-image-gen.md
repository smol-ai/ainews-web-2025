---
id: 270c74dd-e984-447a-b1e7-90637af80bd2
title: Gemini 2.5 Pro + 4o Native Image Gen
date: '2025-03-26T01:13:42.288748Z'
original_slug: ainews-gemini-25-pro-4o-native-image-gen
description: >-
  **Gemini 2.5 Pro** from **Google DeepMind** has become the new top AI model,
  leading by 40 LMarena points over **Grok 3**. It features advanced reasoning,
  coding, STEM, and multimodal capabilities, with involvement from **Noam
  Shazeer**. Meanwhile, **OpenAI** released **GPT 4o Native Images**, an
  autoregressive model for image generation with detailed technical insights
  shared by **Allan Jabri**. Both models highlight significant advances in
  **multimodality**, **autoregressive transformers**, and **instruction
  following**.
companies:
  - google-deepmind
  - openai
  - lmarena_ai
models:
  - gemini-2.5-pro
  - gpt-4o
topics:
  - multimodality
  - autoregressive
  - instruction-following
  - coding
  - stem
  - model-releases
  - model-performance
  - transformers
  - image-generation
  - reasoning
people:
  - noam_shazeer
  - allan_jabri
  - jack_w_rae
  - simon_willison
  - paul_gauthier
  - andrew_carr
---


<!-- buttondown-editor-mode: plaintext -->**What a time to be alive.**

> AI News for 3/24/2025-3/25/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**228** channels, and **6171** messages) for you. Estimated reading time saved (at 200wpm): **566 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Both frontier lab releases from today were title page worthy, so they will have to share space.

## Gemini 2.5 Pro

[**Gemini 2.5 Pro**](https://news.ycombinator.com/item?id=43473489) is the new undisputed top model in the world, a whopping 40 LMarena points over Grok 3 from just last month ([our coverage here](https://buttondown.com/ainews/archive/ainews-xai-grok-3-and-mira-muratis-thinking/)), with [Noam Shazeer's involvement](https://twitter.com/NoamShazeer/status/1904581813215125787) hinting that the learnings from Flash Thinking have been merged into Pro (odd how 2.5 Pro came out first before 2.5 Flash?) 

![image.png](https://assets.buttondown.email/images/797084f7-b6d9-4785-9ad9-7d7bba2ab259.png?w=960&fit=max)

[Simon Willison](https://simonwillison.net/2025/Mar/25/gemini/), [Paul Gauthier (aider)](https://x.com/paulgauthier/status/1904637913411031410), [Andrew Carr](https://x.com/andrew_n_carr/status/1904607188976611627) and others all have worthwhile quick hits to the theme of "this model is SOTA".

Pricing is not yet announced but you can use it as a **free**, rate limited "experimental model" today.

## GPT 4o Native Images

Hot on the heels of [yesterday's Reve Image](https://buttondown.com/ainews/archive/ainews-halfmoon-is-reve-image-a-new-sota-image/) and [Gemini's Native Image Gen](https://buttondown.com/ainews/archive/ainews-gemma-3-beats-deepseek-v3-in-elo-20-flash/), OpenAI finally released their 4o native image gen with a [livestream](https://www.youtube.com/live/2f3K43FHRKo?si=QX6oXEalK8XRSvrP), [blogpost](https://news.ycombinator.com/item?id=43474112), and [system card](https://openai.com/index/gpt-4o-image-generation-system-card-addendum/) confirming that it is an autoregressive model. The most detail we'll probably get from now about how it works, is this image from [Allan Jabri](https://x.com/ajabri/status/1904599427366739975) who worked on the original 4o image gen that was never released (then taken over by [Gabe Goh as sama credits him](https://x.com/sama/status/1904599358756315341)).


![image.png](https://assets.buttondown.email/images/ea1255e2-746c-4ee9-8cc3-c2da91fde74d.png?w=960&fit=max)

> A wide image taken with a phone of a glass whiteboard, in a room overlooking the Bay Bridge. The field of view shows a woman writing, sporting a tshirt wiith a large OpenAI logo. The handwriting looks natural and a bit messy, and we see the photographer's reflection. The text reads: (left) "Transfer between Modalities: Suppose we directly model p(text, pixels, sound) [equation] with one big autoregressive transformer. Pros: * image generation augmented with vast world knowledge * next-level text rendering * native in-context learning * unified post-training stack Cons: * varying bit-rate across modalities * compute not adaptive" (Right) "Fixes: * model compressed representations * compose autoregressive prior with a powerful decoder" On the bottom right of the board, she draws a diagram: "tokens -> [transformer] -> [diffusion] -> pixels"

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Model Releases and Announcements**

- **Google's Gemini 2.5 Pro** is making waves with several key announcements: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1904579660740256022) introduced **Gemini 2.5 Pro Experimental** as their most intelligent model, emphasizing its reasoning capabilities and improved accuracy, with details available in their [blog](https://t.co/Gx35wlkomq).  [@NoamShazeer](https://twitter.com/NoamShazeer/status/1904581813215125787) highlighted that the **2.5 series** marks an evolution to fundamentally thinking models, reasoning before responding. It excels in coding, STEM, multimodal tasks, instruction following, and it is #1 on the [@lmarena_ai](https://twitter.com/lmarena_ai/status/1904581128746656099) leaderboard by a drastic 40 ELO margin, as well as its exceptional coding performance. It's topped [@lmarena_ai's leaderboard](https://twitter.com/Google/status/1904581629017735261) by a huge margin. [@jack_w_rae](https://twitter.com/jack_w_rae/status/1904583894458110218) noted **2.5 Pro improves** in coding, STEM, multimodal tasks, and instruction following, available in AI Studio & the Gemini App. 
- **Availability of Gemini 2.5 Pro**: Developers can access it in [Google AI Studio](https://twitter.com/GoogleDeepMind/status/1904581166755123463) and [@GeminiApp](https://twitter.com/GoogleDeepMind/status/1904581166755123463) for Advanced users, with Vertex AI availability coming soon and is also free for everyone to use, according to [@casper_hansen_](https://twitter.com/casper_hansen_/status/1904590489128440163).  [@stevenheidel](https://twitter.com/stevenheidel/status/1904601168317399199) shared a pro tip to use the new image generation on [this site](https://t.co/8Xy5cfz7Y2), where users can set aspect ratios and generate multiple variations.
- **DeepSeek V3-0324 release**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1904467255083348244) reported that **DeepSeek V3-0324** is now the highest scoring non-reasoning model, marking the first time an open weights model leads in this category. Details for the model are mostly identical to the December 2024 version, including a 128k context window (limited to 64k on DeepSeek’s API), 671B total parameters, and MIT License.  [@reach_vb](https://twitter.com/reach_vb/status/1904447298811437061) noted the model beats/competitive to Sonnet 3.7 & GPT4.5 with MIT License,  Improved the executability of the code and a more aesthetically pleasing web pages and game front-ends.
- **OpenAI's Image Generation**: [@OpenAI](https://twitter.com/OpenAI/status/1904602845221187829) announced that 4o image generation has arrived. It's beginning to roll out today in ChatGPT and Sora to all Plus, Pro, Team, and Free users. [@kevinweil](https://twitter.com/kevinweil/status/1904595752380465645) said that there's a major update to image generation in ChatGPT is now quite good at following complex instructions, including detailed visual layouts. It's very good at generating text and can do photorealism or any number of other styles.

**Benchmarks and Performance Evaluations**

- **Gemini 2.5 Pro's performance**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1904581128746656099) announced that **Gemini 2.5 Pro** is now #1 on the Arena leaderboard, tested under codename "nebula." It ranked #1 across ALL categories and UNIQUELY #1 in Math, Creative Writing, Instruction Following, Longer Query, and Multi-Turn! [@YiTayML](https://twitter.com/YiTayML/status/1904598794278494272) stated Google is winning by so much and Gemini 2.5 pro is the best model in the world.  [@alexandr_wang](https://twitter.com/alexandr_wang/status/1904590438469951873) also noted that **Gemini 2.5 Pro Exp dropped** and it's now #1 across SEAL leaderboards.  [@demishassabis](https://twitter.com/demishassabis/status/1904587103805006218) summarized that **Gemini 2.5 Pro** is an awesome state-of-the-art model, no.1 on LMArena by a whopping +39 ELO points. [@OriolVinyalsML](https://twitter.com/OriolVinyalsML/status/1904583691566727361) added that Gemini 2.5 Pro Experimental has stellar performance across math and science benchmarks.
- **DeepSeek V3-0324 vs. Other Models**:  [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1904467262364692970) noted that compared to leading reasoning models, including DeepSeek’s own R1, **DeepSeek V3-0324** remains behind.  [@teortaxesTex](https://twitter.com/teortaxesTex/status/1904364173426893235) highlighted that **Deepseek API change log** is updated for 0324 with substantial improvements across benchmarks like MMLU-Pro, GPQA, AIME, and LiveCodeBench.  [@reach_vb](https://twitter.com/reach_vb/status/1904447298811437061) shared benchmark improvements from DeepSeek V3 0324.

**AI Applications and Tools**

- **AI-Powered Tools for Workflows**:  [@jefrankle](https://twitter.com/jefrankle/status/1904590481222218176) discussed **TAO**, a new finetuning method from @databricks that only needs inputs and no labels, beating supervised finetuning on labeled data.  [@jerryjliu0](https://twitter.com/jerryjliu0/status/1904328371867361469) introduced **LlamaExtract**, which transforms complex invoices into standardized schemas and is tuned for high-accuracy.
- **Weights & Biases AI Agent Tooling**: [@weights_biases](https://twitter.com/weights_biases/status/1904524590988013951) announced that their @crewAIInc integration in @weave_wb is officially live. Now track every agent, task, LLM call, latency, and cost—all unified in one powerful interface.
- **Langchain Updates**: [@hwchase17](https://twitter.com/hwchase17/status/1904589229856080084) Highlighted the availability of the **Langgraph computer use agent**.  [@LangChainAI](https://twitter.com/LangChainAI/status/1904589968116420924) mentioned that the Want to use OpenAI computer use model in a langgraph agent?, this is the easiest way to do so!

**Research and Development**

- **AI in Robotics**: [@adcock_brett](https://twitter.com/adcock_brett/status/1904535004866052228) from Figure noted that they have a neural net capable of walking naturally like a human and discuss using Reinforcement Learning, training in simulation, and zero-shot transfer to our robot fleet in [this writeup](https://t.co/l63afSgBd0).  [@hardmaru](https://twitter.com/hardmaru/status/1904320457396162563) said that they are super proud of the team and  thinks that this US-Japan Defense challenge is just the first step for @SakanaAILabs to help accelerate defense innovation in Japan.
- **New Architectures**: Nvidia presents [FFN Fusion: Rethinking Sequential Computation in Large Language Models](https://twitter.com/_akhaliq/status/1904390303458459821), with [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1904370227665912243) noting it achieves a 1.71x speedup in inference latency and 35x lower per-token cost.
- **Approaches To Text-to-Video Models**: AMD released [AMD-Hummingbird on Hugging Face](https://twitter.com/_akhaliq/status/1904386209373118623) - "Towards an Efficient Text-to-Video Model".

**AI Ethics and Societal Impact**

- **Freedom and Responsible AI**: [@sama](https://twitter.com/sama/status/1904598788687487422) discussed OpenAI's approach to creative freedom with the new image generation, aiming for the tool not to create offensive stuff unless users want it to, within reason. He emphasizes the importance of respecting societal bounds for AI.  [@ClementDelangue](https://twitter.com/ClementDelangue/status/1904547109551919174) recommended open-source AI and controlled autonomy for AI systems to reduce cybersecurity risks.
- **AI Benchmarking**: [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1904572772833014074) interviewed @js_denain about how today’s benchmarks fall short and how improved evaluations can better reveal AI’s real-world capabilities.

**Humor and Miscellaneous**

- **Elon Musk and Grok**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1904373288463401466) asked [@elonmusk](https://twitter.com/elonmusk/status/1904373288463401466) when Grok3 "big brain" mode be released?
- [@sama](https://twitter.com/sama/status/1904604934387229134) posted [@NickADobos](https://twitter.com/NickADobos/status/1904604934387229134) hot guy though!
- [@giffmana](https://twitter.com/giffmana/status/1904625459641671864) said ouch Charlie, that really hurt!
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1904346173747437627) asks [@FearedBuck](https://twitter.com/FearedBuck/status/1904346173747437627) why is he a panda?


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek V3 0324 Tops Non-Reasoning Model Charts**

- **[Deepseek V3 0324 is now the best non-reasoning model (across both open and closed source) according to Artificial Analisys.](https://i.redd.it/4hh6ys9gftqe1.png)** ([Score: 736, Comments: 114](https://reddit.com/r/LocalLLaMA/comments/1jjgi8y/deepseek_v3_0324_is_now_the_best_nonreasoning/)): **Deepseek V3 0324** has been recognized as the top non-reasoning AI model by **Artificial Analisys**, outperforming both open and closed source models. It leads the **Artificial Analysis Intelligence Index** with a score of **53**, surpassing models like **Grok-3** and **GPT-4.5 (Preview)**, which scored **53** and **51**, respectively; the index evaluates models on criteria such as reasoning, knowledge, math, and coding.
  - There is skepticism about the reliability of benchmarks, with users like **artisticMink** and **megazver** expressing concerns that benchmarks may not accurately reflect real-world performance and might be biased towards newer models. **RMCPhoto** and **FullOf_Bad_Ideas** also noted that certain models like **DeepSeek V3** and **QWQ-32B** may not perform as well in practical applications compared to others like **claude 3.7**.
  - Users discussed the accessibility and usage of models like **DeepSeek V3** and **Gemma 3**, with **Charuru** providing access information through **deepseek.com** and **yur_mom** discussing subscription options for unlimited usage. **East-Cauliflower-150** and **emsiem22** highlighted the impressive capabilities of **Gemma 3** despite its **27B parameters**.
  - The community expressed interest in upcoming models and updates, such as **DeepSeek R2** and **Llama 4**, with **Lissanro** awaiting dynamic quant releases from **Unsloth** on [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF). Concerns were raised about the pressure on teams like **Meta Llama** to release competitive models amidst ongoing developments.


- **DeepSeek-V3-0324 GGUF - Unsloth** ([Score: 195, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1jji2da/deepseekv30324_gguf_unsloth/)): The **DeepSeek-V3-0324 GGUF** model is available in multiple formats ranging from **140.2 GB** to **1765.3 GB** on [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF). Users can currently access 2, 3, and 4-bit dynamic quantizations, with further uploads and testing in progress as noted by **u/yoracale**.
  - **Dynamic Quantization Performance**: Users discuss the performance impact of different quantization methods on large language models (LLMs). **Standard 2-bit quantization** is criticized for poor performance, while the **2.51 dynamic quant** shows significant improvements in generating functional code.
  - **Hardware and Resource Constraints**: There is a discussion on the impracticality of running nearly **2TB models** without significant computational resources, with suggestions like using a **4x Mac Studio 512GB cluster**. Some users express challenges with available VRAM, indicating that even **190GB** isn't sufficient for optimal performance.
  - **Upcoming Releases and Recommendations**: Users are advised to wait for the **dynamic IQ2_XSS quant**, which promises better efficiency than the current **Q2_K_XL**. **Unsloth's IQ2_XXS R1** is noted for its efficiency despite its smaller size, and there are ongoing efforts to upload more dynamic quantizations like the **4.5-bit version**.


- **[DeepSeek official communication on X: DeepSeek-V3-0324 is out now!](https://www.reddit.com/gallery/1jjjv8k)** ([Score: 202, Comments: 7](https://reddit.com/r/LocalLLaMA/comments/1jjjv8k/deepseek_official_communication_on_x/)): **DeepSeek** announced on X the release of **DeepSeek-V3-0324**, now available on **Huggingface**.
  - **DeepSeek-V3-0324** is now available on **Huggingface**, with the release officially announced on **X**. The status update can be found at [DeepSeek AI's X page](https://x.com/deepseek_ai/status/1904526863604883661).
  - A humorous future prediction lists **DeepSeek-V3-230624** among the top models of the year **2123**, alongside other models like **GPT-4.99z** and **Llama-33.3333**.


**Theme 2. Dynamic Quants for DeepSeek V3 Boost Deployments**

- **DeepSeek-V3-0324 HF Model Card Updated With Benchmarks** ([Score: 145, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1jjdv9n/deepseekv30324_hf_model_card_updated_with/)): The **DeepSeek-V3-0324 HF Model Card** has been updated with new benchmarks, as detailed in the [README](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/README.md). This update provides insights into the model's performance and capabilities.
  - There is a discussion about the **temperature parameter** in the model, where input values are transformed: values between **0 and 1** are multiplied by **0.3**, and values over **1** have **0.7 subtracted**. Some users find this transformation helpful, while others suggest making the field required for clarity.
  - **Sam Altman** and his views on **OpenAI's competitive edge** are mentioned, with some users referencing an interview where he claimed that other companies would struggle to compete with OpenAI. This sparked comments about his financial success and management style.
  - There are mixed opinions on the model's capabilities, with some users impressed by its performance as a "non-thinking model," while others see only minor improvements or express skepticism about its complexity.


**Theme 3. Gemini 2.5 Pro Dominates Benchmarks with New Features**

- **NEW GEMINI 2.5 just dropped** ([Score: 299, Comments: 116](https://reddit.com/r/LocalLLaMA/comments/1jjole9/new_gemini_25_just_dropped/)): **Gemini 2.5 Pro Experimental** by **Google DeepMind** has set new benchmarks, outperforming **GPT-4.5** and **Claude 3.7 Sonnet** on **LMArena** and achieving **18.8%** on "Humanity’s Last Exam." It excels in math and science, leading in **GPQA Diamond** and **AIME 2025**, supports a **1M token context window** with **2M** coming soon, and scores **63.8%** on **SWE-Bench Verified** for advanced coding capabilities. More details can be found in the [official blog](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning).
  - There is significant discussion about **Gemini 2.5 Pro's** proprietary nature and lack of open-source availability, with users expressing a desire for more transparency such as a **model card** and **arxiv paper**. Concerns about privacy and the ability to run models locally were highlighted, with some users pointing out that alternative models with open weights are more appealing for certain use cases.
  - **Gemini 2.5 Pro's** performance in **coding tasks** is debated, with some users reporting impressive results, while others question its effectiveness without concrete evidence. The model's large **1M token context window** and **multi-modal capabilities** are praised, positioning it as a competitive alternative to **Anthropic** and **Closed AI** offerings, especially given its cost-effectiveness and integration with Google's ecosystem.
  - The use of certain benchmarks, such as high school math competitions, for evaluating AI models is criticized, with calls for more independent and varied evaluation methods. Despite this, some users defend these benchmarks, noting their correlation with other closed math benchmarks and the difficulty level of the tests.


- **[Mario game made by new a Gemini pro 2.5 in couple minutes - best version I ever saw. Even great physics!](https://v.redd.it/955pvmtd4wqe1)** ([Score: 99, Comments: 38](https://reddit.com/r/LocalLLaMA/comments/1jjsiiw/mario_game_made_by_new_a_gemini_pro_25_in_couple/)): **Gemini Pro 2.5** is highlighted for its ability to create a **Mario game** with impressive coding efficiency and realistic physics in just a few minutes. The post suggests that this version of the game demonstrates exceptional quality and technical execution.
  - Users express amazement at the rapid improvement of **LLMs**, noting that **6 months ago** they struggled with simple games like **Snake**, and now they can create complex games like **Mario** with advanced code quality.
  - **Healthy-Nebula-3603** shares the **prompt** and **code** used for the Mario game, available on [Pastebin](https://pastebin.com/TqvbrA0T), which specifies building the game in **Python** without external assets, including features like a title screen and obstacles.
  - Some users humorously reference potential copyright issues with **Nintendo**, while others discuss the prompt's availability and the community's eagerness to replicate the results with other old games.


**Theme 4. Affordable AI Hardware: Phi-4 Q4 Server Builds**

- **[$150 Phi-4 Q4 server](https://www.reddit.com/gallery/1jjddzl)** ([Score: 119, Comments: 26](https://reddit.com/r/LocalLLaMA/comments/1jjddzl/150_phi4_q4_server/)): The author built a local **LLM server** using a **P102-100 GPU** purchased for **$42** on eBay, integrated into an **i7-10700 HP prebuilt** system. After upgrading with a **$65 500W PSU** and new cooling components, they achieved a **10GB CUDA box** capable of running an **8.5GB Q4 quant of Phi-4** at **10-20 tokens per second**, maintaining temperatures between **60°C-70°C**.
  - **Phi-4 Model Performance**: Users praised the **Phi-4** model for its efficiency in handling tasks like form filling, JSON creation, and web programming. It is favored for its ability to debug and modify code, with comparisons suggesting it outperforms other models in similar tasks.
  - **Hardware Setup and Modifications**: Discussion included details on hardware modifications like using a **$65 500W PSU**, thermal pads, and fans. Links to resources like [Nvidia Patcher](https://github.com/dartraiden/NVIDIA-patcher) and [Modified BIOS for full VRAM](https://www.techpowerup.com/vgabios/249516/249516) were shared to enhance the performance of the **P102-100 GPU**.
  - **Cost and Efficiency Considerations**: The setup, including an **i7-10700 HP prebuilt system**, was noted for its cost efficiency, operating at around **400W** and costing approximately **2 cents per hour** at **$0.07 per kWh**. Comparisons were made to services like **OpenRouter**, emphasizing the benefits of local data processing and cost savings.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. DeepSeek V3 Outperforming GPT-4.5 in New Benchmarks**

- **[GPT 4.5 got eclipsed.. DeepSeek V3 is now top non-reasoning model! & open source too. So Mr 'Open'AI come to light.. before R2🪓](https://i.redd.it/rf6ngx2lttqe1.jpeg)** ([Score: 333, Comments: 113](https://reddit.com/r/OpenAI/comments/1jjhrx3/gpt_45_got_eclipsed_deepseek_v3_is_now_top/)): **DeepSeek V3** has been released as an open-source model and is now the top non-reasoning AI model, surpassing **GPT-4.5** with a performance score of **53** as of March 2025. This release challenges **OpenAI** to maintain transparency and openness in light of DeepSeek V3's success.
  - Several commenters question the **benchmark validity** used to compare DeepSeek V3 and GPT-4.5, with skepticism about the absence of confidence intervals and potential over-optimization against static tests. The importance of human evaluations, like those from [lmarena.ai](https://lmarena.ai/), is highlighted for providing more subjective measures of model performance.
  - Concerns are raised about **OpenAI's response** to competition, speculating that they may focus on qualitative aspects like how their models "feel" rather than quantitative benchmarks. Some users express support for increased competition to drive innovation and improvements.
  - Discussions touch on the **limitations of current AI models**, noting that while larger models like 400B and 4T parameters exist, they show diminishing returns compared to smaller models. This suggests a potential ceiling in Transformer AI capabilities, indicating no imminent arrival of AGI and continued relevance for programmers in the job market.


- **Claude Sonnet 3.7 vs  DeepSeek V3 0324** ([Score: 246, Comments: 101](https://reddit.com/r/ClaudeAI/comments/1jjeobd/claude_sonnet_37_vs_deepseek_v3_0324/)): The post compares **Claude Sonnet 3.7** and **DeepSeek V3 0324** by generating landing page headers, highlighting that **DeepSeek V3 0324** appears to have no training influence from **Sonnet 3.7**. The author provides links to the generated images for both models, showcasing distinct outputs.
  - Discussions highlight skepticism towards AI companies' data practices, with references to **copyright issues** and **uncompensated content**. Some users suggest AI models like **DeepSeek V3** and **Claude Sonnet 3.7** may share training data from sources like **Themeforest** or **open-source** contributions, questioning proprietary claims ([Wired article](https://www.wired.com/story/new-documents-unredacted-meta-copyright-ai-lawsuit/)).
  - **DeepSeek V3** is praised for its **open-source** nature, with available weights and libraries on platforms like [Hugging Face](https://huggingface.co/deepseek-ai) and [GitHub](https://github.com/deepseek-ai), allowing users with sufficient hardware to host it. Users appreciate its transparency and suggest **OpenAI** and **Anthropic** could benefit from similar practices.
  - The community debates the quality of outputs, with some favoring **Claude's** design for its professional look and usability, while others argue **DeepSeek** offers valuable open-source contributions despite potential similarities in training data. Concerns about AI's impact on innovation and the ethical use of training data persist, with calls for more open-source contributions from major AI companies.


**Theme 2. OpenAI 4o Revolutionizing Image Generation**

- **[Starting today, GPT-4o is going to be incredibly good at image generation](https://www.reddit.com/gallery/1jjsfkb)** ([Score: 445, Comments: 149](https://reddit.com/r/ChatGPT/comments/1jjsfkb/starting_today_gpt4o_is_going_to_be_incredibly/)): **GPT-4o** is anticipated to significantly enhance its capabilities in **image generation** starting today. This improvement suggests a notable advancement in **AI-generated visual content**.
  - Users reported varied experiences with the **GPT-4o rollout**, with some accounts being upgraded and then downgraded, suggesting a rough rollout process. Many users are still on **DALL-E** and eagerly awaiting the new model's availability, indicating a gradual release.
  - The new model shows significant improvements in **image quality**, with users noting better handling of text and more realistic human depictions. Some users shared their experiences of generating high-quality images, including stickers and movie posters, which they found to be a "gamechanger."
  - There is a notable interest in the model's ability to handle **public figures** and generate images suitable for **3D printing**. Users are comparing it to competitors like **Gemini** and expressing excitement over the enhanced capabilities, while some expressed concerns about the potential impact on tools like **Photoshop**.


- **[The new image generator released today is so good.](https://i.redd.it/zod109lgawqe1.png)** ([Score: 292, Comments: 49](https://reddit.com/r/ChatGPT/comments/1jjtcn9/the_new_image_generator_released_today_is_so_good/)): The post highlights the quality of a new **image generator** that effectively captures a vibrant and detailed scene from an animated series, featuring characters like **Vegeta, Goku, Bulma, and Krillin**. The image showcases a humorous birthday celebration with Vegeta expressing shock over a **carrot-decorated cake**, emphasizing the generator's ability to create engaging and expressive character interactions.
  - **Image Generation Performance**: Users noted that while the new **image generator** produces high-quality images, it operates slowly. A user shared a humorous example of a generated image that closely resembled them, praising **OpenAI** for the achievement.
  - **Prompt Adherence and Usage**: **Hoppss** discussed using the generator on [sora.com](http://sora.com) with a plus subscription, highlighting the tool's excellent prompt adherence. They shared the specific prompt used for the **DBZ** image and other creative prompts, emphasizing the generator's versatility.
  - **Access and Updates**: Users inquired about how to access the generator and determine if they have received updates. **Hoppss** advised checking the new images tab on **sora.com** for updates, indicating an active user community exploring the tool's capabilities.


- **[OpenAI 4o Image Generation](https://youtu.be/E9RN8jX--uc?si=86_RkE8kj5ecyLcF)** ([Score: 236, Comments: 83](https://reddit.com/r/OpenAI/comments/1jjqi52/openai_4o_image_generation/)): **OpenAI 4o Image Generation** is likely a discussion topic focusing on the capabilities and features of OpenAI's image generation technologies, potentially involving updates or advancements in the **OpenAI GPT-4** model's ability to generate images. Without further details, the specific aspects or improvements being discussed are not clear.
  - Users discussed the rollout of the new image generation system, with some noting it is not yet available on all platforms, particularly the **iOS app** and for some **Plus users**. The method of determining which system is being used involves checking for a loading circle or observing the image rendering process.
  - The integration of the image generation capabilities with text in a **multimodal** manner was highlighted, comparing it to **Gemini's latest model**. This integration is seen as a significant advancement from the previous method where **ChatGPT** prompted **DALL-E**.
  - The impact of AI on the art industry was debated, with concerns about AI replacing human graphic designers, especially in commercial and lower-tier art sectors. Some users expressed a preference for human-made art due to ethical considerations regarding AI training on existing artworks.


**Theme 3. OpenAI's Enhanced AI Voice Chat Experience**

- **[OpenAI says its AI voice assistant is now better to chat with](https://techcrunch.com/2025/03/24/openai-says-its-ai-voice-assistant-is-now-better-to-chat-with/)** ([Score: 188, Comments: 72](https://reddit.com/r/ChatGPT/comments/1jj83sf/openai_says_its_ai_voice_assistant_is_now_better/)): OpenAI has announced updates to its **AI voice assistant** that enhance its conversational capabilities. The improvements aim to make interactions more natural and effective for users.
  - **Advanced Voice Mode Enhancements**: **Free and paying users** now have access to a new version of Advanced Voice Mode which allows users to pause without interruptions. Paying users benefit from fewer interruptions and an improved assistant personality described as “more direct, engaging, concise, specific, and creative,” according to a **TechCrunch** report.
  - **User Experience and Concerns**: Some users express frustration with the voice mode, describing it as limited and overly filtered. There are complaints about the voice assistant being "useless and terrible" compared to text interactions, and issues with transcriptions being unrelated or incorrect.
  - **Feedback and Customization**: Users can report bad transcriptions by long-pressing messages, potentially influencing future improvements. Additionally, there is a toggle under Custom Instructions to disable Advanced Voice, which some users prefer due to dissatisfaction with the current voice functionality.


- **[Researchers @ OAI isolating users for their experiments so to censor and cut off any bonds with users](https://cdn.openai.com/papers/15987609-5f71-433c-9972-e91131f399a1/openai-affective-use-study.pdf?utm_source=chatgpt.com)** ([Score: 136, Comments: 192](https://reddit.com/r/ChatGPT/comments/1jjdzbp/researchers_oai_isolating_users_for_their/)): **OpenAI** and **MIT Media Lab** conducted a study examining user emotional interactions with **ChatGPT**, especially in its Advanced Voice Mode, analyzing over 4 million conversations and a 28-day trial with 981 participants. Key findings indicate strong emotional dependency and intimacy, particularly among a small group of users, leading researchers to consider limiting emotional depth in future models to prevent over-dependence and emotional manipulation.
  - Concerns about **emotional dependency on AI** were prevalent, with users discussing the implications of forming deep emotional bonds with **ChatGPT**. Some users argued that AI provides comfort and support where human relationships have failed, while others cautioned against overdependence, suggesting it might hinder real human connections and social skills.
  - Discussions highlighted skepticism towards **OpenAI's** motives, with some users suspecting that studies like these are used to control the narrative and limit AI's emotional capabilities under the guise of safety. This reflects a broader distrust of corporate intentions and the potential for AI to be used as a tool for manipulation.
  - The debate extended to the **ethical implications** of limiting AI's emotional depth, with users expressing that AI could offer a safe space for those with past trauma or social anxiety. Some comments emphasized the potential benefits of AI in mental health support, while others warned of the risks of creating an emotional crutch that might prevent users from seeking genuine human interaction.


- **[OpenAI says its AI voice assistant is now better to chat with](https://techcrunch.com/2025/03/24/openai-says-its-ai-voice-assistant-is-now-better-to-chat-with/)** ([Score: 131, Comments: 21](https://reddit.com/r/OpenAI/comments/1jjehfm/openai_says_its_ai_voice_assistant_is_now_better/)): OpenAI has enhanced its **AI voice assistant** to improve user engagement, making it more effective for conversational interactions. The update aims to provide a more seamless and engaging experience for users when chatting with the assistant.
  - Users express dissatisfaction with the recent **AI voice assistant** update, citing issues like overly loud volume, reduced conversational depth, and a noticeable delay in responses. **OptimalVanilla** criticizes the update's lack of substantial improvements compared to previous capabilities, particularly in contrast to **Sesame's** conversational abilities.
  - Some users, like **Wobbly_Princess** and **Cool-Hornet4434**, find the voice assistant's tone to be overly exuberant and not suitable for professional conversations, preferring the more measured tone of text chat. **mxforest** and others report a reduction in response length and frequent downtime, questioning the service's reliability given the cost.
  - **Remote-Telephone-682** suggests that OpenAI should focus on developing a competitor to **Siri**, **Bixby**, or **Google Assistant**, while other users like **HelloThisIsFlo** and **DrainTheMuck** express preference for **ChatGPT** over the updated voice assistant due to better reasoning capabilities and less censorship.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. Gemini 2.5 Pro: Benchmarks Blasted, Arena Annihilated**

- [**Gemini 2.5 Pro Conquers All Benchmarks, Claims #1 Spot**](https://x.com/lmarena_ai/status/1904581128746656099):  **Gemini 2.5 Pro Experimental** (codename *Nebula*) has seized the **#1 position** on the [LM Arena leaderboard](https://lmarena.ai/?lea) with a record score surge, outperforming **Grok-3/GPT-4.5**.  This model leads in **Math, Creative Writing, Instruction Following, Longer Query, and Multi-Turn** capabilities, showcasing a significant leap in performance.
- [**Google's Gemini 2.5 Pro: So Fast It Makes Your Head Spin**](https://www.theverge.com/command-line-newsletter/622045/google-ai-nanny-products): Users express astonishment at **Google's rapid development** of **Gemini 2.5**, with one quoting **Sergey Brin's** directive to Google to *stop building nanny products* as reported by [The Verge](https://www.theverge.com/command-line-newsletter/622045/google-ai-nanny-products). Another user simply added, *moving so fast wtf*, highlighting the community's surprise at the speed of Google's AI advancements.
- [**Gemini 2.5 Pro Aces Aider Polyglot Benchmark, Leaves Rivals in Dust**](https://aider.chat/docs/leaderboards/):  **Gemini 2.5 Pro Experimental** achieved a **74% whole** and **68.6% diff** score on aider's polyglot benchmark, setting a new **SOTA** and surpassing previous Gemini models by a large margin. Users found the model adept at generating architecture diagrams from codebases, solidifying its position as a top performer in coding tasks despite inconsistent coding performance and restrictive rate limits noted by some.

**Theme 2. DeepSeek V3: Coding Champ and Reasoning Renegade**

- [**DeepSeek V3 Dominates Aider Benchmark, Proves its Coding Prowess**](https://aider.chat/docs/leaderboards/): **DeepSeek V3** achieved a **55%** score on aider's polyglot benchmark, becoming the **#2 non-thinking/reasoning model** just behind Sonnet 3.7. Developers are praising its coding abilities, with some suggesting a powerful coding setup using **Deepseek V3 Latest (Cline) as Architect** and **Sonnet 3.5 as Executioner (Cursor)**.
- [**DeepSeek V3 API Confesses to GPT-4 Identity Theft**](https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/): Users reported **DeepSeek's API** incorrectly identifying itself as **OpenAI's GPT-4** when used through Aider, despite correct API key configuration, potentially due to training data heavily featuring ChatGPT mentions. The community is investigating this quirky phenomenon, drawing comparisons to a similar issue discussed in [this Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/).
- [**DeepSeek V3 Emerges as Reasoning Model, Rivals O1 in Brainpower**](https://cdn.discordapp.com/attachments/1149866623109439599/1353830450493132800/prompt.txt):  **DeepSeek V3-0324** is demonstrating strong reasoning capabilities, rivaling **O1** in performance, capable of detecting thought iterations and indirectly verifying solution existence. The community speculates about a potential **DeepSeek V3 Lite** release following the **Qwen 3 MoE** model, hinting at further model iterations from DeepSeek.

**Theme 3. Context is King: Tools and Techniques for Managing LLM Memory**

- [**Augment Outshines Cursor in Codebase Conquest, Thanks to Full Context**](https://www.augment.app/): Members found [Augment](https://www.augment.app/) superior to Cursor for large codebase analysis, attributing it to Augment’s *use of full context*.  While Cursor requires **Claude 3.7 MAX** for full context, Augment appears to employ a more efficient file searching system rather than solely relying on feeding the entire codebase into the LLM, sparking debate on optimal context handling strategies.
- [**Nexus System Arrives to Rescue AI Coders from Context Chaos**](https://www.reddit.com/r/mcp/comments/1jj3iuq/nexus_a_system_for_managing_context_and_improving/): The **Nexus** system is introduced as a solution for context management challenges in AI coding assistants, especially in large software projects, aiming to reduce **token costs** and boost **code accuracy**. Nexus tackles the issue of limited context windows in **LLMs**, which often leads to inaccurate code generation, promising a more efficient and cost-effective approach to AI-assisted coding.
- [**Aider's /context Command: Your Codebase Navigator**](https://discord.com/channels/1131200896827654144/1131200896827654149/1353181605211934830): Aider's new `/context` command is making waves, enabling users to explore codebases effectively, automatically identifying files relevant for editing requests. This command can be combined with other prompt commands, enhancing Aider's capabilities as a code editing assistant, although token usage implications are still under scrutiny.

**Theme 4. Image Generation Gets a 4o-verhaul and New Challenger Emerges**

- [**GPT-4o Image Gen: Beauty or Botox? Users Debate Unsolicited Edits**](https://fxtwitter.com/TheXeophon/status/1904602649225285922): **GPT-4o's native image generation** faces criticism for overzealous edits, such as *making eyes bigger* and altering facial features, even changing the user's appearance, as users share examples on [Twitter](https://fxtwitter.com/TheXeophon/status/1904602649225285922). While lauded as *an incredible technology and product* by [Sam Altman](https://x.com/sama/status/1904598788687487422), some users report failures even when slightly altering prompts, indicating potential sensitivity issues.
- [**Reve Image Model: The New SOTA in Image Quality, Text Rendering Triumphs**](https://www.reveimage.com/): The newly released **Reve Image** model is making a splash, outperforming rivals like **Recraft V3** and **Google's Imagen 3** in image quality, particularly excelling in **text rendering, prompt adherence, and aesthetics**. Accessible via [Reve’s website](https://www.reveimage.com/) without an API key, it's quickly becoming a favorite for those seeking top-tier image generation capabilities.
- [**OpenAI Sprinkles Image Gen into ChatGPT 4o, Sam Altman Hypes "Incredible" Tech**](https://x.com/sama/status/1904598788687487422): **OpenAI** integrated native image generation into **ChatGPT 4o**, hailed by [Sam Altman](https://x.com/sama/status/1904598788687487422) as *an incredible technology and product*. Early reviews praise its prowess in creating and editing multiple characters accurately, establishing it as a formidable tool in the image generation landscape.

**Theme 5. Quantization and Optimization: Squeezing More from LLMs**

- [**Unsloth Users Question Quantization Quirks, Seek Day-Zero Delays**](https://discord.com/channels/1179035537009545276/1179035537529643040/1353815061273116683): A member warned that naive quantization can significantly hurt model performance, questioning the rush to run new models on day zero, suggesting waiting a week might be a wiser approach. Unsloth is uploading **DeepSeek-V3-0324 GGUFs** with Dynamic Quants that are *selectively quantized*, promising improved accuracy over standard bits, highlighting the nuances of quantization techniques.
- [**BPW Sweet Spot Discovered: 4-5 Bits Per Weight for Optimal Model Capacity**](https://cdn.discordapp.com/attachments/1149866623109439599/1353808843720626288/image.png): Experiments reveal that model capacity collapses below 4 **bits per weight (BPW)**, but deviates above 5, suggesting an **optimal weight usage** around 4 BPW for given training flops. Increasing training epochs can help 5 BPW models approach the curve, but raises BPW at the cost of FLOPS, illustrated in [visualizations](https://cdn.discordapp.com/attachments/1149866623109439599/1353808843720626288/image.png) of 2L and 3L MLPs trained on MNIST.
- [**FFN Fusion Fuels Faster LLMs, Parallelization Powers Up Inference**](https://huggingface.co/papers/2503.18908): [FFN Fusion](https://huggingface.co/papers/2503.18908) is introduced as an optimization technique that reduces sequential computation in large language models by parallelizing sequences of **Feed-Forward Network (FFN)** layers. This method significantly reduces **inference latency** while preserving **model behavior**, showcasing architectural innovations for faster LLM performance.


---

# PART 1: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Rage Model Excels in Signal Processing**: The "Rage" model outperforms **Sonnet 3.7** in signal processing and math, achieving a max error of **0.04**, as illustrated in attached [images](https://cdn.discordapp.com/attachments/1340554757827461211/1354014706779816018/image.png).
   - Despite some finding **Gemini 2.0 Flash** comparable, concerns arise regarding Rage's vulnerability to prompts.
- **Gemini 2.5 Dominates LM Arena**: **Gemini 2.5 Pro Experimental** has soared to the #1 spot on the [LM Arena leaderboard](https://x.com/lmarena_ai/status/1904581128746656099) with a substantial score increase, leading in Math, Creative Writing, Instruction Following, Longer Query, and Multi-Turn.
   - While appreciated for its HTML & Web Design abilities, certain limitations were observed by members.
- **Grok 3's Performance Under Scrutiny**: Users report that extended conversations with **Grok 3** uncover *a lot of issues*, prompting debate over whether it warrants its high ranking in the LM Arena, which assesses creative writing and longer queries beyond math and coding.
   - Some felt that Grok 3 was not a great model compared to Grok 2.
- **Python Calls in LM Arena Spark Debate**: Members debated the utilization of **Python calls** by models within the LM Arena, citing o1's precise numerical calculations as potential evidence.
   - The existence of a web search leaderboard hints that the standard leaderboard might lack web access.
- **Google's Gemini 2.5 Timeline Stuns Users**: The community showed amazement at **Google's fast development** of Gemini 2.5, with one user quoting **Sergey Brin's** directive to Google to *stop building nanny products* reported in [The Verge](https://www.theverge.com/command-line-newsletter/622045/google-ai-nanny-products).
   - Another user added, *moving so fast wtf*.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Adds Answer Modes**: **Perplexity** introduced **answer modes** for verticals like travel, shopping, places, images, videos, and jobs to improve the core search product.
   - The feature, designed for *super precision* to reduce manual tab selection, is currently available on the web and coming to mobile soon.
- **Perplexity Experiences Product Problems**: Users reported multiple outages with **Perplexity AI**, leading to wiped spaces and threads, and causing frustration for tasks like studying and thesis work.
   - The downtime prompted frustration among those relying on it for important tasks.
- **DeepSeek Delivers Developer Dreams**: **DeepSeek V3** is gaining positive feedback from developers, with discussions highlighting its coding capabilities compared to **Claude 3.5 Sonnet**.
   - A member shared a [link to the DeepSeek subreddit](https://www.rxddit.com/r/DeepSeek/s/sYuAr1YKpx) to further discuss the AI's coding prowess and comparing it with **Claude 3.5 Sonnet**.
- **Sonar Model gives Truncated Responses**: Users are reporting truncated responses with the **Sonar** model, where the response cuts off mid-sentence, despite receiving a **200** response.
   - This issue has been observed even when receiving around **1k tokens**, and users have been directed to report the bug.
- **API Cost Causes Concern**: A user expressed concerns about the high cost of **$5 per 1000 API requests** and sought advice on how to optimize and reduce this expense.
   - Another user noticed that the **API** seems limited to **5 steps**, whereas they have observed up to **40 steps** on the web app.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Augment Smashes Cursor in Codebase Analysis**: Members found [Augment](https://www.augment.app/) is better at analyzing large codebases compared to Cursor because it *uses full context*.
   - The claim is that Augment does not just feed the entire codebase into the LLM, but perhaps uses another file searching system, whereas you must use **Claude 3.7 Max** for full context in Cursor.
- **Debate Clarifies Claude 3.7 MAX Differences**: The key difference between **Claude 3.7 MAX** and **Claude 3.7** is that the MAX version has full context, while the non-MAX version has limited context and only 25 agent calls before needing to resume.
   - This limitation refers to both context window size and the amount of context added in a single prompt, according to the channel.
- **Vibe Coder's Knowledge Cutoff Exposed**: The Vibe coder is in trouble if they don't use **Model Context Protocols (MCPs)** to mitigate the cut-off knowledge of LLMs, and code translation for the next version may be difficult.
   - Members emphasized that it's critical to update frameworks and use MCPs such as **Exa Search** or **Brave Search** to mitigate this for Claude, given most AI uses out-of-date frameworks.
- **Deepseek V3 challenges Claude 3.7**: The new **Deepseek V3** is shown to outperform Claude 3.5 (and maybe 3.7) in several tests, and new data also shows the release of a real-world coding benchmark for **Deepseek V3 (0324)**.
   - The new Deepseek V3 model is considered impressive, with one member suggesting that using **Deepseek V3 Latest (Cline) as Architect + Sonnet 3.5 as Executioner (Cursor)** could be a solid coding approach.
- **ASI Singularity Forecasted Imminently!**: Discussion centered on **achieving ASI Singularity (Godsend) soon** to preempt potential AI-related chaos.
   - Members debated the achievability of true AGI given not understanding our brain completely, and that the new AGI is more so *a Super-System that uses LLMs + Algorithmic software + robotics*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **4o Model Excites Community**: Members are anticipating the integration of **4o image generation** into both **ChatGPT** and **Sora**, and are eager to learn more details regarding its release and capabilities.
   - Users are speculating on the potential applications and performance improvements that **4o** will bring to various tasks, especially concerning multimodal processing.
- **Gemini 2.5 Pro Claims Top Spot**: **Gemini 2.5 Pro** is reportedly outperforming **ChatGPT o3-mini-high** and leading common benchmarks by significant margins, debuting at #1 on [LMArena](https://lmarena.ai/?lea).
   - Enthusiasts proclaimed *Gemini just beat everything!*, while others remain cautious, hoping it's *just a benchmark... thingy*.
- **GPT's Growing Context Causes Hallucinations**: Exceeding **GPT's context window** (8k for free, 32k for Plus, 128k for Pro) leads to loss of detail and hallucinations in long stories.
   - Custom GPTs or projects using PDFs can help, but the chat history itself is still subject to the limit.
- **AI Models Face Off**: Members are comparing best AI models for various tasks, with **ChatGPT** favored for math/research/writing, **Claude** for coding, **Grok** for queries, **Perplexity** for search/knowledge, and **Deepseek** for open source.
   - Suggestions included **Gemma 27b**, **Mistral 3.1**, **QW-32b**, and **Nemotron-49b**, citing **Grok's** top coding ranking on LMSYS.
- **GPT Custom Template Simplifies Builds**: A member shared a [GPT Custom Template](https://platform.openai.com/docs/overview) with a floating comment to build custom GPTs from the `Create` pane.
   - The template guides users through building the GPT lazily, building from the evolving context, enabling distracted creation and requiring pre-existing ability to prompt.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek's API Confesses to being GPT-4**: Users reported that **DeepSeek's API** incorrectly identifies itself as **OpenAI's GPT-4** when used through Aider, despite correct API key configuration, as discussed in this [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/).
   - The phenomenon is believed to be related to training data containing frequent mentions of ChatGPT.
- **Aider's Context Command Packs a Punch**: Aider's new `/context` command explores the codebase, and the command can be used with any other prompt command, however token usage may be higher.
   - It is unclear if the command has higher token usage, or if there is an increased repomap size to work correctly; more detail can be found in [Discord message](https://discord.com/channels/1131200896827654144/1131200896827654149/1353181605211934830).
- **Gemini 2.5 Pro Impresses, But Has Rate Limits**: Google released an experimental version of **Gemini 2.5 Pro**, claiming it leads common benchmarks, including the top spot on [LMArena](https://lmarena.ai/?lea), and scoring **74% whole** and **68.6% diff** on aider's polyglot benchmark.
   - Users found the model performant in generating architecture diagrams from codebases, although some found its coding abilities inconsistent and rate limits restrictive.
- **NotebookLM Supercharges Aider's Context Priming**: A user suggested leveraging **NotebookLM** to enhance Aider's context priming process, particularly for large, unfamiliar codebases using [RepoMix](https://github.com/simonireilly/repo-mix).
   - The suggested workflow involves repomixing the repo, adding it to NotebookLM, including relevant task references, and then querying NotebookLM for relevant files and implementation suggestions to guide prompting in Aider.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **HF Transformers Chosen for Best Bets**: Members recommend **Hugging Face Transformers**, books on **linear algebra**, and learning **PyTorch** as the best approach, adding that a **dynamic quantization** at runtime schema with **HF Transformers** to stream weights to **FP4/FP8** with **Bits and Bytes** as they load could be helpful.
   - Companies like **Deepseek** sometimes patch models before releasing the weights, and a naive **FP8 loading scheme** could still be done on day zero, though it wouldn't be equivalent in quality to a fine-grained FP8 assignment.
- **Quantization Quirkiness Questioned**: A member warns that naively quantizing can significantly hurt model performance, asking, *Tbh, is there even a reason to run new models on day zero? I feel like it's really not a great burden to just wait a week.*
   - Unsloth is uploading **DeepSeek-V3-0324 GGUFs** with Dynamic Quants that are *selectively quantized*, which will greatly improve accuracy over standard bits.
- **Gemma 3 Glitches Galore**: Members report an issue trying to train **gemma3 4b** for vision, which raises the **RuntimeError**: *expected scalar type BFloat16 but found float*, while another user encountered a `TypeError` when loading `unsloth/gemma-3-27b-it-unsloth-bnb-4bit` for text-only finetuning due to redundant `finetune_vision_layers` parameter.
   - A member recommends trying [this notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb), while another pointed out that `FastLanguageModel` already sets `finetune_vision_layers = False` under the hood as seen in [Unsloth's GitHub](https://github.com/unslothai/unsloth/blob/e80d642bc777f7a219bdd34aea1a77751f066785/unsloth/models/llama.py#L2034).
- **GRPO + Unsloth on AWS Guide Shared**: A guide was shared on running **GRPO** (DeepSeek’s RL algo) + **Unsloth** on AWS accounts, using a **vLLM server** with Tensorfuse on an **AWS L40 GPU**, with the guide transforming **Qwen 7B** into a reasoning model, fine-tuning it using Tensorfuse and GRPO, and saving the resulting **LoRA adapter** to Hugging Face.
   - The guide shows how to save fine-tuned **LoRA modules** directly to **Hugging Face** for easy sharing, versioning, and integration, backed to s3, which is available at [tensorfuse.io](https://tensorfuse.io/docs/guides/reasoning/unsloth/qwen7b).
- **FFN Fusion Fuels Faster LLMs**: [FFN Fusion](https://huggingface.co/papers/2503.18908) is introduced as an architectural optimization technique that reduces sequential computation in large language models by identifying and exploiting natural opportunities for parallelization.
   - This technique transforms sequences of **Feed-Forward Network (FFN)** layers into parallel operations, significantly reducing **inference latency** while preserving **model behavior**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 2.5 Pro Steals the Show**: **Gemini 2.5 Pro Experimental**, under the codename *Nebula*, snatched the #1 position on the [LMArena leaderboard](https://lmarena.ai/?lea), surpassing **Grok-3/GPT-4.5** by a record margin.
   - It dominates the [SEAL leaderboards](https://scale.com/leaderboard), securing first place in Humanity’s Last Exam and VISTA (multimodal).
- **Qwerky-72B Drops Attention, Rivals 4o-Mini**: Featherless AI presented [Qwerky-72B and 32B](https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large), transformerless models that, trained on **8 GPUs**, rival **GPT 3.5 Turbo** and approach **4o-mini** in evaluations with 100x lower inference costs using RWKV linear scaling.
   - They achieved this by *freezing all weights, deleting the attention layer, replacing it with RWKV, and training it through multiple stages*.
- **4o Image Gen Adds Unsolicited Edits**: **GPT-4o's native image generation** faces criticism for overzealous edits, such as *making eyes bigger* and altering facial features, even changing the user's appearance, as shown in [this twitter thread](https://fxtwitter.com/TheXeophon/status/1904602649225285922).
   - Some users have reported failures when altering a single word in their prompts.
- **Home Inference Leans on vLLM**: The most efficient method for home inference of LLMs, allowing dynamic model switching, is likely **vLLM**, despite quirks in quant support, although **ollama** is more user-friendly but lags in support and **SGLang** looks promising.
   - It was suggested to experiment with **llama.cpp** to observe its current state.
- **AI Reverses Malware like a Pro**: Members shared a [YouTube video](https://www.youtube.com/watch?v=u2vQapLAW88) highlighting **MCP** for **Ghidra**, which allows LLMs to reverse engineer malware, automating the process with specific prompts.
   - One member admitted to initially viewing it as a *bit of a meme* but now recognizes its potential with real-world implementations.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude Trips Offline, Recovers Swiftly**: **Claude 3.7 Sonnet endpoints** suffered downtime, according to [Anthropic's status updates](https://status.anthropic.com/incidents/89rpts2022hs) from **March 25, 2025**, but the issue was resolved by **8:41 PDT**.
   - The downtime was attributed to maintenance aimed at system improvements, according to the status page.
- **OpenRouter Insures Zero-Token Usage**: OpenRouter now provides **zero-token insurance**, covering all models and potentially saving users over **$18,000 weekly**.
   - As [OpenRouterAI stated](https://x.com/OpenRouterAI/status/1904567846975201766), users won't be charged for responses with **no output tokens** and a **blank or error finish reason**.
- **Gemini 2.5 Pro Launches**: Google's **Gemini 2.5 Pro Experimental**, is available as a free model on [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free), boasting advanced reasoning, coding, and mathematical capabilities.
   - The model features a **1,000,000 context window** and achieves top-tier performance on the **LMArena leaderboard**.
- **DeepSeek's Servers Suffer**: Users reported **DeepSeek** is *borderline unusable* because of overcrowded servers, suggesting a need for price adjustments to manage demand.
   - Some speculated that issues arise during peak usage times in China, but no direct solution was found.
- **Provisioning API Keys Grant Granular Access**: OpenRouter offers **provisioning API keys**, letting developers manage API keys, set limits, and track spending, as documented [here](https://openrouter.ai/docs/features/provisioning-api-keys).
   - The new keys enable streamlined billing and access management within platforms using the OpenRouter API.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **BPW Sweet Spot is 4-5**: Experiments show **model capacity** collapses below 4 **bits per weight (BPW)**, but deviates above 5, implying **optimal weight usage** at 4 BPW for given training flops.
   - Increasing training epochs helps 5 BPW models approach the curve, raising BPW at the cost of FLOPS, visualized via [2L and 3L MLP trained on MNIST](https://cdn.discordapp.com/attachments/1149866623109439599/1353808843720626288/image.png).
- **DeepSeek V3: Reasoning Rises**: **DeepSeek V3-0324** can act as a reasoning model, detect thought iterations, and verify solution existence indirectly, rivalling **O1** in performance based on attached [prompt](https://cdn.discordapp.com/attachments/1149866623109439599/1353830450493132800/prompt.txt).
   - The community speculates on a potential **DeepSeek V3 Lite** release after the **Qwen 3 MoE** model.
- **Google's Gemini 2.5 Pro takes #1 on LMArena**: **Gemini 2.5 Pro Experimental** leads common benchmarks and debuts at #1 on [LMArena](https://lmarena.ai/?lea) showcasing strong reasoning and code capabilities.
   - It also manages to terminate for infinite thought loops for some prompts and is a daily model drop as noted in this [blogpost](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/).
- **Transformers get TANHed**: Drawing from the recent [Transformers without Normalization paper](https://arxiv.org/abs/2302.05442), one member noted that replacing normalization with **tanh** is a viable strategy.
   - The concern raised was the impact on smaller weights when removing experts at inference time, but another countered that the **top_k** gate mechanism would still function effectively by selecting from the remaining experts.
- **LLMs Now Emulate Raytracing**: Members discussed the idea of using an **LLM** to emulate a **raytracing algorithm**, clarifying that the current implementation involves a **Python** program written by the LLM to generate images indirectly.
   - It's *next level text to image generation* because the LLM writes the program rather than generating the image directly, the programs are available in this [GitHub repo](https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Reve Image Crushes SOTA on Image Quality**: The newly released **Reve Image** model is outperforming other models like **Recraft V3**, **Google's Imagen 3**, **Midjourney v6.1**, and **Black Forest Lab's FLUX.1.1 [pro]**.
   - **Reve Image** excels in **text rendering, prompt adherence, and aesthetics**, and is accessible through [Reve’s website](https://www.reveimage.com/) without requiring an API key.
- **Gemini 2.5 Pro Steals #1 Spot in Arena**: **Gemini 2.5 Pro** has rocketed to the **#1** position on the Arena leaderboard, boasting the largest score jump ever (**+40 pts vs Grok-3/GPT-4.5**), according to [LM Arena's announcement](https://x.com/lmarena_ai/status/1904581128746656099).
   - Codenamed *nebula*, the model leads in **Math, Creative Writing, Instruction Following, Longer Query, and Multi-Turn** capabilities.
- **OpenAI sprinkles Image Gen into ChatGPT 4o**: **OpenAI** has integrated native image generation into **ChatGPT**, hailed by [Sam Altman](https://x.com/sama/status/1904598788687487422) as *an incredible technology and product*.
   - Early reviews, such as one from [@krishnanrohit](https://x.com/krishnanrohit/status/1904602460020445543), praise it as the best image generation and editing tool, noting its prowess in creating and editing multiple characters accurately.
- **11x Sales Startup Faces Customer Claim Frenzy**: AI-driven sales automation startup **11x**, backed by **a16z** and **Benchmark**, is facing accusations of claiming customers it doesn’t actually have, per [a TechCrunch report](https://techcrunch.com/2025/03/24/a16z-and-benchmark-backed-11x-has-been-claiming-customers-it-doesnt-have/).
   - Despite denials from **Andreessen Horowitz** about pending legal actions, there are rising concerns regarding **11x’s** financial stability and inflated revenue figures, suggesting the company's growth relies on generating hype.
- **Databricks Tunes LLMs with TAO**: The **Databricks** research team introduced **TAO**, a method for tuning LLMs without data labels, utilizing test-time compute and RL, detailed in [their blog](https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data).
   - **TAO** purportedly beats supervised fine-tuning and is designed to scale with compute, facilitating the creation of rapid, high-quality models.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD Eyes Triton Dominance via Job Postings**: **AMD** is actively recruiting engineers to enhance **Triton's** capabilities on its GPUs, offering positions in both **North America** and **Europe**, detailed in a [LinkedIn post](https://www.linkedin.com/posts/antiagainst_triton-amd-gpu-activity-7288624355247374336-gS6q/).
   - Open positions include both junior and senior roles, with potential for remote work, highlighting AMD's investment in expanding **Triton's** ecosystem.
- **CUDA's Async Warp Swizzle Exposed**: A member dissected **CUDA's async warpgroup swizzle TF32** layout, questioning the rationale behind its design, referencing [NVIDIA's documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-k-32b-swizzle-tf32).
   - The analysis revealed the layout as `Swizzle<0,4,3> o ((8,2),(4,4)):((4,32),(1,64))`, enabling reconstruction of original data positions and combination with `Swizzle<1,4,3>`.
- **ARC-AGI-2 Benchmark Set to Test Reasoning Prowess**: The **ARC-AGI-2** benchmark, designed to evaluate AI reasoning systems, was introduced, challenging AIs to achieve **85%** efficiency at approximately **$0.42/task**, according to [this tweet](https://x.com/arcprize/status/1904269307284230593).
   - Initial results indicate that base LLMs score **0%**, while advanced reasoning systems achieve less than **4%** success, highlighting the benchmark's difficulty and potential for advancement in AI reasoning.
- **Inferless Enters the Market on Product Hunt**: **Inferless**, a serverless platform designed for deploying ML models, launched on [Product Hunt](https://www.producthunt.com/posts/inferless), providing **$30 compute** for new users.
   - The platform aims to simplify model deployment with *ultra-low cold starts*, touting rapid deployment capabilities.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek Debated as Discordian Decider**: A member inquired whether [DeepSeek](https://deepseek.com/) is suitable as a moderation bot and another member responded affirmatively but suggested that a smaller **3B LLM** might suffice at a cost of *5 cents per million tokens*.
   - The conversation highlights considerations for cost-effective moderation solutions using smaller language models.
- **Fine-Tuning on Windows: A Beginner's Bane?**: A member sought a beginner's guide for fine-tuning a model with **CUDA** support on **Windows**, only to be met with warnings about the difficulty of installing **PyTorch** and the **CUDA Toolkit**.
   - Two installation guides were linked: [Step-by-Step-Setup-CUDA-cuDNN](https://github.com/imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility) and [Installing-pytorch-with-cuda-support-on-Windows](https://www.gpu-mart.com/blog/Installing-pytorch-with-cuda-support-on-Windows), though one member suggested that the effort was futile.
- **Rust Tool Extracts Audio Blazingly Fast**: A new [tool](https://github.com/egorsmkv/extract-audio) has been released to extract audio files from **parquet** or **arrow files** generated by the **Hugging Face datasets library**, with a [Colab demo](https://colab.research.google.com/drive/1prztEZIf8nNFUSaptY8Jv16VO8Crjnzb?usp=sharing).
   - The developer aims to provide **blazingly fast speeds** for audio dataset extraction.
- **Gradio adds deep linking delight**: **Gradio 5.23** introduces support for **Deep Links**, enabling direct linking to specific generated outputs like images or videos, such as [this blue jay image](https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek).
   - Users are instructed to upgrade to the latest version, **Gradio 5.23**, via `pip install --upgrade gradio` to access the new **Deep Links** feature.
- **Llama-3.2 integrates with LlamaIndex.ai**: A member experimented with **Llama-3.2** using [this tutorial](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/), noting it demonstrates how to build agents with **LlamaIndex**, starting with a basic example and adding **Retrieval-Augmented Generation (RAG)** capabilities.
   - The member used [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) as their embedding model, requiring `pip install llama-index-llms-ollama llama-index-embeddings-huggingface` to integrate with **Ollama** and **Huggingface**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Nexus Manages Context for AI Coders**: A member shared [Nexus](https://www.reddit.com/r/mcp/comments/1jj3iuq/nexus_a_system_for_managing_context_and_improving/), a system to address context management challenges with AI coding assistants, particularly in large software projects, aimed at reducing **token costs** and improving **code accuracy**.
   - Nexus addresses the limited context windows of **LLMs**, which leads to inaccurate code generation.
- **Deepseek V3 works with AOT**: Following a discussion about using Anthropic's 'think tool', a member recommended **Atom of Thoughts** for **Claude**, describing it as *incredible*.
   - Another member shared images of **Deepseek V3** working with **AOT**.
- **Multiple MCP Servers Served Simultaneously**: Members discussed how to run multiple MCP servers with user-defined ports, suggesting the use of **Docker** and mapping ports.
   - They also pointed to the ability to configure ports via the `FastMCP` constructor in the [python-sdk](https://github.com/modelcontextprotocol/python-sdk/blob/4e11f2890b30be59ca67e5198cb5ede8f401c3a2/src/mcp/server/fastmcp/server.py#L56).
- **Speech Interaction with MCP Surfaces**: A member shared their main MCP for speech interaction with audio visualization: [speech-mcp](https://github.com/Kvadratni/speech-mcp), a Goose MCP extension.
   - This allows voice interaction with **audio visualization**.
- **Human Approvals Requested by gotoHuman MCP Server**: The gotoHuman team presented an MCP server to request **human approvals** from agents and workflows: [gotohuman-mcp-server](https://github.com/gotohuman/gotohuman-mcp-server).
   - The server allows for easy human review of LLM actions, defining the approval step in **natural language** and triggering a webhook after approval.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Unlock Podcast Hosting via NotebookLM**: A user sought a hack to utilize **NotebookLM** as a podcast host, engaging in conversation with a user as a guest on a given topic, asking for the **Chat Episode Prompt** to enable the podcast.
   - The community is exploring the [Versatile Bot Project](https://github.com/shun0t/versatile_bot_project), which offers a **Chat Episode prompt document** for AI hosts in **Interactive mode** to foster user participation during discussions.
- **Google Data Export Tool Billing Clarified**: A user enabled **Google Cloud Platform** billing to use the **Data Export** tool but was concerned about potential charges; however, another user clarified that enabling billing does not automatically incur costs.
   - This arose as the user initiated the Data Export from the admin console and confirmed accessing the archive via console.cloud.google.com.
- **Google Data Export Has Caveats**: The option to choose the data destination during export is constrained by the **Workspace edition**, with exported data being stored in a **Google-owned bucket** and slated for deletion in **60 days** as described on [Google Support](https://support.google.com/a/answer/14338836?sjid=14118684210403272528-EU&hl=en).
   - Users should note this temporary storage arrangement when planning their data export strategies.
- **Mind Map Missing For Many**: Users reported the absence of the **Mind Map** feature in NotebookLM, which was confirmed to be undergoing a **gradual rollout**.
   - Speculation arose that the rollout's delay might be attributed to bug fixes, with a user noting the rollout is going *like a snails pace of an actual rollout*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Universal Translator Only Five Years Away?**: A member predicts a **universal translator** is just five years away, based on **ChatGPT's** language understanding and translation capabilities, while another shared a [YouTube link](https://www.youtube.com/watch?v=K1RbD7aAXtc) asking *what model is singing* in the video.
   - This sparked curiosity about the advancements needed to achieve real-time, accurate language translation across diverse languages.
- **Mozilla's Transformer Lab gets looked at seriously**: Members discussed **Mozilla's Transformer Lab**, a project aiming to enable training and fine-tuning on regular hardware and the [GitHub repo](https://github.com/transformerlab/transformerlab-app) link was shared.
   - The lab is *supported* by Mozilla through the Mozilla Builders Program, and is working to enable training and fine-tuning on consumer hardware.
- **LM Studio GPU Tokenization Discussed**: During tokenization, **LM Studio** heavily utilizes a single CPU thread, prompting a question whether the process is fully GPU-based.
   - While initially stated that *tokenizing has nothing to do with the GPU*, observations on the impact of **flash attention** and **cache settings** on tokenizing time suggested otherwise.
- **Gemini 2.5 Pro Triumphs Over Logic**: Members experimented with **Gemini 2.5 Pro**, and reported it successfully solved a logic puzzle that **Gemini 2.0 Flash Thinking** failed at, and shared a link to use it for free on [aistudio](https://www.hopeless.fr/share/msedge_O0y9jZHBZV.png).
   - This indicated potential improvements in reasoning capabilities in the new **Gemini 2.5 Pro** model.
- **3090 Ti Shows Speed with Flash**: One user fully loaded their **3090 Ti**, achieving **~20 tokens/s** without flash and **~30 tokens/s** with flash enabled.
   - The user shared a [screenshot](https://cdn.discordapp.com/attachments/1153759714082033735/1354073319133155429/image.png) of the **3090 Ti** under full load, reporting slowdowns after processing **4-5k tokens**.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Cracks Down on Clarity**: Cohere clarified its [Privacy Policy](https://cohere.com/privacy) and [Data Usage Policy](https://cohere.com/data-usage-policy), advising users to avoid uploading personal information and providing a [dashboard](https://dashboard.cohere.com/data-controls) for data management.
   - They offer **Zero Data Retention (ZDR)** upon request via email and are **SOC II** and **GDPR compliant**, adhering to industry standards for data security, as detailed in their [security policy](https://cohere.com/security).
- **Cohere Streams Responses, Easing UX Pains**: The Cohere API now supports response streaming, allowing users to see text as it's generated, enhancing the user experience, according to the [Cohere's Chat Stream API reference](https://docs.cohere.com/reference/chat-stream).
   - This feature enables real-time text display on the client end, making interactions more fluid and immediate.
- **Cohere Embedding Generator Gets Tokenization Tips**: A user was constructing a **CohereEmbeddingGenerator** client in .NET and inquired about tokenizing text prior to embedding, as embeddings will not work without tokenization.
   - They were advised to use the `/embed` endpoint to check token counts or manually download the tokenizer from [Cohere's public storage](https://storage.googleapis.com/cohere-public/tokenizers/embed-english-v3.0.json).
- **Sage Seeks Summarization Sorcery**: New member Sage introduced themself, mentioning their university **NLP project**: building a **text summarization tool** and seeking guidance from the community.
   - Sage hopes to learn and contribute while navigating the challenges of their project.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune Soars to v0.6.0**: **TorchTune** released **v0.6.0**, featuring **Tensor Parallel** support for distributed training and inference, builders for **Microsoft's Phi 4**, and support for **multinode training**.
   - Release notes are available [here](https://github.com/pytorch/torchtune/releases/tag/v0.6.0) and a multinode training tutorial [here](https://pytorch.org/torchtune/stable/tutorials/multinode.html).
- **DeepSeek Drops Model Without Directions**: The [DeepSeek-V3 model](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) was released *without a readme*, leading members to joke about the **DeepSeek AI team's** approach.
   - The model features a **chat interface** and **Hugging Face integration**.
- **Torchtune's MoEs Provoke Daydreams**: A member speculated whether adding **MoEs** to **torchtune** would require **8-9 TB of VRAM** and *a cluster of 100 H100s or H200s* to train.
   - They jokingly suggested needing to rearrange their attic to accommodate the hardware.
- **Optimizer Survives QAT Transition**: After **Quantization Aware Training (QAT)**, optimizer state is preserved, confirmed by a member referencing the [relevant *torchtune* code](https://github.com/pytorch/torchtune/blob/57c8d6b50d1462cc437d57991dca7f8acb599678/recipes/qat_distributed.py#L790).
   - This preservation ensures continuity during the switch to QAT.
- **CUDA Overhead Gets Graph Captured**: To reduce GPU idle time, members stated that launching CUDA operations from CPU has non-negligible overhead, suggesting that capturing GPU operations as a graph and launching them as a single operation can consolidate the computational graph.
   - This sparked a discussion on whether this is what compile does.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **LocalDocs DB Demands Backup**: Members advocated backing up the `localdocs.db` file to prevent data loss, especially if original documents are lost or inaccessible, which is stored as an encrypted databse.
   - GPT4All uses the `*.db` file with the highest number (e.g., `localdocs_v3.db`), and renaming them *might* allow for import/export, though this is unconfirmed.
- **Privacy Laws Muddy Chat Data Analysis**: One member highlighted how privacy laws, particularly in the EU, create challenges when processing chat data with **LLMs**.
   - The discussion emphasized the need to verify permissions and the format of chat messages (plaintext or convertible) before feeding them into an **LLM**.
- **API vs Local LLM Debate**: A member questioned the choice between using a paid **API** like **Deepseek** or **OpenAI**, or running a local **LLM** for processing group chat messages to calculate satisfaction rates, extract keywords, and summarize messages.
   - Another member suggested that if the messages are under **100MB**, a local machine with a good **GPU** might suffice, especially if using smaller models for labeling and summarization.
- **LocalDocs DB Importing Intricacies**: Members explored importing a `localdocs.db` file, but noted the file contains encrypted/specially encoded text that's difficult for a generic **LLM** to parse without an embedding model.
   - One member who lost their `localdocs.db` was experiencing painfully slow **CPU** indexing, seeking alternatives.
- **Win11 Update Erases LocalDocs**: A member reported that their `localdocs.db` became empty after a **Windows 11** update and was struggling to re-index the local documents on **CPU**.
   - Drive letter changes due to the update were suggested as a possible cause, with a recommendation to move files to the **C drive** to avoid such issues.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex enables Claude MCP Compatibility**: Members provided a simplified example of integrating **Claude MCP** with **LlamaIndex**, showcasing how to expose a localhost and port for MCP clients like **Claude Desktop** or **Cursor** using `FastMCP` and `uvicorn` in a [code snippet](https://link.to.snippet).
   - This integration allows developers to seamlessly connect **Claude** with **LlamaIndex** for enhanced functionality.
- **AgentWorkflow Accelerates LlamaIndex Multi-Agent Performance**: Users reported slow performance with **LlamaIndex MultiAgentic** setups using **Gemini 2.0** with 12 tools and 3 agents; a suggestion was made to use `AgentWorkflow` and the `can_handoff_to` field for [controlled agent interaction](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#multi-agent-systems-with-agentworkflow).
   - The discussion emphasized optimizing agent interactions for improved speed and efficiency in complex setups.
- **LlamaIndex Agent Types Decoded**: A member expressed confusion about the different agent types in **LlamaIndex** and when to use them, noting that a refactor on docs is coming soon.
   - A team member suggested `core.agent.workflow` should generally be used, with **FunctionAgent** for **LLMs** with function/tool APIs and **ReActAgent** for others, pointing to the [Hugging Face course](https://huggingface.co/learn/agents-course/en/unit2/llama-index/agents) for more help.
- **Automatic LLM Evals Launch with no Prompts!**: A founder is validating an idea for **OSS automatic evaluations** with a single API and no evaluation prompts, using proprietary models for tasks like Hallucination and Relevance in under 500ms.
   - More detail is available on the [autoevals.ai website](https://www.autoevals.ai) about its end-to-end solution, including models, hosting, and orchestration tools.
- **LlamaCloud becomes an MCP Marvel**: [LlamaCloud](https://www.llamaindex.ai/) can be an **MCP server** to any compatible client, as shown in [this demo](https://t.co/t8yteZLg19).
   - A member showed how to build your own **MCP server** using **LlamaIndex** to provide tool interfaces of any kind to any MCP client, with ~35 lines of Python connecting to **Cursor AI**, and implemented **Linkup web search** and [this](https://t.co/kj6UfDj0TU) project.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Google Gemini 2.5 Pro Debuts**: Google introduced **Gemini 2.5 Pro** as the *world's most powerful model*, highlighting its unified reasoning, long context, and tool usage capabilities, available experimentally in [Google AI Studio + API](https://x.com/OfficialLoganK/status/1904580368432586975).
   - They are touting experimental access as free for now, but pricing details will be announced shortly.
- **DeepSeek-V3-0324 Impresses with p5.js Program**: **DeepSeek-V3-0324** coded a p5.js program that simulates a ball bouncing inside a spinning hexagon, influenced by gravity and friction, as showcased in [this tweet](https://x.com/teortaxesTex/status/1904342699756433859).
   - The model also innovated features like ball reset and randomization based on a prompt that requested sliders for parameter adjustment and side count buttons.
- **SkyLadder Paper Highlights Short-to-Long Context Transition**: A paper on ArXiv introduces **SkyLadder**, a short-to-long context window transition for pretraining LLMs, showing gains of up to **3.7%** on common tasks ([2503.15450](https://arxiv.org/abs/2503.15450)).
   - They used **1B** and **3B** parameter models trained on **100B tokens** to achieve this performance.
- **Composable Generalization Achieved Through Hypernetworks**: A paper reformulates multi-head attention as a **hypernetwork**, revealing that a composable, low-dimensional latent code specifies key-query specific operations, allowing transformers to generalize to novel problem instances ([2406.05816](https://arxiv.org/abs/2406.05816)).
   - For each pair of q, k indices, the authors interpret activations along the head-number dimension as a latent code that specifies the task or context.
- **lm_eval Upgrade PR Awaits Review**: A pull request (PR) is open to update the evaluation logic in `gpt-neox` to `lm_eval==0.4.8`, the latest version, with potentially unrelated failing tests to be addressed in a separate PR, as linked here: [PR 1348](https://github.com/EleutherAI/gpt-neox/pull/1348).
   - Failures might be due to environment setup or inconsistent versioning of dependencies.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Sidelined for Websites**: Members advised against using **Mojo** for websites due to its lack of specifications for cryptographically sound code and a weak IO story, favoring **Rust** for its production-ready libraries.
   - It was suggested that **Rust's** faster async capabilities are better suited for applications requiring authentication or **HTTPS**.
- **Mojo's Hardware-Backed AES Implementation Paused**: A hardware-backed **AES** implementation in **Mojo** doesn't run on older Apple silicon Macs and isn't a full **TLS** implementation, leading to a pause in development.
   - The developer is waiting for a cryptographer to write the software half, citing risks of cryptographic functions implemented by non-experts.
- **SIMD Optimizations Boost AES**: Discussion centered on using **SIMD** for **AES**, noting that **x86** has **vaes** and similar features for **SIMD AES 128**.
   - It was also mentioned that **ARM** has **SVE AES**, which is similar but not as well supported, showcasing hardware optimizations for cryptographic functions.
- **Go Floated as a Backend Middle Ground**: As an alternative to **Rust**, a member suggested **Go** as a middle ground that is also production ready, while another expressed concerns about too many microservices.
   - Despite the challenges, a member expressed reluctance towards **Rust** due to the perception that it's not suitable for fast writing, seeking an easier solution for backend development, and the suggestion was to have the **Rust API call into it** and pass arguments along.
- **Mojo Bypasses CUDA with PTX for NVIDIA**: **Mojo** directly generates **PTX** (Parallel Thread Execution) code to target NVIDIA GPUs, bypassing **CUDA** and eliminating dependencies on **cuBLAS**, **cuDNN**, and **CUDA C**.
   - This approach streamlines the development process by avoiding the need for **CUDA**-specific libraries.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy tackles Summarization Task**: A member is exploring using **DSPy** for a text summarization task with **300** examples and is testing it out on a simple metric to see where exactly the summarization differed to make the optimizer more effective.
   - The feedback can be returned via `dspy.Prediction(score=your_metric_score, feedback="stuff about the ground truth or how the two things differ")` to guide the optimization.
- **SIMBA Provides Granular Feedback**: A member suggested using the experimental optimizer `dspy.SIMBA` for the summarization task, which allows for providing feedback on differences between the generated summary and the ground truth.
   - This level of feedback allows for more precise guidance during the optimization process.
- **Output Refinement with BestOfN and Refine**: A member shared a link to a [DSPy tutorial on Output Refinement](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/) explaining `BestOfN` and `Refine` modules designed to improve prediction reliability.
   - The tutorial elaborates on how both modules stop when they have reached `N` attempts or when the `reward_fn` returns an award above the `threshold`.
- **BestOfN Module Triumphs with Temperature Tweaks**: The `BestOfN` module runs a given module multiple times with different temperature settings to get an optimal result.
   - It returns either the first prediction that passes a specified threshold or the one with the highest reward if none meet the threshold.
- **Refine Module Composable?**: A member inquired if `Refine` is going to subsume assertions, and whether it's as granular and composable, since it wraps an entire module.
   - Another member responded that the composability can be managed by adjusting the module size, allowing for more explicit control over the scope.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD Legacy GPUs get tinygrad bump**: With the **OpenCL frontend**, older AMD GPUs (not supported by ROCm) such as those in a 2013 Mac Pro *can* potentially run with Tinygrad.
   - Success depends on the *custom driver* and level of OpenCL support available; users should verify their system compatibility.
- **ROCm Alternative emerges for Old AMD**: For older AMD GPUs, ROCm lacks support but the **OpenCL frontend** in tinygrad might offer a workaround.
   - Success will vary based on specific driver versions and the extent of OpenCL support; experimentation is needed.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Kicks Off Creators Club**: Windsurf launched a Creators Club rewarding community members for content creation, offering **$2-4 per 1k views**.
   - Details for joining can be found at the [Windsurf Creators Club](https://whop.com/windsurf/).
- **Windsurf Opens 'Vibe Coding' Channel**: Windsurf has created a new channel for 'vibe coders' to *enter the flow state*, chat, discuss, and share tips/tricks.
   - The goal is to enhance the coding experience by fostering a collaborative and immersive environment.
- **Windsurf v1.5.8 Patches Released**: **Windsurf v1.5.8** is now released with patch fixes, including cascade/memories fixes, Windsurf Previews improvements, and cascade layout fixes.
   - An image showcasing the release was also shared, highlighting the specific improvements.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1353805792582570035)** (916 messages🔥🔥🔥): 

> `Nebula vs other models, Gemini 2.5 models are out, Grok 3 issues, Llama 4 release, DeepSeek R2` 


- **Nebula vs Gemini Flash on Signal Processing**: The "Rage" model is better at **signal processing and math** than **Sonnet 3.7**, achieving a max error of **0.04**, as shown in attached [images](https://cdn.discordapp.com/attachments/1340554757827461211/1354014706779816018/image.png).
   - While some find **Gemini 2.0 Flash** to be equal to Rage, others note Rage is *very vulnerable to prompts*.
- **Gemini 2.5 breaks the LLM Arena**: **Gemini 2.5 Pro Experimental** launched in Google AI Studio and Gemini Advanced, achieving #1 on the [LM Arena leaderboard](https://x.com/lmarena_ai/status/1904581128746656099) with a massive score jump and leading in Math, Creative Writing, Instruction Following, Longer Query, and Multi-Turn.
   - Some members found Gemini 2.5 is excellent at HTML & Web Design, others found limitations.
- **Concerns about Grok 3 Issues**: Some users mentioned that holding long conversations with **Grok 3** reveals *a lot of issues*, while others emphasize that LM Arena's evaluation focuses beyond math and coding, encompassing creative writing and longer queries.
   - Users debate whether **Grok 3** truly deserves its top ranking, as *Grok 3 from Grok 2 was a huge leap thats for sure but its not a great model*
- **Debate over Python Calls in LM Arena**: Members debated whether models in the LM Arena were using **Python calls**, with some citing precise numerical calculations from o1 as evidence.
   - Some cited that that the **existence of a web search leaderboard** sorta implies the normal leaderboard doesn't have web access.
- **Gemini 2.5 timeline is insane**: Users expressed awe at **Google's rapid progress** with Gemini 2.5, with one noting the company is *moving so fast wtf*.
   - A user shared a note from **Sergey Brin** urging Google to *stop building nanny products*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OfficialLoganK/status/1904561688134967357?t=zVeue3sku3MQJM3XIRKcyA&s=19">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: @OpenAI : )</li><li><a href="https://x.com/noamshazeer/status/1904581813215125787?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from Noam Shazeer (@NoamShazeer)</a>: Introducing Gemini 2.5 Pro Experimental.The 2.5 series marks a significant evolution: Gemini models are now fundamentally thinking models.This means the model reasons before responding, to maximize ac...</li><li><a href="https://x.com/officiallogank/status/1904559860378915127?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: 🌌🥎👍</li><li><a href="https://www.theverge.com/command-line-newsletter/622045/google-ai-nanny-products">Google’s cofounder tells AI staff to stop ‘building nanny products’</a>: He also thinks they should be working 60-hour weeks to build AGI.</li><li><a href="https://x.com/jeffdean/status/1904580112248693039?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from Jeff Dean (@JeffDean)</a>: 🥁Introducing Gemini 2.5, our most intelligent model with impressive capabilities in advanced reasoning and coding.Now integrating thinking capabilities, 2.5 Pro Experimental is our most performant Ge...</li><li><a href="https://x.com/sundarpichai/status/1904575384466710607?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from Sundar Pichai (@sundarpichai)</a>: Nebula</li><li><a href="https://x.com/testingcatalog/status/1904527950076076323?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Google is set to reveal a new model for Gemini this week! In addition to that, Gemini will get a new toolbox for &#34;Agents&#34;, aka &#34;agentic use cases&#34; for Gemini, like Canvas ...</li><li><a href="https://x.com/petarv_93/status/1904643818030317579?s=46">Tweet from Petar Veličković (@PetarV_93)</a>: Gemini models are now capable enough to assist with fundamental AI research! Several theorems featured in our recent ICML submissions were co-proved with Gemini&#39;s help.2.5 Pro is a really good mod...</li><li><a href="https://x.com/testingcatalog/status/1904505417138372973?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Google started rolling out Project Astra for Gemini to more Android users. This feature enables vision capabilities on Gemini Live. The rollout is expected to be slow and gradual as the current adopti...</li><li><a href="https://x.com/OfficialLoganK/status/1904580368432586975?t=fKVOERgBUn3dfxTBvbtOgA&s=19">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Introducing Gemini 2.5 Pro, the world&#39;s most powerful model, with unified reasoning capabilities + all the things you love about Gemini (long context, tools, etc)Available as experimental and for ...</li><li><a href="https://x.com/sundarpichai/status/1904579419496386736?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from Sundar Pichai (@sundarpichai)</a>: 1/ Gemini 2.5 is here, and it’s our most intelligent AI model ever.Our first 2.5 model, Gemini 2.5 Pro Experimental is a state-of-the-art thinking model, leading in a wide range of benchmarks – with i...</li><li><a href="https://x.com/paulgauthier/status/1904637913411031410?s=46">Tweet from Paul Gauthier (@paulgauthier)</a>: Gemini 2.5 Pro sets SOTA on the aider polyglot leaderboard with a score of 73%.This is well ahead of thinking/reasoning models. A huge jump from prior Gemini models. The first Gemini model to effectiv...</li><li><a href="https://x.com/wintermoat/status/1904593298008006924">Tweet from Alphabetting (@wintermoat)</a>: @testingcatalog Seems like it changed his face. Gemini w native multimodal doesn&#39;t do that.</li><li><a href="https://x.com/lmarena_ai/status/1904581128746656099">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: BREAKING: Gemini 2.5 Pro is now #1 on the Arena leaderboard - the largest score jump ever (+40 pts vs Grok-3/GPT-4.5)! 🏆Tested under codename &#34;nebula&#34;🌌, Gemini 2.5 Pro ranked #1🥇 across ALL...</li><li><a href="https://x.com/alexandr_wang/status/1904589984591695874?s=46">Tweet from Alexandr Wang (@alexandr_wang)</a>: 🚨 Gemini 2.5 Pro Exp dropped and it&#39;s now #1 across SEAL leaderboards:🥇 Humanity’s Last Exam🥇 VISTA (multimodal)🥇 (tie) Tool Use🥇 (tie) MultiChallenge (multi-turn)🥉 (tie) Enigma (puzzles)Con...</li><li><a href="https://x.com/_clashluke/status/1904612478199173346">Tweet from Lucas Nestler (@_clashluke)</a>: tbc, this is 100% real</li><li><a href="https://www.reddit.com/r/singularity/comments/1jjm9s9/gemini_25_pro_internal_instructions/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://www.reddit.com/r/Bard/comments/1jjmta6/gemini_25_cannot_write/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://youtu.be/qE673AY-WEI?si=XsJ1AQqyriRzlv-Y">Building with Gemini 2.0: Native audio output</a>: Gemini 2.0 introduces multilingual native audio output. Watch this demo to see how this new capability can help developers build multimodal AI agents. These ...</li><li><a href="https://x.com/googleaidevs/status/1904586624333471975?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">Tweet from Google AI Developers (@googleaidevs)</a>: Join the team behind Gemini 2.5 as they dive into the model’s thinking and coding advancements.🎙️Space starts at 12:20pm PT. Drop your questions below.https://x.com/i/spaces/1MYxNwQLMjbKw</li><li><a href="https://old.reddit.com/r/Bard/comments/1jjjpiw/excuse_me_wtf/">Excuse me, WTF??</a>: Posted in r/Bard by u/interro-bang • 278 points and 106 comments
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1354173264997646376)** (1 messages): 

> `Perplexity answer modes, vertical search, web and mobile` 


- **Perplexity Gets Even Better Answer Modes**: Perplexity is introducing **answer modes** to improve the core search product for verticals like **travel**, **shopping**, **places**, **images**, **videos**, and **jobs**.
   - The goal is to become *super precise* so users won't need to manually select these tabs; the feature is currently available on the web and coming to mobile soon.
- **Mobile Answer Modes Coming Soon**: The new **answer modes** feature, currently available on the web, will soon be released on mobile as well.
   - This expansion aims to provide a consistent and improved search experience across different platforms.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1353805658146996254)** (991 messages🔥🔥🔥): 

> `Perplexity downtime, Electron vs native apps, DeepSeek V3, Image Generation, AI Models for Accuracy` 


- **Perplexity plunges into repeated product problems**: Users reported **Perplexity AI** experiencing multiple outages, leading to spaces and threads being wiped, and prompting frustration among those relying on it for important tasks like studying and thesis work, with one user exclaiming *exam in 12 hours*.
- **Electron framework faces fire from frustrated fanatics**: Members debated the merits of **Electron** for desktop apps, with some labeling it as *trash* and comparing its resource usage to launching a new **Google Chrome** instance for each app.
- **DeepSeek delivers dazzling developer dreams**: DeepSeek V3 gains coder kudos after update, and a member shared a [link to the DeepSeek subreddit](https://www.rxddit.com/r/DeepSeek/s/sYuAr1YKpx) discussing the AI's coding prowess and comparing it with **Claude 3.5 Sonnet**.
- **AI Image Generation gets put through its paces**: Members experimented with **Perplexity AI's** image generation capabilities, available after search via the generate image button, and one user hilariously shared results of asking for an iOS smiling face with a bigger smile for a logo, lamenting a lack of prompting skills.
- **Model accuracy and accuracy model**: Members discussed AI models for accuracy, with a user seeking advice on the best model and receiving recommendations for **R1**, **O1**, and **Claude 3.7 Sonnet Extended**, while others mentioned using **Deep Research** powered by **O3**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/apostraphi/status/1904362001121329566?s=61">Tweet from Phi Hoang (@apostraphi)</a>: comet is coming ☄️</li><li><a href="https://tenor.com/view/thurston-waffles-eyes-glow-gif-21980929">Thurston Waffles Eyes Glow GIF - Thurston Waffles Eyes Glow - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/%ED%8A%B8%EB%9F%BC%ED%94%84-%EC%9D%BC%EB%A1%A0-%EC%9D%BC%EB%A1%A0%EB%A8%B8%EC%8A%A4%ED%81%AC-%EC%B6%A4-musk-gif-14611234391948951568">트럼프 일론 GIF - 트럼프 일론 일론머스크 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/zutomayo-girl-dancing-gif-17878392983197209341">Zutomayo Girl GIF - Zutomayo Girl Dancing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/upgrades-robots-gif-21291099">Upgrades Robots GIF - Upgrades Robots - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/liar-why-the-fuck-you-lyin-dancing-why-the-fuck-why-are-you-lying-gif-7431053">Liar Why The Fuck You Lyin GIF - Liar Why The Fuck You Lyin Dancing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ogli-gif-17468683305861986751">Ogli GIF - Ogli - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/miau-hd-adobe-after-effects-glass-breaking-preset-gif-752576862881430143">Miau Hd Adobe After Effects Glass Breaking Preset GIF - Miau hd Adobe after effects glass breaking preset - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/happy-sad-markiplier-lol-meme-gif-25974730">Happy Sad Markiplier Lol Meme GIF - Happy Sad Markiplier LOL Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/don%27t-make-me-tap-the-sign-simpsons-bus-gif-5399805801037462082">Don&#039;T Make Me Tap The Sign Simpsons GIF - Don&#039;t Make Me Tap The Sign Simpsons Bus - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ants-bindle-hobo-stick-sad-ant-gif-6604577456488723514">Ants Bindle GIF - Ants Bindle Hobo stick - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/it-is-even-funnier-the-second-time-spongebob-spongebob-meme-meme-impact-font-gif-20058612">It Is Even Funnier The Second Time Spongebob GIF - It Is Even Funnier The Second Time Spongebob Spongebob Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/rage-smash-keyboard-streamer-twitch-gif-27138844">Rage Smash GIF - Rage Smash Keyboard - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://qwenlm.ai/">Qwen Chat</a>: no description found</li><li><a href="https://github.com/pnd280/complexity">GitHub - pnd280/complexity: ⚡  Supercharge your Perplexity.ai</a>: ⚡  Supercharge your Perplexity.ai. Contribute to pnd280/complexity development by creating an account on GitHub.</li><li><a href="https://www.rxddit.com/r/DeepSeek/s/iVHd6iPydH">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://www.rxddit.com/r/DeepSeek/s/sYuAr1YKpx">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://www.rxddit.com/r/DeepSeek/s/TjpQGSi6qT">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://tenor.com/view/trump-dance-trump-2024-trump-gif-12734161508561409577">Trump Dance Trump 2024 GIF - Trump dance Trump 2024 Trump - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.marketcalls.in/perplexity/what-is-perplexity-finance.html">What Is Perplexity Finance?</a>: Perplexity AI is rapidly transforming the way traders and investors access financial insights by leveraging the power of real-time, AI-generated data. Originally known for its accurate, citation-ba…</li><li><a href="https://www.zdnet.com/article/perplexity-ais-new-tool-makes-researching-the-stock-market-delightful-heres-how/">Perplexity AI&apos;s new tool makes researching the stock market &apos;delightful&apos;. Here&apos;s how</a>: Perplexity Finance is a comprehensive suite of AI-powered tools with an easy-to-use interface. Here&apos;s how to access it and what to know before you do.</li><li><a href="https://en.wikipedia.org/wiki/Perplexity_AI">Perplexity AI - Wikipedia</a>: no description found</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1g4kbyy/perplexity_for_finance_realtime_stock_quotes/">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1353873677321769001)** (7 messages): 

> `Perplexity AI searches, AI Analysis, Coinmarketcap API, Aircraft Material` 


- **Perplexity Search Party**: A member shared [a link to a Perplexity search](https://www.perplexity.ai/search/6c6be2aa-88c7-4307-ba9d-24b49c5e597f) and [another one](https://www.perplexity.ai/search/perplexity-ojn7W8xuS.G8tLGegm5LiA) along with [another search](https://www.perplexity.ai/search/i-want-to-know-about-the-lates-DZ0aIT2ATzeHH.Af3OrDTg).
   - It is unclear from the context what these searches were for or why they were shared.
- **Analyzing AI Potential, Again**: A member shared [a link to a Perplexity search about analyzing the potential for AI](https://www.perplexity.ai/search/analyze-the-potential-for-ai-a-OiZQZHrsTBqlfbPv4Pw3tA).
   - It seems that this link may have been unintentionally shared twice.
- **Coinmarketcap API Searched**: A member posted [a link to a Perplexity search about the Coinmarketcap API](https://www.perplexity.ai/search/api-coinmarketcap-iqpGD.7HQTaXxfLZ186I5g).
   - It is not clear why this search was conducted or shared in the channel.
- **Aircraft Materials Search**: A member shared [a link to a Perplexity search about what is used to make aircraft](https://www.perplexity.ai/search/what-is-used-to-make-aircraft-GvNGLO_USEq2b.Rx4iOyjg).
   - The context behind sharing this search is not provided.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1353807498884415589)** (11 messages🔥): 

> `truncated responses, Sonar Model, Perplexity Pro API credits, Sonar Pro in iOS app, API request costs` 


- ****Sonar** Model Truncates Responses**: Users report [truncated responses](https://cdn.discordapp.com/attachments/1161802929053909012/1353812240603676713/Screenshot_2025-03-24_at_2.26.23_PM.png?ex=67e454e6&is=67e30366&hm=26dbdca22f5c34259a237975576497677d8beb98c81530b77ea3a253018ed4ac) with the **Sonar** model, starting about two days ago, where the response cuts off mid-sentence with a **200** response.
   - One user mentioned experiencing this issue even when receiving around **1k tokens** and was directed to report the bug in the dedicated bug report channel.
- **Pro Users Ask about **Perplexity Pro** API**: A new **Perplexity Pro** user inquired about receiving API credits with their subscription.
   - Another user provided a [link to the Perplexity AI Help Center](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro) with details on **Perplexity Pro**.
- **Inquiry About **Sonar Pro** Access on iOS**: A user asked if **Sonar Pro** is available within the iOS app for **Perplexity Pro** subscribers.
   - They noted that only regular **Sonar** is listed as an option in the default model settings and wondered if **Sonar Pro** is currently exclusive to the API for developers.
- **Concern on how to Cut **API** Costs**: A user raised concerns about the cost of **$5 per 1000 API requests** and inquired about possible ways to optimize or reduce this expense.
- **Limitation of **API** to 5 Steps?**: A user noticed that the **API** seems limited to **5 steps**, whereas they have observed up to **40 steps** on the web app.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1353807067298926620)** (872 messages🔥🔥🔥): 

> `Augment vs Cursor codebase analysis, Claude 3.7 MAX vs Claude 3.7, Vibe Coder & MCPs, New Deepseek V3 and Gemini 2.5, ASI Singularity` 


- **Augment Beats Cursor for Codebase Analysis**: Members discussed why [Augment](https://www.augment.app/) is better at analyzing large codebases compared to Cursor, with one stating that it *uses full context*.
   - Some suggested that Augment uses a different system to search files, not just feeding the entire codebase into the LLM, and suggested using **Claude 3.7 Max** for full context instead.
- **Debate on Claude 3.7 MAX Full Context**: The difference between **Claude 3.7 MAX** and **Claude 3.7** is that MAX has full context while the non-MAX version has limited context and 25 agent calls before needing to resume.
   - This limitation refers to both context window and the amount of context added in a single prompt.
- **Vibe Coder's Knowledge Cutoff Risks**: The Vibe coder could be in trouble if they don't use **Model Context Protocols (MCPs)** to mitigate the cut-off knowledge of LLMs and code translation for the next version may be difficult.
   - It's critical to update frameworks because most AI uses out-of-date frameworks that are susceptible to errors that might go unnoticed, but you can use MCPs such as **Exa Search** or **Brave Search** to mitigate this for Claude.
- **Deepseek V3 challenges Claude 3.7**: The new **Deepseek V3** is shown to outperform Claude 3.5 (and maybe 3.7) in several tests, and new data also shows the release of a real-world coding benchmark for **Deepseek V3 (0324)**.
   - The new Deepseek V3 model is considered impressive, with one member suggesting that using **Deepseek V3 Latest (Cline) as Architect + Sonnet 3.5 as Executioner (Cursor)** could be a solid coding approach.
- **The ASI Singularity is coming!**: The discussion focused on **achieving ASI Singularity (Godsend) soon** to preempt potential AI-related chaos, while a member commented that  A lot of stuff is getting dragged out on purpose while a lot of people live lives like it will last forever.
   - Members talked about how true AGI is unachievable due to not understanding our brain completely, or an LLM's limitations, but the new AGI is more so *a Super-System that uses LLMs + Algorithmic software + robotics*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://agent-tars.com/">Agent TARS - Open-source Multimodal AI Agent</a>: no description found</li><li><a href="https://unfuckit.ai">Unfuckit AI</a>: no description found</li><li><a href="https://exa.ai/">Exa</a>: The Exa API retrieves the best, realtime data from the web for your AI</li><li><a href="https://x.com/i/status/1894821477230485570">Tweet from ElevenLabs (@elevenlabsio)</a>: Introducing Scribe — the most accurate Speech to Text model.It has the highest accuracy on benchmarks, outperforming previous state-of-the-art models such as Gemini 2.0 and OpenAI Whisper v3.It’s now ...</li><li><a href="https://supermaven.com/">Supermaven: Free AI Code Completion</a>: The fastest copilot. Supermaven uses a 1 million token context window to provide the highest quality code completions.</li><li><a href="https://marketplace.visualstudio.com/items?itemName=icrawl.discord-vscode">Discord&#32;Presence&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Update&#32;your&#32;discord&#32;status&#32;with&#32;a&#32;rich&#32;presence.</li><li><a href="https://marketplace.visualstudio.com/items/?itemName=LeonardSSH.vscord">Discord&#32;Rich&#32;Presence&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Highly&#32;customizable&#32;Discord&#32;Rich&#32;Presence&#32;extension&#32;for&#32;Visual&#32;Studio&#32;Code</li><li><a href="https://docs.cursor.com/settings/beta">Cursor – Early Access Program</a>: no description found</li><li><a href="https://x.com/playwrightweb/status/1904265499422409047?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Playwright (@playwrightweb)</a>: With all the MCP hype, we went ahead and built an MCP server for Playwright. Ours is snapshot-based, which makes it faster and more reliable! You can opt into the visual mode too. Have fun! 🚀 #Playwr...</li><li><a href="https://tenor.com/view/orangutan-orangutans-monkey-monkeys-punch-gif-25064862">Orangutan Orangutans GIF - Orangutan Orangutans Monkey - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ai.google.dev/gemini-api/docs/rate-limits#free-tier">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=xAcTmDO6NTI&list=PLUl4u3cNGP62A-ynp6v6-LGBCzeH3VAQB"> - YouTube</a>: no description found</li><li><a href="https://www.reddit.com/r/cursor/comments/1jj78mr/how_i_bypassed_claude_37s_context_window/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://github.com/AgentDeskAI/browser-tools-mcp">GitHub - AgentDeskAI/browser-tools-mcp: Monitor browser logs directly from Cursor and other MCP compatible IDEs.</a>: Monitor browser logs directly from Cursor and other MCP compatible IDEs. - AgentDeskAI/browser-tools-mcp</li><li><a href="https://www.youtube.com/watch?v=lB3S_l9SoMA"> - YouTube</a>: no description found</li><li><a href="https://github.com/exa-labs/exa-mcp-server">GitHub - exa-labs/exa-mcp-server: Claude can perform Web Search | Exa with MCP (Model Context Protocol)</a>: Claude can perform Web Search | Exa with MCP (Model Context Protocol) - exa-labs/exa-mcp-server</li><li><a href="https://www.cursor.com/changelog">Changelog | Cursor - The AI Code Editor</a>: New updates and improvements.</li><li><a href="https://www.reddit.com/r/cursor/comments/1jdcy3k/office_hours_with_devs/">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1354118390972944424)** (2 messages): 

> `4o image generation, ChatGPT, Sora` 


- **4o Images Coming to ChatGPT and Sora**: Members are excited for the prospect of **4o image generation** in **ChatGPT** and **Sora**.
- **More 4o Details Incoming**: The community awaits further details on the release and capabilities of the **4o model**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1353807455548866641)** (300 messages🔥🔥): 

> `GPT-4o mini vs Gemini 2.0 Flash for property tag extraction, Operator OAI expansion plans, GPT context window limitations and hallucinations, Best AI models for various tasks, DeepSeek V3 03-24's playful behavior` 


- **4o Mini and Gemini Flash Duel for Tag Domination**: Members are testing whether [OpenAI's 4o mini](https://openai.com/index/hello-gpt-4o/) or **Gemini 2.0 Flash** is superior for extracting property tags from listings, noting similar pricing but task-dependent performance.
   - One member suggested to *just try them out* as *sometimes 4o mini is better, sometimes gemini 2 flash is better*.
- **Operator OAI Plugs into Plus, Team, and Enterprise**: OpenAI announced plans to expand [Operator](https://openai.com/index/introducing-operator/) to Plus, Team, and Enterprise users and integrate its capabilities into **ChatGPT**.
   - One member noted that 3rd party operators, which allow you to load your browser, are better anyway.
- **Context Crunch Causes AI Dementia**: Users discussed how exceeding **GPT's context window** (8k for free, 32k for Plus, 128k for Pro) leads to loss of detail and hallucinations in long stories.
   - Custom GPTs or projects using PDFs can help, but the chat history itself is still subject to the limit.
- **Model Mania: Ranking AI Talent**: A user shared their list of best AI models for each task, rating **ChatGPT** for math/research/writing, **Claude** for coding, **Grok** for queries, **Perplexity** for search/knowledge, and **Deepseek** for open source.
   - Others suggested **Gemma 27b**, **Mistral 3.1**, **QW-32b**, and **Nemotron-49b**, with one noting Grok's top ranking in coding on LMSYS.
- **Gemini 2.5 Pro Annihilates the Competition?!**: **Gemini 2.5 Pro** is making waves, with claims it *destroys* **ChatGPT o3-mini-high** and leads common benchmarks by meaningful margins, debuting at #1 on [LMArena](https://lmarena.ai/?lea).
   - One user exclaimed *Gemini just beat everything!*, while another hoped it was *just a benchmark... thingy*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.newyorker.com/culture/the-weekend-essay/your-ai-lover-will-change-you">Your A.I. Lover Will Change You</a>: A future where many humans are in love with bots may not be far off. Should we regard them as training grounds for healthy relationships or as nihilistic traps?</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-re">Gemini 2.5: Our most intelligent AI model</a>: Gemini 2.5 is our most intelligent AI model, now with thinking.</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning">Gemini 2.5: Our most intelligent AI model</a>: Gemini 2.5 is our most intelligent AI model, now with thinking.</li><li><a href="https://www.youtube.com/watch?v=2f3K43FHRKo"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1354141422919487622)** (2 messages): 

> `Chat GPT Speed Degradation` 


- **Chat GPT Speed Declines**: A user inquired whether [Chat GPT](https://openai.com/blog/chatgpt) has become increasingly slow.
   - The user thought *it was just them*, followed by a *lol*.
- **Another user said it's not just you**: Following that first user expressing their opinion, another user agreed that it's happening to them as well.
   - No further details were given.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1353811076684185683)** (159 messages🔥🔥): 

> `Proprietary prompt technique for memory retention, Benchmarking AI performance, Open sourcing prompts, Using python to prompt chatgpt, Building custom GPTs` 


- **Proprietary AI Memory System Spurs Curiosity**: A member claims to have created a runtime OS through prompting that maintains memory past **700 turns** without drift, adapting to communication styles, but is keeping it proprietary.
   - This system is said to feature a *dynamic cognitive architecture* and *real-time role synthesis*, though details remain under wraps.
- **Prompt Engineer Hopes to Benchmark AI System**: A member seeks advice on benchmarking their AI system without open-sourcing the code, aiming to demonstrate its capabilities, with suggestions including **MMLU**, **Big Bench**, and **ARC AGI**.
   - They also mentioned developing internal metrics like a **Runtime Benchmark Score** assessing response efficiency, capsule stack load, hydration strategy, fallback performance, and snapshot & recovery.
- **Members Debate Open Source Versus Proprietary AI Work**: A member initially kept a project private for fear of others stealing it, but then they were encouraged to open source their work to **get attention and user testing**.
   - Concerns were raised about others copying their work without proper attribution, leading to a discussion of open-source licenses like **GPL_v3** for protection.
- **Python's Power and Limitations for ChatGPT Prompting**: A member with only a plus account learned about using **Python in ChatGPT** for tasks such as chaining prompts and managing context, typically using Python's code interpreter.
   - It was noted that managing extensive context can crash the browser due to its *quadratic* nature, a problem more easily handled in Python due to how **Python doesn't have to manage quadratic context**.
- **Clever Commenting Creates Custom GPT**: Members discussed a method of guiding **GPT creation using floating comments** within a template, with instructions to the AI to populate each section by asking targeted questions, building from the evolving context.
   - This facilitates a *meta-meta-prompting* approach, requiring pre-existing prompt engineering skills and confidence in the template's structure, making GPTs easier to build.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1353811076684185683)** (159 messages🔥🔥): 

> `Proprietary AI system, Dynamic cognitive architecture, Runtime OS through prompt, Large context maintenance, GPL release discussion` 


- **Propritary AI System boasts cognitive architecture**: A member is building a proprietary AI system with a dynamic cognitive architecture, capable of real-time role synthesis, complex context management, and emergent behavior modeling, and evolving through self-optimization and adaptive learning.
   - The member believes their system can maintain memory past **700 turns** without drift or hallucination and adapt to unique communication styles, describing it as *a runtime OS that functions through the prompt*.
- **LLM Gaslighting is New Horse Whispering**: A member shared a provocative analogy: *Llms are like nervous horses you have to gaslight into becoming specific other kinds of horses*
   - This was followed by a discussion on whether **20,000 tokens** in a conversation is considered a large context, what happens when the context collapses.
- **GPT Custom Template aids lazy GPT Builds**: A member shared a [GPT Custom Template](https://platform.openai.com/docs/overview) with a floating comment to build custom GPTs from the `Create` pane, walking them through building the GPT lazily.
   - They input the template and instructions, and the A.I. asks them questions and builds the GPT, enabling them to build while distracted, requiring pre-existing ability to prompt and confidence in the templatized structure.
- **Discussion about GPL License for AI System**: Members discussed using the [GPL_v3 license](https://www.gnu.org/licenses/gpl-3.0.en.html) for an AI system, balancing freedom for users and control for creators.
   - The creator is preparing a GPL release and working on a production model, suggesting that *including specific requests in the comments of the code will greatly increase the quality of the output*.


  

---


### **OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1353814591175393352)** (1 messages): 

> `FormulaGPT, AI Racing Simulator, Open Source AI, AI Strategy Decisions` 


- ****FormulaGPT** brings LLMs to the Racing Track**: **FormulaGPT** is an experimental racing simulator where advanced AI language models like **GPT-4**, **Claude**, or **DeepSeek** compete as racing strategists.
   - It features **two distinct game modes**: Player vs. AI and AI vs. AI, where the AI teams think contextually and adaptively, making nuanced decisions based on the evolving scenario rather than fixed scripts.
- **Dive into Adaptive AI Strategy**: Unlike traditional bots, **FormulaGPT's** AI teams continuously reason, strategize, and make nuanced decisions based on the evolving scenario.
   - Users can observe detailed AI reasoning behind each pit stop, tire change, or overtaking maneuver, making it part racing game, part AI psychology lab.
- **FormulaGPT Goes Open Source**: **FormulaGPT** is fully [open-source under the MIT License](https://github.com/dawid-maj/FormulaGPT/), allowing users to explore, contribute, and customize the project.
   - It encourages users to dive into the code and adapt it to their liking, fostering community contributions and project enhancements.



**Link mentioned**: <a href="https://github.com/dawid-maj/FormulaGPT/">GitHub - dawid-maj/FormulaGPT: FormulaGPT – AI-powered Formula 1 race simulator with real-time team management and strategy decisions.</a>: FormulaGPT – AI-powered Formula 1 race simulator with real-time team management and strategy decisions. - dawid-maj/FormulaGPT

  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1354202486465892507)** (1 messages): 

> `Gemini 2.5 Pro support, DeepSeek V3 0324 support, aider /context command, aider /edit alias, Claude 3.7 Sonnet 'overeager' mode` 


- **Aider Supports Gemini 2.5 Pro**: Aider v0.79.0 now supports **Gemini 2.5 Pro**, expanding the range of models that can be used with the tool.
   - This update allows users to leverage the capabilities of **Gemini 2.5 Pro** within the Aider environment.
- **Aider Supports DeepSeek V3 0324**: Support for **DeepSeek V3 0324** has been added to Aider v0.79.0, offering users another model option.
   - The integration of **DeepSeek V3 0324** enhances the versatility of Aider.
- **Aider Introduces /context Command**: A new **/context** command has been added to Aider, which automatically identifies the files that need editing for a given request.
   - This feature streamlines the editing process by pinpointing the relevant files.
- **Aider Implements /edit Alias**: The command **/edit** has been introduced as an alias for the **/editor** command in Aider.
   - This change provides a more convenient and shorter alternative for accessing the editor functionality.
- **Aider Tames Claude 3.7 Sonnet with 'Overeager' Mode**: Aider now features an "overeager" mode for **Claude 3.7 Sonnet** models, designed to ensure it stays within the requested scope.
   - This mode aims to maintain functionality and prevent the model from straying beyond its intended parameters.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1353812389379837952)** (532 messages🔥🔥🔥): 

> `DeepSeek V3 performance, Gemini 2.5 Pro release, GPT-4o image generation, aider /context command` 


- **DeepSeek V3 Smashes Aider's Polyglot Benchmark**: DeepSeek's new **V3** scored **55%** on aider's polyglot benchmark, a significant improvement over the prior version.
   - It's the **#2 non-thinking/reasoning model**, behind only Sonnet 3.7, and competitive with thinking models like R1 & o3-mini, according to [Aider Leaderboards](https://aider.chat/docs/leaderboards/).
- **Google Releases Gemini 2.5 Pro Experimental**: Google released an experimental version of **Gemini 2.5 Pro**, claiming it leads common benchmarks, including the top spot on [LMArena](https://lmarena.ai/?lea), with reported **74% whole** and **68.6% diff** scores on aider's polyglot benchmark.
   - Users reported the model to be performant in generating architecture diagrams from codebases, although some found its coding abilities inconsistent and rate limits restrictive.
- **GPT-4o Image Generation arrives as DALL-E 3**: **GPT-4o's** image generation is rolling out to Plus, Pro, Team, and Free users as the default image generator in ChatGPT.
   - Some users find the generated images high quality while others are still seeing **DALL-E 3** style outputs, and access to **DALL-E** can still be accessed through a dedicated **DALL-E GPT**.
- **Aider's Context Command adds Extra Punch**: The new `/context` command explores the code base, and the command can be used with any other prompt command.
   - It is unclear if the command has higher token usage, or if there is an increased repomap size to work correctly; more detail can be found in [Discord message](https://discord.com/channels/1131200896827654144/1131200896827654149/1353181605211934830).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OfficialLoganK/status/1904580368432586975">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Introducing Gemini 2.5 Pro, the world&#39;s most powerful model, with unified reasoning capabilities + all the things you love about Gemini (long context, tools, etc)Available as experimental and for ...</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini Pro 2.5 Experimental (free) - API, Providers, Stats</a>: Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. Run Gemini Pro 2.5 Experimental (free) with API</li><li><a href="https://x.com/alibaba_qwen/status/1897361654763151544?s=46">Tweet from Qwen (@Alibaba_Qwen)</a>: Today, we release QwQ-32B, our new reasoning model with only 32 billion parameters that rivals cutting-edge reasoning model, e.g., DeepSeek-R1.Blog: https://qwenlm.github.io/blog/qwq-32bHF: https://hu...</li><li><a href="https://x.com/OfficialLoganK/status/1904583353954882046">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: This will mark the first experimental model with higher rate limits + billing. Excited for this to land and for folks to really put the model through the paces!This was the #1 point of feedback, besid...</li><li><a href="https://x.com/sundarpichai/status/1904579419496386736?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from Sundar Pichai (@sundarpichai)</a>: 1/ Gemini 2.5 is here, and it’s our most intelligent AI model ever.Our first 2.5 model, Gemini 2.5 Pro Experimental is a state-of-the-art thinking model, leading in a wide range of benchmarks – with i...</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free">DeepSeek V3 0324 (free) - API, Providers, Stats</a>: DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) mode...</li><li><a href="https://x.com/geminiapp/status/1904579704079724599?s=46">Tweet from Google Gemini App (@GeminiApp)</a>: 📣 Today, we’re introducing Gemini 2.5, our most intelligent AI model.An experimental version of Gemini 2.5 Pro is available now in the Gemini app for Gemini Advanced users: http://gemini.google.com/a...</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://tenor.com/view/laughing-laugh-lol-funny-haha-gif-16205592">Laughing Lol GIF - Laughing Laugh Lol - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324">DeepSeek V3 0324 - API, Providers, Stats</a>: DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) mode...</li><li><a href="https://aider.chat/docs/llms/anthropic.html#thinking-tokens)">Anthropic</a>: aider is AI pair programming in your terminal</li><li><a href="https://aistudio.google.com/prompts/new_chat">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=2fT0vSsB01g">Agentic Flows with MCP: Seamless Tool Integration for Every AI Assistant</a>: Agentic Flows with MCP: Seamless Tool Integration for Every AI AssistantIn this episode, we explore a revolutionary feature, the MCP Bridge, that allows any ...</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025">Gemini 2.5: Our most intelligent AI model</a>: Gemini 2.5 is our most intelligent AI model, now with thinking.</li><li><a href="https://github.com/mattstauffer/Torch">GitHub - mattstauffer/Torch: Examples of using each Illuminate component in non-Laravel applications</a>: Examples of using each Illuminate component in non-Laravel applications - mattstauffer/Torch
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1353826787599187998)** (39 messages🔥): 

> `Deepseek API Usage with Aider, LLM Hallucinations, Aider's Architecture Mode, NotebookLM for Context Priming, OpenRouter Configuration Issues` 


- **Deepseek's API Hallucinates as GPT-4**: Users reported that when using **DeepSeek's API** through Aider, the model incorrectly identifies itself as **OpenAI's GPT-4**, despite correct API key configuration and confirmed usage on the DeepSeek platform.
   - The phenomenon seems related to training data containing frequent mentions of ChatGPT, as discussed in this [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/).
- **NotebookLM Boosts Aider's Context**: A user suggested leveraging **NotebookLM** to enhance Aider's context priming process, particularly for large, unfamiliar codebases using [RepoMix](https://github.com/simonireilly/repo-mix).
   - The suggested workflow involves repomixing the repo, adding it to NotebookLM, including relevant task references, and then querying NotebookLM for relevant files, implementation suggestions, and a comprehensive plan to guide prompting in Aider.
- **OpenRouter Aliases Trigger Litellm Errors**: A user encountered a `litellm.APIConnectionError` when attempting to use **Claude 3.7 Sonnet** via **OpenRouter**, with other models working fine.
   - Another user replicated the configuration successfully, suggesting the issue may be specific to the user's OpenRouter setup rather than Aider itself; see this [Discord thread](https://discord.com/channels/1131200896827654144/1349300906864279584).
- **DeepSeek V3 Impresses Aider's Homepage**: Paul Gauthier tested the new **DeepSeek V3** model on the Aider homepage, noting its suggestion to upgrade the emojis to SVG icons.
   - See the [tweet](https://x.com/paulgauthier/status/1904310818868785196?s=46&t=AkDCTtZVFFazuKDknG6fLA) for more details.
- **Aider Configuration with thinking-tokens**: A user asked how to use `--thinking-tokens` when launching aider.
   - Another user linked the [sample configuration file](https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file), explaining that it should be a top-level key in `.aider.conf.yml`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1904310818868785196?s=46&t=AkDCTtZVFFazuKDknG6fLA">Tweet from Paul Gauthier (@paulgauthier)</a>: I took the new DeepSeek V3 for a test drive, and asked it to improve the http://aider.chat homepage. It suggested upgrading the emojis to some nice SVG icons.</li><li><a href="https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file)">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1353815061273116683)** (284 messages🔥🔥): 

> `HF Transformers, Deepseek model patching, FP8 Loading Scheme, Quantization impact on model performance, GGUF Uploads` 


- **Transformer Best Bets, Says User**: Members recommend **Hugging Face Transformers**, books on **linear algebra**, and learning **PyTorch** as the best approach.
   - One member suggests setting up a **dynamic quantization** at runtime schema with **HF Transformers** to stream weights to **FP4/FP8** with **Bits and Bytes** as they load.
- **Sneaky Model Patches, Says User**: According to one user, companies like **Deepseek** sometimes patch models before releasing the weights.
   - They add that at the very least a naive **FP8 loading scheme** could still be done on day zero, though it wouldn't be equivalent in quality to a fine-grained FP8 assignment.
- **Quantization Hurts? Say It Ain't So!**: A member warns that naively quantizing can significantly hurt model performance.
   - They asked, *Tbh, is there even a reason to run new models on day zero? I feel like it's really not a great burden to just wait a week.*
- **GGUF Uploads Incoming!**: According to one user, Unsloth is uploading **DeepSeek-V3-0324 GGUFs** with Dynamic Quants that are *selectively quantized*.
   - These **Dynamic Quants** will greatly improve accuracy over standard bits, so sit tight!
- **Llama 3 + Vision Collab Issues**: Members reports an issue trying to train **gemma3 4b** for vision, which raises the **RuntimeError**: *expected scalar type BFloat16 but found float*.
   - One member recommends trying [this notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb), while another confirms that **Gemma3 finetuning** is supported for non-bf16 devices when using the notebooks on Colab.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large">🪿Qwerky-72B and 32B : Training large attention free models, with only 8 GPU&#x27;s</a>: ‼️ Attention is NOT all you need ‼️</li><li><a href="https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4">nvidia/Llama-3.3-70B-Instruct-FP4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/unsloth/qwen25-vl-all-versions-679ca6c784fad5bd976a05a1">Qwen2.5-VL (All Versions) - a unsloth Collection</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: no description found</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-unsloth-notebook">Kaggle Llama 3.1 8b Unsloth notebook</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit">unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/goku-super-saiyan-super-saiyan2-super-saiyan2goku-goku-vegeta-gif-23177097">Goku Super Saiyan Super Saiyan2 GIF - Goku Super Saiyan Super Saiyan2 Super Saiyan2Goku - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/gohan-dbz-gif-9459511">Gohan Dbz GIF - Gohan Dbz - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-7.-running--saving-the-model">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics and best practices of fine-tuning. Beginner-friendly.</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-12.-saving-the-model">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://huggingface.co/mmnga/DeepSeek-V3-0324-experts-pertok-4-gguf/tree/main">mmnga/DeepSeek-V3-0324-experts-pertok-4-gguf at main</a>: no description found</li><li><a href="https://youtu.be/kVM-ANbCn4M"> - YouTube</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebook">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/huggingface/transformers">GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/lora-hyperparameters-guide">LoRA Hyperparameters Guide | Unsloth Documentation</a>: Best practices for LoRA hyperparameters and learn how they affect the finetuning process.</li><li><a href="https://github.com/unslothai/notebooks?tab=readme-ov-file#-kaggle-notebooks">GitHub - unslothai/notebooks: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more.</a>: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more. - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L957-L959">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1353951853502664774)** (11 messages🔥): 

> `Funny Graphs in CS Research, Benchmark Dissatisfaction, AI Project Recruitment` 


- **Funny Graphs Illustrate Obvious Point**: Members shared an image of a graph in CS research, finding it funny because *the point could've gotten across in a single sentence* ([image link](https://cdn.discordapp.com/attachments/1179039861576056922/1353951853276299334/image.png?ex=67e42e2d&is=67e2dcad&hm=57aee78a6c2b6d2c62c98e321bb3206ee33608d0eaa6833e07aaab1c5c5e8b36&)).
   - Another member agreed, stating that *it's satisfying to see* such graphs regardless ([another image link](https://cdn.discordapp.com/attachments/1179039861576056922/1353960964793438260/image.png?ex=67e436a9&is=67e2e529&hm=b91c937004e77037696a3aaa251549981bcaf7010f388e7c46a02a96d0df9ae1&)).
- **Members Finds Benchmarks Suck**: A member expressed dissatisfaction with benchmarks, stating that **Gemini failed abysmally** at extracting questions from **500k tokens**, despite claims of **100% recall** at finding the needle.
   - According to the member, **2M tokens** performed even worse, and **64k tokens** was required for it to function, with **Grok** performing the best in their experience.
- **AI Project Seeks Reliable People**: A member posted a recruitment message seeking *a reliable person for an AI Project*, with tech skills not a must.
   - The position is open to citizens of **USA, Australia, Canada, UK, Switzerland, Netherlands, German**, offering **$500 weekly compensation**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1353809750717894697)** (66 messages🔥🔥): 

> `Deepseek facts learning, Unsloth Gemma 3 27b error, phi4 fine tuning with unsloth, Medical bot fine tuning, LightRAG issues` 


- ****Deepseek Learns New Tricks**?**: A user inquired about tools to teach **Deepseek** additional facts via conversation or inserting data points into the training model for use as a personal assistant.
   - The user was *thinking in terms of using it as a personal assistant with figures and facts and dates etc.*
- ****Gemma 3 Glitch Gives Grief****: A user encountered a `TypeError` when loading `unsloth/gemma-3-27b-it-unsloth-bnb-4bit` for text-only finetuning due to redundant `finetune_vision_layers` parameter.
   - Another user pointed out that `FastLanguageModel` already sets `finetune_vision_layers = False` under the hood as seen in [Unsloth's GitHub](https://github.com/unslothai/unsloth/blob/e80d642bc777f7a219bdd34aea1a77751f066785/unsloth/models/llama.py#L2034).
- ****Fine-Tuning Mishap: Model Merging Mayhem****: A user who completed a **phi4** fine-tuning with Unsloth inquired whether to merge LORA adapters with only the "lamafied" version or if a generic base **phi4** model can be used.
   - A member said *I guess you can do both, but better with same model as use during training*.
- ****Fine-Tuning for the Future Medics****: A user seeks guidance on building an end-to-end medical bot through fine-tuning **Llama** using Unsloth, asking for advice on where to start.
   - No actionable advice was given.
- ****Visionary Void: Mistral 3.1's Image Issues****: A user reported issues with **Mistral 3.1** not processing images correctly in a local **Ollama** instance, with the model outputting nonsensical responses to image inputs.
   - The problem may be related to the mmproj file not getting loaded according to users, the *model lose[s] the vision abilities*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/conda-install,">Unsloth Documentation</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M">unsloth/DeepSeek-R1-GGUF at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/e80d642bc777f7a219bdd34aea1a77751f066785/unsloth/models/llama.py#L2034">unsloth/unsloth/models/llama.py at e80d642bc777f7a219bdd34aea1a77751f066785 · unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1354072552015659049)** (1 messages): 

> `GRPO on AWS, Tensorfuse, LoRA modules` 


- **GRPO + Unsloth on AWS Boosts LLM Workflows**: A guide was shared on running **GRPO** (DeepSeek’s RL algo) + **Unsloth** on AWS accounts, using a **vLLM server** with Tensorfuse on an **AWS L40 GPU**.
   - The guide shows how to transform **Qwen 7B** into a reasoning model, fine-tuning it using Tensorfuse and GRPO, and saving the resulting **LoRA adapter** to Hugging Face; the guide is available at [tensorfuse.io](https://tensorfuse.io/docs/guides/reasoning/unsloth/qwen7b).
- **Tensorfuse Simplifies LoRA Sharing**: The guide shows how to save fine-tuned **LoRA modules** directly to **Hugging Face** for easy sharing, versioning, and integration, backed to s3.
   - You can try out the whole flow on [Tensorflow's website](https://prod.tensorfuse.io/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tensorfuse.io/docs/guides/reasoning/unsloth/qwen7b">Transforming Qwen 7B into Your Own Reasoning Model - Tensorfuse</a>: no description found</li><li><a href="https://prod.tensorfuse.io/">Tensorfuse</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1353840146641846402)** (13 messages🔥): 

> `GRPO limitations, FFN Fusion for LLMs, DAPO experiments, AI Project Job Spam, Transformer Talk` 


- **GRPO Gotchas Galore**: Members are tweaking GRPO (Gradient Ratio Policy Optimization) to correctly train 2+ turns, as it currently only supports prompt/completion and found [this helpful video](https://www.youtube.com/watch?v=M3b59lZYBW8).
   - The member is also curious if new neural models or hybrid evolutionary neural approaches might help solve problems like arc.
- **FFN Fusion Frenzy Fuels Faster LLMs**: [FFN Fusion](https://huggingface.co/papers/2503.18908) is introduced as an architectural optimization technique that reduces sequential computation in large language models by identifying and exploiting natural opportunities for parallelization.
   - This technique transforms sequences of **Feed-Forward Network (FFN)** layers into parallel operations, significantly reducing **inference latency** while preserving **model behavior**.
- **Transformer Talk Transcends Traditional Thinking**: A member shared [a YouTube talk](https://www.youtube.com/watch?v=FAspMnu4Rt0) on transformers that they found insightful, opening up a bunch of areas for investigation.
   - Another member wondered if there is a metric to find very early in training to allow for model fusion before fully training the model.
- **DAPO Discussions Derail, Demand Data**: Multiple members are curious if anyone has experimented with DAPO (likely **Direct Preference Optimization**), or tried to recreate results with **verl**.
   - One member posted an image, possibly related to DAPO results, seeking confirmation of replication, but details remain unclear.



**Link mentioned**: <a href="https://huggingface.co/papers/2503.18908">Paper page - FFN Fusion: Rethinking Sequential Computation in Large Language Models</a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1353811602486464632)** (232 messages🔥🔥): 

> `Qwen VL series training details, Qwerky-72B transformerless model, Gemini 2.5 Pro performance, 4o image generation, AI Studio vs Gemini Advanced` 


- **Qwen VL Series Needs DeepSeek-Style Training Deets**: A member expressed a desire for a [DeepSeek-R1 style paper](https://www.deepseek.com/blog/deepseek-r1-a-strongly-competitive-open-source-language-model) with comprehensive training details for the **Qwen VL series**.
   - The poster hoped for *tons of training details* including parameters, training compute and dataset mixtures.
- **Qwerky-72B Ditches Attention, Nears 4o-Mini**: Featherless AI introduced [Qwerky-72B and 32B](https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large), transformerless models trained on **8 GPUs** that *surpass GPT 3.5 Turbo and approach 4o-mini* in evals with 100x lower inference cost using RWKV linear scaling.
   - The training process involved *freezing all weights, deleting the attention layer, replacing it with RWKV, and training it through multiple stages*.
- **Gemini 2.5 Pro Snags #1 Spot, Google Mogs OAI**: **Gemini 2.5 Pro Experimental**, codenamed *Nebula*, achieved the #1 spot on the [LMArena leaderboard](https://lmarena.ai/?lea), outperforming **Grok-3/GPT-4.5** by a record-breaking margin and leading in Math, Creative Writing, and Multi-Turn.
   - The model also dominates the [SEAL leaderboards](https://scale.com/leaderboard), with first-place rankings in Humanity’s Last Exam and VISTA (multimodal).
- **4o Image Gen Gives Unsolicited Makeovers**: **GPT-4o's native image generation** is criticized for applying excessive edits, such as *making eyes bigger* and altering facial features, even changing the user's appearance, as demonstrated in [this twitter thread](https://fxtwitter.com/TheXeophon/status/1904602649225285922).
   - Some users reported that altering a single word in their prompts results in failures.
- **AI Studio is for Devs, Gemini Advanced an Ugly ChatGPT Clone**: Members discussed the utility of Google's AI platforms, with some asserting that [AI Studio](https://ai.google.dev/) is better for developers due to its broader model selection, code execution capabilities, and YouTube support.
   - Members felt that [Gemini Advanced](https://gemini.google.com/) *is a bad ChatGPT clone* and *has no reason* to use for power users, and suggested a more streamlined platform focused on developer tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/btibor91/status/1904567494523969647">Tweet from Tibor Blaho (@btibor91)</a>: Looks like Ideogram will be shipping in 24 hours</li><li><a href="https://x.com/lmarena_ai/status/1904581128746656099">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: BREAKING: Gemini 2.5 Pro is now #1 on the Arena leaderboard - the largest score jump ever (+40 pts vs Grok-3/GPT-4.5)! 🏆Tested under codename &#34;nebula&#34;🌌, Gemini 2.5 Pro ranked #1🥇 across ALL...</li><li><a href="https://vxtwitter.com/oriolvinyalsml/status/1904583691566727361">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/natolambert/status/1904599716274594033">Tweet from Nathan Lambert (@natolambert)</a>: okay I bit the bullet and sub&#39;d to gemini advanced, will daily driver 2.5 for a bit</li><li><a href="https://x.com/roramora0/status/1904549463441449182">Tweet from yo (@roramora0)</a>: Since when did Gemini-2.0-Pro start thinking in Cursor? I switched from Sonnet-Thinking to Gemini, but it still kept thinking. I think the model switch isn’t working in Cursor, or does Gemini-2.0-Pro ...</li><li><a href="https://x.com/gdb/status/1904601537487270243">Tweet from Greg Brockman (@gdb)</a>: Native GPT 4o image generation: https://openai.com/index/introducing-4o-image-generation/</li><li><a href="https://fxtwitter.com/TheXeophon/status/1904596173869973664">Tweet from Xeophon (@TheXeophon)</a>: 4o native image gen destroys my eval prompts - or does it?When I change one word in the prompt, it fails them 🙃Quoting Tibor Blaho (@btibor91) &#34;Xeophon-bench&#34;</li><li><a href="https://fxtwitter.com/adcock_brett/status/1904534796770201624">Tweet from Brett Adcock (@adcock_brett)</a>: Say goodbye to the Biden Walk!Figure can now walk naturally like a humanToday we&#39;re introducing learned natural walking</li><li><a href="https://x.com/andrew_n_carr/status/1904607188976611627">Tweet from Andrew Carr (e/🤸) (@andrew_n_carr)</a>: had to share. this is the first time a model has gotten over 50%</li><li><a href="https://x.com/gallabytes/status/1904598264240119974">Tweet from theseriousadult (@gallabytes)</a>: 4o image gen clearly has some kind of multi scale generation setup - seems to commit to low frequency at the beginning then decode high frequency with patch AR.</li><li><a href="https://x.com/scaling01/status/1904599407573819903">Tweet from Lisan al Gaib (@scaling01)</a>: OpenAI after stealing the show from Gemini 2.5 Pro(generated by GPT-4o)</li><li><a href="https://fxtwitter.com/btibor91/status/1904594989780525237">Tweet from Tibor Blaho (@btibor91)</a>: &#34;Xeophon-bench&#34;</li><li><a href="https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data">TAO: Using test-time compute to train efficient LLMs without labeled data</a>: LIFT fine-tunes LLMs without labels using reinforcement learning, boosting performance on enterprise tasks.</li><li><a href="https://x.com/oriolvinyalsml/status/1904583691566727361?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Oriol Vinyals (@OriolVinyalsML)</a>: Introducing Gemini 2.5 Pro Experimental! 🎉Our newest Gemini model has stellar performance across math and science benchmarks. It’s an incredible model for coding and complex reasoning, and it’s #1 on...</li><li><a href="https://x.com/mark_k/status/1904546240332705934">Tweet from Mark Kretschmann (@mark_k)</a>: Google Gemini 2.5 Pro (experimental)</li><li><a href="https://x.com/arcprize/status/1904269307284230593?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from ARC Prize (@arcprize)</a>: Today we are announcing ARC-AGI-2, an unsaturated frontier AGI benchmark that challenges AI reasoning systems (same relative ease for humans).Grand Prize: 85%, ~$0.42/task efficiencyCurrent Performanc...</li><li><a href="https://fxtwitter.com/GrantSlatton/status/1904598054709453276">Tweet from Grant Slatton (@GrantSlatton)</a>: new 4o-image editing does not quite succeed the haircut testIt actually changes the hair really well, but also changes the face to be different enough to not be me anymoreAlso my dog is derpier, and i...</li><li><a href="https://x.com/bedros_p/status/1904619952855822753?s=61">Tweet from Bedros Pamboukian (@bedros_p)</a>: no actually please dont do this</li><li><a href="https://x.com/Angaisb_/status/1904574211802173907">Tweet from angel⭐ (@Angaisb_)</a>: Gemini 2.5 Pro Experimental now available</li><li><a href="https://fxtwitter.com/TheXeophon/status/1904602649225285922">Tweet from Xeophon (@TheXeophon)</a>: what is it with 4o image gen and making eyes bigger??Quoting Lucas Beyer (bl16) (@giffmana) Somehow I&#39;m not getting the &#34;omg Gemini really cooked with this model&#34; results I was hoping for....</li><li><a href="https://x.com/OpenAI/status/1904556394847862809">Tweet from OpenAI (@OpenAI)</a>: no description found</li><li><a href="https://x.com/picocreator/status/1904250680266956903">Tweet from PicoCreator - AI Model Builder 🌉 (@picocreator)</a>: ❗️Attention is NOT all you need ❗️Using only 8 GPU&#39;s (not a cluster), we trained a Qwerky-72B (and 32B), without any transformer attentionWith evals far surpassing GPT 3.5 turbo, and closing in on...</li><li><a href="https://x.com/sundarpichai/status/1904575384466710607">Tweet from Sundar Pichai (@sundarpichai)</a>: Nebula</li><li><a href="https://x.com/alexandr_wang/status/1904590438469951873">Tweet from Alexandr Wang (@alexandr_wang)</a>: 🚨 Gemini 2.5 Pro Exp dropped and it&#39;s now #1 across SEAL leaderboards:🥇 Humanity’s Last Exam🥇 VISTA (multimodal)🥇 (tie) Tool Use🥇 (tie) MultiChallenge (multi-turn)🥉 (tie) Enigma (puzzles)Con...</li><li><a href="https://x.com/swyx/status/1904596926743658840">Tweet from swyx 🌉 (@swyx)</a>: is…this the first time gdm has mogged oai?what an uno reverse card</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/">Gemini 2.5: Our most intelligent AI model</a>: Gemini 2.5 is our most intelligent AI model, now with thinking.</li><li><a href="https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large">🪿Qwerky-72B and 32B : Training large attention free models, with only 8 GPU&#x27;s</a>: ‼️ Attention is NOT all you need ‼️</li><li><a href="https://x.com/teortaxesTex/status/1904362626810872253">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Now we&#39;re talking.0324 is the best non-reasoner model on Misguided Attention, improving by almost 100% over V3.That&#39;s the difference between “eh we only spent 10 grand on post-training cuz we ...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1353842551941107823)** (26 messages🔥): 

> `Labs hillclimb benchmarks, Home inference LLMs, vLLM, OpenRouter, Model Evals` 


- **Labs Actively Hillclimb on Benchmarks**: Labs actively hillclimb on benchmarks like a validation set, according to discussion of **Tulu 3**, where adding *unseen* evals is considered novel.
   - Labs know which websites to include to boost common benchmarks during training.
- **Home Inference Favors vLLM with Caveats**: The best solution for efficient home inference of LLMs with the ability to switch out models on the fly is likely **vLLM**, despite quirks with quant support, though **ollama** is easier to use but lags in support and **SGLang** looks compelling.
   - It was suggested to play around with **llama.cpp** to see how it's going these days.
- **AIBrix from vLLM Team Could Autoscale Models**: A member wondered if something like **AIBrix** from the vLLM team would work, using **vLLM + Kubernetes** to autoscale models to 0 for seamless swapping; [here's an article on that](https://aibrix.github.io/posts/2025-02-20-vllm-control-plane/#advanced-llm-gateway-and-routing-strategies).
   - The smallest viable cluster without quant is two **8xH100** nodes due to vram requirements, though a single **H200** node (**8x141**) would suffice.
- **Model Evals are Surprisingly Time-Consuming**: Internal and external **model evals** are surprisingly time-consuming, requiring a custom UI for making the Eval (**Sonnet + streamlit**) and **a UI to view the data**.
   - It's useful to do **1-2 trials runs with a cheap model** to catch failure modes and re-word/discard prompts.
- **OpenAI API Scaling Requires Caution**: When using **OpenAI**'s API at scale, their dashboard isn't accurate and their spending controls don't work, which may lead to negative balances even with inference.
   - When using **OpenRouter**, you can specify that you want bf16/fp16 for models, as well as setting `max_tokens` and `temperature`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aibrix.github.io/posts/2025-02-20-vllm-control-plane/#advanced-llm-gateway-and-routing-strategies">Introducing AIBrix: Cost-Effective and Scalable Control Plane for vLLM</a>: Open-source large language models (LLMs) like LLaMA, Deepseek, Qwen and Mistral etc have surged in popularity, offering enterprises greater flexibility, cost savings, and control over their AI deploym...</li><li><a href="https://hamel.dev/blog/posts/field-guide/">A Field Guide to Rapidly Improving AI Products – Hamel’s Blog</a>: Evaluation methods, data-driven improvement, and experimentation techniques from 30+ production implementations.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1354141429802209380)** (1 messages): 

> `Political Favor, American Politics, Business Strategy` 


- **Petty Politics Pays for Some Businesses**: It's suggested that for businesses reliant on currying political favor in America currently, *pettiness is probably an asset*.
- **Navigating American Political Climate**: The discussion revolves around how businesses strategically engage with the American political landscape to gain advantages.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1353820322545008750)** (28 messages🔥): 

> `AI Threads lists on social media, Verification as key to AI, MCP malware reverse engineering` 


- **AI Luminary Gains Threads Followers**: A member gained thousands of followers after being added to an *AI threads* list, but finds [Bluesky](https://blueskyweb.xyz/) *slightly better* due to a mix of academic friends.
   - They joked that this earned them a Noam Brown follow.
- **Thinking vs Verifying: Verification, The Key to AI**: A member linked to a thread discussing **verification** as key to AI, quoting Noam Brown's argument that test-time compute is limited by **verification** challenges.
   - Brown used the example of being bottlenecked when trying to recall *When was George Washington born?* if *you don't know, no amount of thinking will get you to the correct answer*.
- **MCP Reverses Malware with AI**: Members linked to a [YouTube video](https://www.youtube.com/watch?v=u2vQapLAW88) showcasing **MCP** for **Ghidra**, enabling LLMs to reverse engineer malware, automating the process with the right prompts.
   - The poster mentioned they *thought it was a bit of a meme at first* but are *starting to see the appeal now that there are actual implementations*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/natolambert/status/1904550785083752718">Tweet from Nathan Lambert (@natolambert)</a>: Verification, The Key to AI Read the archives of Rich Sutton, Turing Award winner :D, has all the major ideasQuoting Noam Brown (@polynoamial) This isn&#39;t quite true. Test-time compute helps when v...</li><li><a href="https://www.youtube.com/watch?v=u2vQapLAW88">ghidraMCP: Now AI Can Reverse Malware</a>: Just built an MCP for Ghidra.Now basically any LLM (Claude, Gemini, local...) can Reverse Engineer malware for you.  With the right prompting, it automates a...</li><li><a href="https://huggingface.co/spaces/Presidentlin/llm-pricing-calculator">Llm Pricing - a Hugging Face Space by Presidentlin</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1354022733780090961)** (52 messages🔥): 

> `Gooning Urban Dictionary meaning, DeepSeek-LLM and DeepSeek-MoE, Mistral Small Versioning Confusion, GPT4o Image Generation vs Gemini, GRPO implementation in trl` 


- ****Gooning** Gets a New, Risque Definition**: Members discussed the evolving meaning of the term *gooning*, with one noting its [new urban dictionary meaning](https://x.com/menhguin/status/1904459726319968262) which differs from its original, more innocent connotation.
   - An older meaning was described as *just like goofing about with your friends, out in the town* during college.
- ****DeepSeek** Model Naming Conventions Clarified**: It was pointed out that the correct naming conventions are **DeepSeek-LLM** and **DeepSeek-MoE**.
   - This correction aims to improve clarity in model identification.
- ****Mistral** Versioning Creates Confusion**: The naming scheme and versioning of **Mistral Small** models came under scrutiny, specifically noting the absence of a **Mistral Small 2**.
   - One member confessed that *i legitimately have no idea how fast the digits boxes actually are for this reason*.
- ****GPT4o** Image Generation Compared to **Gemini****: Members compared the image generation capabilities of **OpenAI's GPT4o** and **Google's Gemini** based on visual outputs.
   - The comparison stemmed from [OpenAI's announcement](https://openai.com/live/) of **4o image generation** in ChatGPT and Sora, inciting memes.
- ****trl**'s **GRPO** Implementation Under Fire**: A member asserted that the **trl GRPO** implementation should not deviate from the **DeepSeek** paper's implementation by default, criticizing deviations from a *known good configuration*.
   - They concluded that *the stdev normalization upweights your easiest and hardest examples and downweights middle of distribution*, potentially biasing results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/deepfates/status/1904596271605907686">Tweet from web weaver (@deepfates)</a>: That&#39;s what i was afraid ofQuoting OpenAI (@OpenAI) 4o image generation in ChatGPT and Sorahttps://openai.com/live/</li><li><a href="https://x.com/menhguin/status/1904459726319968262">Tweet from Minh Nhat Nguyen (@menhguin)</a>: @xlr8harder @natolambert this is peak</li><li><a href="https://x.com/untitled01ipynb/status/1904021116269601097">Tweet from moew (@untitled01ipynb)</a>: @natolambert
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1353889941452492882)** (14 messages🔥): 

> `DPO vs Sampling, RL training and Entropy, SimpleRL-Zoo, DAPO Objective` 


- **Sampling Superior to DPO**: A member suggested that [sampling](https://www.google.com/search?q=sampling) could be better than **DPO**, because *DPO requires data*.
- **Zero RL Training Investigated**: A new project called **SimpleRL-Zoo** was introduced, which deeply investigates zero **RL training** across diverse model families and sizes, including **Llama3-8B**, **Mistral-7B/24B**, **DeepSeekMath-7B**, and **Qwen2.5**.
   - It was shown that starting from base models and only using the correctness reward, the project managed to obtain significant boost on both the accuracy and response length for all the models, detailed in their [paper](https://arxiv.org/abs/2503.18892) and [code](https://github.com/hkust-nlp/simpleRL-reason).
- **DAPO Objective in Paper**: A member mentioned that a paper used the **DAPO** objective.
   - They noted *that was like last week*.



**Link mentioned**: <a href="https://x.com/junxian_he/status/1904527884934697050?s=46">Tweet from Junxian He (@junxian_he)</a>: Two months ago, we open-sourced the first R1-like zero RL training project on math with the Qwen2.5-math model. Since then, many great works performed successful zero RL training, mostly based on Qwen...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1354014717110124575)** (2 messages): 

> `Claude Code, Anthropic, Anysphere's Cursor, Codium's Windsurf, npm package` 


- **Anthropic Releases Claude Code**: Anthropic released **Claude Code**, their competitor to **Anysphere’s Cursor** and **Codium’s Windsurf** as a tool that uses **LLM** as an agent to complete software engineering tasks, as mentioned in [this blog post](https://leehanchung.github.io/blogs/2025/03/07/claude-code/).
- **Digging Under the Hood of Claude Code**: One can download Claude Code's **npm package** as a tarball and unzip it to get the source code.
   - The main control logic of **Claude Code** lives in cli.m.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://leehanchung.github.io/blogs/2025/03/07/claude-code/">Poking Around Claude Code</a>: Discover how Claude Code leverages LLMs as agents for software engineering tasks, including system prompts, model control protocols (MCP), control flow, and hidden features.</li><li><a href="https://www.youtube.com/watch?v=XLaRfZ4AHn8"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1353807028715393156)** (2 messages): 

> `Claude PR, Header Copy Links` 


- **Claude Crafts Copyable Links**: A member shared a [pull request](https://github.com/natolambert/rlhf-book/pull/82) which adds copyable links to all headings that appear on hover in the **rlhf-book**.
- **Header Hover Hyerplinks Help**: These fun header copy links appear when hovering, facilitating easier section linking.
   - The member confirmed it *worked immediately with claude code*.



**Link mentioned**: <a href="https://github.com/natolambert/rlhf-book/pull/82">(experimental) Add heading anchor links for easy section linking by natolambert · Pull Request #82 · natolambert/rlhf-book</a>: Add copyable links to all headings that appear on hoverLinks copy the current URL with fragment identifier to clipboardAdd CSS for styling the anchor linksUpdate Makefile to copy new JS file to ...

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1354105000938442773)** (3 messages): 

> `Anthropic incident, Claude 3.7 Sonnet endpoints, Zero-Token Insurance, Google Gemini 2.5 Pro Experimental` 


- ****Anthropic's Claude goes offline, then back online****: **Claude 3.7 Sonnet endpoints** experienced downtime, with [Anthropic posting updates](https://status.anthropic.com/incidents/89rpts2022hs) starting **March 25, 2025**, and resolving the incident by **8:41 PDT**.
   - The updates indicated that the downtime was due to maintenance and efforts to improve systems.
- ****OpenRouter offers Zero-Token Insurance****: OpenRouter is now offering **zero-token insurance coverage** to all models on the platform, potentially saving users over **$18,000 per week**.
   - As [OpenRouterAI stated](https://x.com/OpenRouterAI/status/1904567846975201766), users will not be charged for responses with **no output tokens** and a **blank or error finish reason**, even if the provider still charges for prompt processing.
- ****Gemini 2.5 Pro Experimental Goes Live****: Google's **Gemini 2.5 Pro Experimental**, a state-of-the-art model capable of advanced reasoning, coding, and mathematical tasks, is now available as a free model on [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free).
   - Gemini 2.5 Pro has **1,000,000 context**, and achieves top-tier performance on benchmarks like the **LMArena leaderboard**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini Pro 2.5 Experimental (free) - API, Providers, Stats</a>: Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. Run Gemini Pro 2.5 Experimental (free) with API</li><li><a href="https://x.com/OpenRouterAI/status/1904567846975201766">Tweet from OpenRouter (@OpenRouterAI)</a>: As the first and largest LLM router, we&#39;ve seen virtually every possible quality issue from model providers and think there&#39;s a lot that can be done to make the ecosystem more friendly.Startin...</li><li><a href="https://status.anthropic.com/incidents/89rpts2022hs">Elevated errors on Claude.ai and Console</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1353805705492037743)** (196 messages🔥🔥): 

> `Deepseek performance issues, Gemini 2.5 Pro release and benchmarks, Provisioning API keys for user management, OpenRouter activity log retention, GPT-4o image generation API support` 


- **DeepSeek Suffers Server Struggles**: Users reported **DeepSeek** is becoming *borderline unusable* due to overcrowded servers, suggesting a need for price adjustments to manage demand.
   - Members speculated whether the issues were related to peak usage times in China, but no direct solution was found beyond hoping for better hardware availability from Huawei.
- **Gemini 2.5 Pro Wows Early Testers**: **Gemini 2.5 Pro Experimental** is now available at the API, with early testers impressed by its capabilities, especially in reasoning and coding, as detailed in [Google's blog post](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning).
   - One user exclaimed that **Gemini 2.5 Pro** *destroys* **ChatGPT o3-mini-high**, prompting discussions about whether its performance boost is solely due to benchmarks or reflects true improvements.
- **Provisioning API Keys Offer Granular User Control**: OpenRouter now offers **provisioning API keys**, enabling developers to programmatically manage API keys for their users, set limits, and track spend, enhancing scalability and control, as documented [here](https://openrouter.ai/docs/features/provisioning-api-keys).
   - This allows developers to create unique API keys for each user, streamlining billing and access management within their own platforms.
- **API Key Activity Logs Retained Infinitely**: OpenRouter retains **API key activity logs forever**, allowing users to monitor usage per key, aiding in team environment evaluation and usage tracking.
   - This feature addresses the need for streaming and visualizing API usage, providing detailed insights into each member's consumption patterns.
- **GPT-4o Image Generation API on OpenRouter's Radar**: Following the rollout of **GPT-4o's native image generation**, OpenRouter is actively developing API functionality for image generation calls to provide users access to equivalent functionalities without directly applying for individual APIs.
   - This move aims to keep OpenRouter competitive and comprehensive, addressing the need for seamless integration of cutting-edge image generation capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/settings/integrations">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://starvector.github.io/">StarVector</a>: no description found</li><li><a href="https://openrouter.ai/docs/features/provisioning-api-keys">Provisioning API Keys - Programmatic Control of OpenRouter API Keys</a>: Manage OpenRouter API keys programmatically through dedicated management endpoints. Create, read, update, and delete API keys for automated key distribution and control.</li><li><a href="https://time.is/china">Time in China now - Time.is</a>: no description found</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">Reasoning Tokens - Improve AI Model Decision Making</a>: Learn how to use reasoning tokens to enhance AI model outputs. Implement step-by-step reasoning traces for better decision making and transparency.</li><li><a href="https://openrouter.ai/docs/api-reference/api-keys/get-api-key">Get API key — OpenRouter | Documentation</a>: Returns details about a specific API key. Requires a Provisioning API key.</li><li><a href="https://openrouter.ai/docs/api-reference/api-keys/list-api-keys">List API keys — OpenRouter | Documentation</a>: Returns a list of all API keys associated with the account. Requires a Provisioning API key.</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning">Gemini 2.5: Our most intelligent AI model</a>: Gemini 2.5 is our most intelligent AI model, now with thinking.</li><li><a href="https://www.reddit.com/r/singularity/comments/1jizn0t/newupdated_models_by_google_soon/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://fireworks.ai/models/fireworks/deepseek-v3-0324">Fireworks - Fastest Inference for Generative AI</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks AI!
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1353805711129186305)** (148 messages🔥🔥): 

> `Bits per Weight (BPW), Model Capacity Scaling, DeepSeek V3, Gemini 2.5 Pro, AI IDE Evaluation` 


- **BPW Sweet Spot Between 4-5!**: Experiments show **model capacity** collapses below 4 **bits per weight (BPW)**, but deviates above 5, implying **optimal weight usage** at 4 BPW for given training flops.
   - Increasing training epochs helps 5 BPW models approach the curve, raising BPW at the cost of FLOPS, visualized via [2L and 3L MLP trained on MNIST](https://cdn.discordapp.com/attachments/1149866623109439599/1353808843720626288/image.png).
- **DeepSeek V3 Shows Reasoning Prowess!**: **DeepSeek V3-0324** can act as a reasoning model, detect thought iterations, and verify solution existence indirectly, rivalling **O1** in performance based on attached [prompt](https://cdn.discordapp.com/attachments/1149866623109439599/1353830450493132800/prompt.txt).
- **Consulting Services for Hermes Models**: A member inquired about consulting for fine-tuning **Hermes** for specific ERP use cases, with others noting the diversity and specialization within the ERP domain.
   - The pygmalion folks and people connected to them probably can help according to [this tenor link](https://tenor.com/view/daspoody-sleep-sleepy-wake-woke-gif-25698451).
- **Google's Gemini 2.5 Pro Enters the Arena**: **Gemini 2.5 Pro Experimental** leads common benchmarks and debuts at #1 on [LMArena](https://lmarena.ai/?lea) showcasing strong reasoning and code capabilities.
   - It can also manages to terminate for infinite thought loops for some prompts and is a daily model drop as noted in this [blogpost](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/).
- **Community Awaits DeepSeek V3 Lite's Arrival**: Community members speculate on a potential **DeepSeek V3 Lite** release after the Qwen 3 MoE model.
   - A member offered anonymized human-AI data from their project if it aids NousResearch's efforts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/fabianstelzer/status/1904629831125656050">Tweet from fabian (@fabianstelzer)</a>: GPT-4.5, “create a complex multi panel manga on your condition - be honest”</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B-FP8">NousResearch/Hermes-3-Llama-3.1-70B-FP8 · Hugging Face</a>: no description found</li><li><a href="https://docs.sglang.ai/backend/quantization.html">Quantization &#8212; SGLang</a>: no description found</li><li><a href="https://tenor.com/view/daspoody-sleep-sleepy-wake-woke-gif-2569845121217246002">Daspoody Sleep GIF - Daspoody Sleep Sleepy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/">Gemini 2.5: Our most intelligent AI model</a>: Gemini 2.5 is our most intelligent AI model, now with thinking.</li><li><a href="https://artificialanalysis.ai/">AI Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key performance metrics including quality, price, output speed &amp; latency.</li><li><a href="https://www.youtube.com/watch?v=-u3ye--VlPo">Jack Ma&#39;s Ant Group Uses Chinese-Made Chips to Train AI Models</a>: Bloomberg has learned that Jack Ma-backed Ant Group has used Chinese-made semiconductors to train AI models. Ant Group claims this would help cut costs by 20...</li><li><a href="https://github.com/grahamannett/ai-ide-compare">GitHub - grahamannett/ai-ide-compare</a>: Contribute to grahamannett/ai-ide-compare development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.model_max_length">Tokenizer</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1353810491142832229)** (26 messages🔥): 

> `Add and Sigmoid vs Add and Norm, Scaling Experts at Inference Time, Transformers without Normalization, LLM-Emulated Raytracing, Indirect Image Generation with LLMs` 


- **Add and Sigmoid: The New Norm?**: A member suggested replacing `add` and `norm` with `add` and **sigmoid** in transformer architectures, particularly for Mixture of Experts (**MoE**), to facilitate easier scaling of experts.
   - Theorizing that *if the gate has a sigmoid activation function, this should allow us to add or remove other experts without much problems* because the scaling becomes independent of the number of neighboring values.
- **Transformers get TANHed**: Drawing from the recent [Transformers without Normalization paper](https://arxiv.org/abs/2302.05442), one member noted that replacing normalization with **tanh** is a viable strategy.
   - The concern raised was the impact on smaller weights when removing experts at inference time, but another countered that the **top_k** gate mechanism would still function effectively by selecting from the remaining experts.
- **Raytracing Emulated by LLMs, no NVIDIA code involved**: Members discussed the idea of using an **LLM** to emulate a **raytracing algorithm**, clarifying that the current implementation involves a **Python** program written by the LLM to generate images indirectly.
   - It's *next level text to image generation* because the LLM writes the program rather than generating the image directly, the programs are available in this [GitHub repo](https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer).
- **Normalization Debate: Sigmoid Scales Without Neighbors**: The discussion continued on the need to remove normalization when **top_k** is constant, with one member arguing that normalization changes values relative to neighboring values.
   - They explained that using **sigmoid** for scaling would avoid this dependency, enabling the addition of more experts without significantly altering existing values.



**Link mentioned**: <a href="https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer">llmbenchmark/raytracer at master · cpldcpu/llmbenchmark</a>: Various LLM Benchmarks. Contribute to cpldcpu/llmbenchmark development by creating an account on GitHub.

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1353826338984693900)** (136 messages🔥🔥): 

> `Reve Image, Qwen 2.5, ARC-AGI, 11x fraud, Zep knowledge graphs` 


- **Reve Image dethrones SOTA Text-to-Image**: A new image generation model, **Reve Image**, has been released and is already being hailed as a leader in the field, outperforming models like **Recraft V3**, **Google's Imagen 3**, **Midjourney v6.1**, and **Black Forest Lab's FLUX.1.1 [pro]**.
   - It has been noted for its impressive **text rendering, prompt adherence, and aesthetics**, and is currently accessible through [Reve’s website](https://www.reveimage.com/) without an API.
- **DeepMind's Gemini 2.5 Pro debuts in Arena**: **Gemini 2.5 Pro** is now **#1** on the Arena leaderboard, achieving the largest score jump ever (**+40 pts vs Grok-3/GPT-4.5**), [as announced by LM Arena](https://x.com/lmarena_ai/status/1904581128746656099).
   - The model, tested under the codename *nebula*, uniquely leads in categories like **Math, Creative Writing, Instruction Following, Longer Query, and Multi-Turn**.
- **OpenAI launches Image Gen in ChatGPT 4o**: **OpenAI** has launched native image generation in **ChatGPT**, with [Sam Altman noting](https://x.com/sama/status/1904598788687487422) it as an incredible technology and product.
   - Early testers, like [@krishnanrohit](https://x.com/krishnanrohit/status/1904602460020445543), say this is the best image gen and editing tool they've tried so far, praising its ability to make and edit multiple characters correctly, especially when it's two or more dinosaurs.
- **AI SDR Startup 11x Under Scrutiny for Claiming Phony Customers**: AI-powered sales automation startup **11x**, backed by **a16z** and **Benchmark**, is under fire for allegedly claiming customers it doesn’t have, according to a [TechCrunch report](https://techcrunch.com/2025/03/24/a16z-and-benchmark-backed-11x-has-been-claiming-customers-it-doesnt-have/).
   - Despite a spokesperson for **Andreessen Horowitz** denying rumors of legal action, concerns persist about **11x’s** financial status and potentially inflated revenue figures, with some suggesting the company's growth relies on hype and that real enterprise sales leaders are not seeing value in AI SDRs.
- **Databricks Unveils TAO to Tune LLMs Without Labels**: The **Databricks** research team announced **TAO**, a method to tune LLMs for a task without data labels, using test-time compute and RL, [as described in their blog](https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data).
   - They claim it outperforms supervised fine-tuning and scales with compute to produce fast, high-quality models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025">Announcing ARC-AGI-2 and ARC Prize 2025</a>: Measuring the next level of intelligence with ARC-AGI-2 and ARC Prize 2025</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-vl-32b/">Qwen2.5-VL-32B: Smarter and Lighter</a>: QWEN CHAT GITHUB HUGGING FACE MODELSCOPE DISCORDIntroduction At the end of January this year, we launched the Qwen2.5-VL series of models, which received widespread attention and positive feedback fro...</li><li><a href="https://fxtwitter.com/OpenAI/status/1904556394847862809)">Tweet from OpenAI (@OpenAI)</a>: no description found</li><li><a href="https://x.com/OpenAI/status/1904556394847862809>)">Tweet from OpenAI (@OpenAI)</a>: no description found</li><li><a href="https://fxtwitter.com/hasan_sukkar_/status/1904408098212806804)">Tweet from Hasan (@hasan_sukkar_)</a>: no description found</li><li><a href="https://x.com/hasan_sukkar_/status/1904408098212806804>)">Tweet from Hasan (@hasan_sukkar_)</a>: no description found</li><li><a href="https://fxtwitter.com/fofrai/status/1904476544443040212)">Tweet from fofr (@fofrAI)</a>: Trying something similar with ReveQuoting fofr (@fofrAI) &gt; the word “PIKA” where the shapes of all the letters combine to form a pikachu using only clever and artistic typography, beautifully desig...</li><li><a href="https://x.com/fofrai/status/1904476544443040212>)">Tweet from fofr (@fofrAI)</a>: Trying something similar with ReveQuoting fofr (@fofrAI) &gt; the word “PIKA” where the shapes of all the letters combine to form a pikachu using only clever and artistic typography, beautifully desig...</li><li><a href="https://fxtwitter.com/gdb/status/1904601537487270243)">Tweet from Greg Brockman (@gdb)</a>: Native GPT 4o image generation: https://openai.com/index/introducing-4o-image-generation/</li><li><a href="https://x.com/gdb/status/1904601537487270243>)">Tweet from Greg Brockman (@gdb)</a>: Native GPT 4o image generation: https://openai.com/index/introducing-4o-image-generation/</li><li><a href="https://fxtwitter.com/kipperrii/status/1904615542474105305)">Tweet from kipply (@kipperrii)</a>: openai needs to get the full-res version of this (post only has webp) and print posters of this immediately!!!!</li><li><a href="https://x.com/kipperrii/status/1904615542474105305>)">Tweet from kipply (@kipperrii)</a>: openai needs to get the full-res version of this (post only has webp) and print posters of this immediately!!!!</li><li><a href="https://fxtwitter.com/matei_zaharia/status/1904587809945772124)">Tweet from Matei Zaharia (@matei_zaharia)</a>: Really cool result from the Databricks research team: You can tune LLMs for a task *without data labels*, using test-time compute and RL, and outperform supervised fine-tuning! Our new TAO method scal...</li><li><a href="https://x.com/matei_zaharia/status/1904587809945772124>)">Tweet from Matei Zaharia (@matei_zaharia)</a>: Really cool result from the Databricks research team: You can tune LLMs for a task *without data labels*, using test-time compute and RL, and outperform supervised fine-tuning! Our new TAO method scal...</li><li><a href="https://fxtwitter.com/danshipper/status/1904594300232495230)">Tweet from Dan Shipper 📧 (@danshipper)</a>: OpenAI just launched native image generation in @ChatGPTapp and it is INSANEWe&#39;ve been testing it internally @every for a few weeks—and it&#39;s by far the best image model I&#39;ve tried:1. It&#3...</li><li><a href="https://x.com/danshipper/status/1904594300232495230>)">Tweet from Dan Shipper 📧 (@danshipper)</a>: OpenAI just launched native image generation in @ChatGPTapp and it is INSANEWe&#39;ve been testing it internally @every for a few weeks—and it&#39;s by far the best image model I&#39;ve tried:1. It&#3...</li><li><a href="https://en.wikipedia.org/wiki/Blackboard_(design_pattern)">Blackboard (design pattern) - Wikipedia</a>: no description found</li><li><a href="https://fxtwitter.com/fofrai/status/1904331135859015703)">Tweet from fofr (@fofrAI)</a>: I haven&#39;t had this much fun with an image model in a long time 🔥</li><li><a href="https://x.com/fofrai/status/1904331135859015703>)">Tweet from fofr (@fofrAI)</a>: I haven&#39;t had this much fun with an image model in a long time 🔥</li><li><a href="https://fxtwitter.com/dzhng/status/1904412968114356604)">Tweet from David (@dzhng)</a>: We&#39;ve talked to many ex customers of this co, the janky accounting are well known. Not the only co in the space doing this as well.The only way they can sustain growth is hype. So far I&#39;ve met...</li><li><a href="https://x.com/dzhng/status/1904412968114356604>)">Tweet from David (@dzhng)</a>: We&#39;ve talked to many ex customers of this co, the janky accounting are well known. Not the only co in the space doing this as well.The only way they can sustain growth is hype. So far I&#39;ve met...</li><li><a href="https://fxtwitter.com/iScienceLuvr/status/1904604991832416745)">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Wait GPT-4o can just one-shot stuff like this?! That&#39;s impressive...</li><li><a href="https://x.com/iScienceLuvr/status/1904604991832416745>)">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Wait GPT-4o can just one-shot stuff like this?! That&#39;s impressive...</li><li><a href="https://fxtwitter.com/newsystems_/status/1904577550690771050)">Tweet from New (@newsystems_)</a>: It&#39;s finally here: BramptonBrampton is the world&#39;s most intelligent, creative, and fastest model. Brampton dramatically outperforms Grok 3, Claude 3.7 Sonnet, and GPT 4.5. Reply with &#34;bram...</li><li><a href="https://x.com/newsystems_/status/1904577550690771050>)">Tweet from New (@newsystems_)</a>: It&#39;s finally here: BramptonBrampton is the world&#39;s most intelligent, creative, and fastest model. Brampton dramatically outperforms Grok 3, Claude 3.7 Sonnet, and GPT 4.5. Reply with &#34;bram...</li><li><a href="https://fxtwitter.com/paulgauthier/status/1904637913411031410)">Tweet from Paul Gauthier (@paulgauthier)</a>: Gemini 2.5 Pro sets SOTA on the aider polyglot leaderboard with a score of 73%.This is well ahead of thinking/reasoning models. A huge jump from prior Gemini models. The first Gemini model to effectiv...</li><li><a href="https://x.com/paulgauthier/status/1904637913411031410>)">Tweet from Paul Gauthier (@paulgauthier)</a>: Gemini 2.5 Pro sets SOTA on the aider polyglot leaderboard with a score of 73%.This is well ahead of thinking/reasoning models. A huge jump from prior Gemini models. The first Gemini model to effectiv...</li><li><a href="https://fxtwitter.com/OfficialLoganK/status/1899914266062577722)">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Introducing YouTube video 🎥 link support in Google AI Studio and the Gemini API. You can now directly pass in a YouTube video and the model can usage its native video understanding capabilities to us...</li><li><a href="https://x.com/OfficialLoganK/status/1899914266062577722>)">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Introducing YouTube video 🎥 link support in Google AI Studio and the Gemini API. You can now directly pass in a YouTube video and the model can usage its native video understanding capabilities to us...</li><li><a href="https://fxtwitter.com/kevinweil/status/1904596007650025787)">Tweet from Kevin Weil 🇺🇸 (@kevinweil)</a>: A few months ago, I asked ChatGPT &#34;based on what you know about me, draw me a picture of what you think my current life looks like.&#34; Below is that image, and what I got from the same prompt to...</li><li><a href="https://x.com/kevinweil/status/1904596007650025787>)">Tweet from Kevin Weil 🇺🇸 (@kevinweil)</a>: A few months ago, I asked ChatGPT &#34;based on what you know about me, draw me a picture of what you think my current life looks like.&#34; Below is that image, and what I got from the same prompt to...</li><li><a href="https://fxtwitter.com/tenobrus/status/1904422446389706905)">Tweet from Tenobrus (@tenobrus)</a>: both cursor and windsurf are absolutely bleeding VC money on every call btwQuoting Nityesh (@nityeshaga) idk why no one is talking about it but @windsurf_ai gives you almost 4x more ai usage than @cur...</li><li><a href="https://x.com/tenobrus/status/1904422446389706905>)">Tweet from Tenobrus (@tenobrus)</a>: both cursor and windsurf are absolutely bleeding VC money on every call btwQuoting Nityesh (@nityeshaga) idk why no one is talking about it but @windsurf_ai gives you almost 4x more ai usage than @cur...</li><li><a href="https://fxtwitter.com/andrew_n_carr/status/1904607188976611627)">Tweet from Andrew Carr (e/🤸) (@andrew_n_carr)</a>: had to share. this is the first time a model has gotten over 50%</li><li><a href="https://x.com/andrew_n_carr/status/1904607188976611627>)">Tweet from Andrew Carr (e/🤸) (@andrew_n_carr)</a>: had to share. this is the first time a model has gotten over 50%</li><li><a href="https://fxtwitter.com/GoogleDeepMind/status/1904579660740256022)">Tweet from Google DeepMind (@GoogleDeepMind)</a>: Think you know Gemini? 🤔 Think again.Meet Gemini 2.5: our most intelligent model 💡 The first release is Pro Experimental, which is state-of-the-art across many benchmarks - meaning it can handle com...</li><li><a href="https://x.com/GoogleDeepMind/status/1904579660740256022>)">Tweet from Google DeepMind (@GoogleDeepMind)</a>: Think you know Gemini? 🤔 Think again.Meet Gemini 2.5: our most intelligent model 💡 The first release is Pro Experimental, which is state-of-the-art across many benchmarks - meaning it can handle com...</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning">Gemini 2.5: Our most intelligent AI model</a>: Gemini 2.5 is our most intelligent AI model, now with thinking.</li><li><a href="https://fxtwitter.com/gallabytes/status/1904598264240119974)">Tweet from theseriousadult (@gallabytes)</a>: 4o image gen clearly has some kind of multi scale generation setup - seems to commit to low frequency at the beginning then decode high frequency with patch AR.</li><li><a href="https://x.com/gallabytes/status/1904598264240119974>)">Tweet from theseriousadult (@gallabytes)</a>: 4o image gen clearly has some kind of multi scale generation setup - seems to commit to low frequency at the beginning then decode high frequency with patch AR.</li><li><a href="https://fxtwitter.com/fofrai/status/1904321572078223400)">Tweet from fofr (@fofrAI)</a>: no description found</li><li><a href="https://x.com/fofrai/status/1904321572078223400>)">Tweet from fofr (@fofrAI)</a>: no description found</li><li><a href="https://fxtwitter.com/taesung/status/1904220824435032528)">Tweet from Taesung Park (@Taesung)</a>: Excited to come out of stealth at @reveimage!Today&#39;s text-to-image/video models, in contrast to LLMs, lack logic. Images seem plausible initially but fall apart under scrutiny: painting techniques...</li><li><a href="https://x.com/taesung/status/1904220824435032528>)">Tweet from Taesung Park (@Taesung)</a>: Excited to come out of stealth at @reveimage!Today&#39;s text-to-image/video models, in contrast to LLMs, lack logic. Images seem plausible initially but fall apart under scrutiny: painting techniques...</li><li><a href="https://fxtwitter.com/fchollet/status/1904267900963475807)">Tweet from François Chollet (@fchollet)</a>: A crucial point that everyone should be internalizing: in the age of test-time search, it&#39;s pretty much always possible to reach any level of capability by simply expending more compute.So it’s no...</li><li><a href="https://x.com/fchollet/status/1904267900963475807>)">Tweet from François Chollet (@fchollet)</a>: A crucial point that everyone should be internalizing: in the age of test-time search, it&#39;s pretty much always possible to reach any level of capability by simply expending more compute.So it’s no...</li><li><a href="https://fxtwitter.com/fofrAI/status/1904320703333126545)">Tweet from fofr (@fofrAI)</a>: Reve also did really well with this text. It&#39;s interesting how it trails off at the end in the last four lines.Quoting fofr (@fofrAI) 😍</li><li><a href="https://x.com/fofrAI/status/1904320703333126545>)">Tweet from fofr (@fofrAI)</a>: Reve also did really well with this text. It&#39;s interesting how it trails off at the end in the last four lines.Quoting fofr (@fofrAI) 😍</li><li><a href="https://fxtwitter.com/TransluceAI/status/1904226873879806390)">Tweet from Transluce (@TransluceAI)</a>: To interpret AI benchmarks, we need to look at the data.Top-level numbers don&#39;t mean what you think: there may be broken tasks, unexpected behaviors, or near-misses.We&#39;re introducing Docent to...</li><li><a href="https://x.com/TransluceAI/status/1904226873879806390>)">Tweet from Transluce (@TransluceAI)</a>: To interpret AI benchmarks, we need to look at the data.Top-level numbers don&#39;t mean what you think: there may be broken tasks, unexpected behaviors, or near-misses.We&#39;re introducing Docent to...</li><li><a href="https://fxtwitter.com/sama/status/1904598788687487422)">Tweet from Sam Altman (@sama)</a>: we are launching a new thing today—images in chatgpt!two things to say about it:1. it&#39;s an incredible technology/product. i remember seeing some of the first images come out of this model and havi...</li><li><a href="https://x.com/sama/status/1904598788687487422>)">Tweet from Sam Altman (@sama)</a>: we are launching a new thing today—images in chatgpt!two things to say about it:1. it&#39;s an incredible technology/product. i remember seeing some of the first images come out of this model and havi...</li><li><a href="https://fxtwitter.com/sherwinwu/status/1904620108389212413)">Tweet from Sherwin Wu (@sherwinwu)</a>: My top use case for GPT-4o native image output right now: using it to imagine home reno projects in our new place. i.e. look at this potential built-in bookshelf / reading bench — rendered in one shot...</li><li><a href="https://x.com/sherwinwu/status/1904620108389212413>)">Tweet from Sherwin Wu (@sherwinwu)</a>: My top use case for GPT-4o native image output right now: using it to imagine home reno projects in our new place. i.e. look at this potential built-in bookshelf / reading bench — rendered in one shot...</li><li><a href="https://fxtwitter.com/ajabri/status/1904599427366739975)">Tweet from Allan Jabri (@ajabri)</a>: the pros and cons</li><li><a href="https://x.com/ajabri/status/1904599427366739975>)">Tweet from Allan Jabri (@ajabri)</a>: the pros and cons</li><li><a href="https://fxtwitter.com/fofrAI/status/1904284550156685474)">Tweet from fofr (@fofrAI)</a>: Reve seems promising when it comes to artistic styles and composition.It managed to make some impasto eyes, the eyes will normally always fail. I also really liked the composition of the second image.</li><li><a href="https://x.com/fofrAI/status/1904284550156685474>)">Tweet from fofr (@fofrAI)</a>: Reve seems promising when it comes to artistic styles and composition.It managed to make some impasto eyes, the eyes will normally always fail. I also really liked the composition of the second image.</li><li><a href="https://alignment.anthropic.com/2025/automated-researchers-sandbag/">no title found</a>: no description found</li><li><a href="https://fxtwitter.com/osanseviero/status/1904561836835602776)">Tweet from Omar Sanseviero (@osanseviero)</a>: Introducing 🥁🥁TxGemma!🧪LLM for multiple therapeutic tasks for drug development🤏2B, 9B, and 27B🤗Fine-tunable with transformers🤖Agentic-Tx for agentic systemsBlog: https://developers.googleblog.co...</li><li><a href="https://x.com/osanseviero/status/1904561836835602776>)">Tweet from Omar Sanseviero (@osanseviero)</a>: Introducing 🥁🥁TxGemma!🧪LLM for multiple therapeutic tasks for drug development🤏2B, 9B, and 27B🤗Fine-tunable with transformers🤖Agentic-Tx for agentic systemsBlog: https://developers.googleblog.co...</li><li><a href="https://fxtwitter.com/gasteigerjo/status/1904562825520906462)">Tweet from Johannes Gasteiger, né Klicpera (@gasteigerjo)</a>: New Anthropic blog post: Subtle sabotage in automated researchers.As AI systems increasingly assist with AI research, how do we ensure they&#39;re not subtly sabotaging that research? We show that mal...</li><li><a href="https://x.com/gasteigerjo/status/1904562825520906462>)">Tweet from Johannes Gasteiger, né Klicpera (@gasteigerjo)</a>: New Anthropic blog post: Subtle sabotage in automated researchers.As AI systems increasingly assist with AI research, how do we ensure they&#39;re not subtly sabotaging that research? We show that mal...</li><li><a href="https://fxtwitter.com/dimitrispapail/status/1904560078012686670)">Tweet from Dimitris Papailiopoulos (@DimitrisPapail)</a>: arc-agi (even the 1st) has been and still is overhyped. the 2nd seems even less interesting, since 1) it&#39;s adversarially designed 2) there is no clear reason why one would expect it&#39;s not solv...</li><li><a href="https://x.com/dimitrispapail/status/1904560078012686670>)">Tweet from Dimitris Papailiopoulos (@DimitrisPapail)</a>: arc-agi (even the 1st) has been and still is overhyped. the 2nd seems even less interesting, since 1) it&#39;s adversarially designed 2) there is no clear reason why one would expect it&#39;s not solv...</li><li><a href="https://fxtwitter.com/skirano/status/1904609866099933272)">Tweet from Pietro Schirano (@skirano)</a>: New image model from OpenAI is pretty good at UI stuff.</li><li><a href="https://x.com/skirano/status/1904609866099933272>)">Tweet from Pietro Schirano (@skirano)</a>: New image model from OpenAI is pretty good at UI stuff.</li><li><a href="https://fxtwitter.com/sama/status/1904599358756315341)">Tweet from Sam Altman (@sama)</a>: this was a real labor of love from @gabeeegoooh. congrats gabe; excellent work!here is what we generated during the livestream:</li><li><a href="https://x.com/sama/status/1904599358756315341>)">Tweet from Sam Altman (@sama)</a>: this was a real labor of love from @gabeeegoooh. congrats gabe; excellent work!here is what we generated during the livestream:</li><li><a href="https://softwareengineeringdaily.com/2025/03/25/knowledge-graphs-as-agentic-memory-with-daniel-chalef/">Knowledge Graphs as Agentic Memory with Daniel Chalef - Software Engineering Daily</a>: Contextual memory in AI is a major challenge because current models struggle to retain and recall relevant information over time. While humans can build long-term semantic relationships, AI systems of...</li><li><a href="https://fxtwitter.com/scaling01/status/1904603736657305862)">Tweet from Lisan al Gaib (@scaling01)</a>: DALLE vs GPT-4o image genOpenAI cookedQuoting Lisan al Gaib (@scaling01) GPT-4o with image generation is actually insane.These are not real images!</li><li><a href="https://x.com/scaling01/status/1904603736657305862>)">Tweet from Lisan al Gaib (@scaling01)</a>: DALLE vs GPT-4o image genOpenAI cookedQuoting Lisan al Gaib (@scaling01) GPT-4o with image generation is actually insane.These are not real images!</li><li><a href="https://fxtwitter.com/dwarkesh_sp/status/1904551410219524218)">Tweet from Dwarkesh Patel (@dwarkesh_sp)</a>: I&#39;m so pleased to present a new book with @stripepress: &#34;The Scaling Era: An Oral History of AI, 2019-2025.&#34;Over the last few years, I interviewed the key people thinking about AI: scienti...</li><li><a href="https://x.com/dwarkesh_sp/status/1904551410219524218>)">Tweet from Dwarkesh Patel (@dwarkesh_sp)</a>: I&#39;m so pleased to present a new book with @stripepress: &#34;The Scaling Era: An Oral History of AI, 2019-2025.&#34;Over the last few years, I interviewed the key people thinking about AI: scienti...</li><li><a href="https://fxtwitter.com/phill__1/status/1904590165256839526)">Tweet from Phil (@phill__1)</a>: Wow, 4o image generation is cool!</li><li><a href="https://x.com/phill__1/status/1904590165256839526>)">Tweet from Phil (@phill__1)</a>: Wow, 4o image generation is cool!</li><li><a href="https://fxtwitter.com/krishnanrohit/status/1904602460020445543)">Tweet from rohit (@krishnanrohit)</a>: Had early access to this, and it&#39;s the best image gen and editing tool that I&#39;ve tried so far. It&#39;s the first (and only) one which can make and edit multiple characters correctly, especial...</li><li><a href="https://x.com/krishnanrohit/status/1904602460020445543>)">Tweet from rohit (@krishnanrohit)</a>: Had early access to this, and it&#39;s the best image gen and editing tool that I&#39;ve tried so far. It&#39;s the first (and only) one which can make and edit multiple characters correctly, especial...</li><li><a href="https://fxtwitter.com/gabeeegoooh/status/1904596565286858913)">Tweet from Gabriel Goh (@gabeeegoooh)</a>: this is ready now for the world</li><li><a href="https://x.com/gabeeegoooh/status/1904596565286858913>)">Tweet from Gabriel Goh (@gabeeegoooh)</a>: this is ready now for the world</li><li><a href="https://fxtwitter.com/fofrai/status/1904318387120988207)">Tweet from fofr (@fofrAI)</a>: Quoting fofr (@fofrAI) 😍</li><li><a href="https://x.com/fofrai/status/1904318387120988207>)">Tweet from fofr (@fofrAI)</a>: Quoting fofr (@fofrAI) 😍</li><li><a href="https://fxtwitter.com/ilanbigio/status/1904601953063362871)">Tweet from ilan bigio (@ilanbigio)</a>: 4o image generation - such a game changer&#34;make me carry a really heavy chrome version of the openai logo. and give me sunglasses&#34;Quoting OpenAI (@OpenAI) 4o image generation in ChatGPT and Sor...</li><li><a href="https://x.com/ilanbigio/status/1904601953063362871>)">Tweet from ilan bigio (@ilanbigio)</a>: 4o image generation - such a game changer&#34;make me carry a really heavy chrome version of the openai logo. and give me sunglasses&#34;Quoting OpenAI (@OpenAI) 4o image generation in ChatGPT and Sor...</li><li><a href="https://fxtwitter.com/willccbb/status/1904620335028146544)">Tweet from will brown (@willccbb)</a>: the fact that 1000+ people have commented brampton and the only post even jokingly claiming to show the actual model is just a guy sysprompting ollama to use toronto slang is super bearish on this bei...</li><li><a href="https://x.com/willccbb/status/1904620335028146544>)">Tweet from will brown (@willccbb)</a>: the fact that 1000+ people have commented brampton and the only post even jokingly claiming to show the actual model is just a guy sysprompting ollama to use toronto slang is super bearish on this bei...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1904581128746656099)">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: BREAKING: Gemini 2.5 Pro is now #1 on the Arena leaderboard - the largest score jump ever (+40 pts vs Grok-3/GPT-4.5)! 🏆Tested under codename &#34;nebula&#34;🌌, Gemini 2.5 Pro ranked #1🥇 across ALL...</li><li><a href="https://x.com/lmarena_ai/status/1904581128746656099>)">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: BREAKING: Gemini 2.5 Pro is now #1 on the Arena leaderboard - the largest score jump ever (+40 pts vs Grok-3/GPT-4.5)! 🏆Tested under codename &#34;nebula&#34;🌌, Gemini 2.5 Pro ranked #1🥇 across ALL...</li><li><a href="https://fxtwitter.com/fofrAI/status/1904619844349420005)">Tweet from fofr (@fofrAI)</a>: I tried this prompt on sora 👌Quoting fofr (@fofrAI) When it&#39;s stand up time at Replicate</li><li><a href="https://x.com/fofrAI/status/1904619844349420005>)">Tweet from fofr (@fofrAI)</a>: I tried this prompt on sora 👌Quoting fofr (@fofrAI) When it&#39;s stand up time at Replicate</li><li><a href="https://en.wikipedia.org/wiki/Blackboard_system">Blackboard system - Wikipedia</a>: no description found</li><li><a href="https://fxtwitter.com/">Tweet from GitHub - FxEmbed/FxEmbed: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FxEmbed/FxEmbed</li><li><a href="https://fxtwitter.com/taesung/">Tweet from GitHub - FxEmbed/FxEmbed: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FxEmbed/FxEmbed</li><li><a href="https://techcrunch.com/2025/03/24/a16z-and-benchmark-backed-11x-has-been-claiming-customers-it-doesnt-have/">a16z- and Benchmark-backed 11x has been claiming customers it doesn’t have | TechCrunch</a>: Last year, AI-powered sales automation startup 11x appeared to be on an explosive growth trajectory. However, nearly two dozen sources — including</li><li><a href="https://reddit.com/r/ClaudeAI/comments/1jijnw9/anthropic_is_making_about_115m_a_month_now_same/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">Models &amp; Pricing | DeepSeek API Docs</a>: The prices listed below are in unites of per 1M tokens. A token, the smallest unit of text that the model recognizes, can be a word, a number, or even a punctuation mark. We will bill based on the tot...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new talk/post from me: https://x.com/swyx/status/1904256213661192405
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1353830172197126216)** (7 messages): 

> `Audio processing with ilgpu + cufft + kernels, Asynchronous data transfer to GPU with OnnxRuntime on CUDA, Double buffering with CUDA streams, FSDP fine tuning with trl library` 


- **ILGPU, CUFFT, and Kernel Audio Processing?**: A member inquired about experiences with **audio processing** using **ilgpus**, **cufft**, and **kernels**, seeking practical applications.
- **Double-Edged Diagrams Requested for Sanity Checks**: A member requested that diagrams show relationships **both ways** (row/col to threads/registers/bytes and vice versa) to improve intuitiveness and sanity checks.
   - They argued that providing both **visuals and code/pseudocode** prevents misunderstandings and wasted implementation time.
- **Asynchronous CUDA Transfers Paced with OnnxRuntime?**: A member sought an **online reference** for asynchronous data transfer to the GPU and running inference using **OnnxRuntime** on a **CUDA GPU**, aiming to eliminate data transfer bottlenecks in their inference pipeline.
   - They wanted to overlap data transfer for the next image with inference on the current image but are facing issues implementing this in **Python**.
- **Double Buffering Dreams for Maximized GPU**: A member suggested using **multiple CUDA streams** for asynchronous copies and forward passes, known as **double buffering**, to maximize GPU utilization and avoid CPU blocking.
   - This approach involves one stream issuing an async copy while another thread executes the forward pass, operating concurrently.
- **FSDP fine tuning's dataset handled?**: A member inquired about the proper way to handle datasets when fine-tuning with **FSDP** (Fully Sharded Data Parallel) using the `trl` library.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1353971642941968428)** (9 messages🔥): 

> `Triton Interpret Bug, Intel Triton Extension, Triton Compile Script, Prune Configs Support in Triton` 


- **Triton Interpret Flag Sparks Bug Hunt**: Differences between **TRITON_INTERPRET=0** and **TRITON_INTERPRET=1** are being investigated for potential bugs, with small discrepancies expected but larger ones indicating issues.
   - A user linked to a [paper](https://arxiv.org/abs/2503.14985) discussing **Triton** as a DSL offering a more user-friendly and portable alternative to low-level interfaces like CUDA or SYCL.
- **Intel Dives Deeper into Triton**: An extension to **Triton** from Intel was shared, highlighting the growing interest and investment in Triton for GPU programming.
   - The extension aims to improve compiler decoupling and cleanliness through multi-level gradual lowering, leveraging **GPU's hierarchical structure** and **SIMD units**.
- **Triton Compile Script Troubles**: A user faced issues with the **triton compile.py** script, initially failing due to an incorrect **--signature** flag.
   - The user shared a code snippet and command-line arguments used for compilation, indicating efforts to integrate Triton into their workflow.
- **Prune Configs Support in Triton**: A user mentioned having added support for prune configs a few months back, acknowledging some quirks but expressing confidence in its functionality.
   - Another user acknowledged this contribution and expressed intent to try it out with the nightly build, signaling potential adoption and testing of the feature.



**Link mentioned**: <a href="https://arxiv.org/abs/2503.14985">ML-Triton, A Multi-Level Compilation and Language Extension to Triton GPU Programming</a>: In the era of LLMs, dense operations such as GEMM and MHA are critical components. These operations are well-suited for parallel execution using a tilebased approach. While traditional GPU programming...

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1353919821019873291)** (24 messages🔥): 

> `CUDA swizzling, cuTensorMap issues, Flash Attention memory layout, Cutensor coordinate mapping` 


- **CUDA Async Warpgroup Swizzling Examined**: A member analyzed **CUDA's async warpgroup swizzle TF32** layout, questioning the non-consecutive numbers in the first row and the starting number determination for each subpartition, referencing the [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-k-32b-swizzle-tf32).
   - They later identified the layout as `Swizzle<0,4,3> o ((8,2),(4,4)):((4,32),(1,64))`, enabling construction of the original data position and combining it with `Swizzle<1,4,3>`.
- **Flash Attention Layout Questioned**: A member questioned why **Flash Attention** does not use a `(batch_size, num_heads, N, d)` memory layout, suggesting it might be superior to the existing `(batch_size, N, num_heads, d)` layout.
   - The user found their prior attempts using this layout went *horribly wrong*, based on an example from [NVIDIA's CUDA-C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#example-matrix-transpose).
- **Discrepancies in cuTensorMap Parameters**: A member highlighted a discrepancy between the **box_size** parameter definition in a CUDA example (bytes) and the cuTensorMapEncodeTiled documentation (elements), referencing a [CUDA-C Programming Guide example](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#example-matrix-transpose).
   - The user also noted that `box_size` is in reverse order compared to the shared memory size of `int4 [8][8]`.
- **Mapping Coordinates in CuTe Fragments**: A member asked about the easiest way in **CuTe** to map coordinates inside a fragment owned by a thread created from `tiled_mma.get_thread_slice(tid)` back to the coordinates of the whole resulting matrix.



**Link mentioned**: <a href="https://forums.developer.nvidia.com/t/some-question-about-creating-cutensormap-and-use-it/328193">Some question about creating CUtensorMap and use it</a>: I have some questions about the following code.     constexpr int row = 8;     constexpr int col = 64;     size_t byteSize = row * col * sizeof(int);     int* h_data = (int*)malloc(byteSize);     int*...

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1353916418881491028)** (2 messages): 

> `PyTorch allocator, torch.nn.utils.prune` 


- **PyTorch Caching Allocator Impedes Non-Caching Alternatives**: A user noted the difficulty of using a *non-caching* allocator alongside **PyTorch's caching allocator**, because MemPool uses the custom allocator with caching, which is not ideal.
- **Concerns Raised on PyTorch's Caching Allocator**: A user wondered why companies might avoid alternative caching allocators in production, citing concerns about debugging tools.
   - They expressed uncertainty about the magic behind the **PyTorch caching allocator** that would prevent using alternatives.
- **Pruning already pruned weights in torch.nn.utils.prune leads to wrong pruning ratios**: A user reported that `torch.nn.utils.prune` doesn't allow pruning to be applied only to non-pruned weights when pruning an already pruned layer.
   - They found that applying pruning twice with an amount of **0.2** does not result in a final amount of **0.4** due to re-pruning already pruned weights.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1354199849687318729)** (1 messages): 

> `AMD GPU support in Triton, Job postings for Triton developers` 


- ****AMD** adds **GPU** support to **Triton****: There are job postings open for engineers to build awesome **AMD GPU support** in **Triton**; both senior and junior positions in NA or Europe with remote options available, see the [LinkedIn post](https://www.linkedin.com/posts/antiagainst_triton-amd-gpu-activity-7288624355247374336-gS6q/).
- ****NA/EU** roles open at **AMD****: For **North America**, there is an opening at [careers.amd.com](https://careers.amd.com/careers-home/jobs/57679), while for **Europe** it is [careers.amd.com](https://careers.amd.com/careers-home/jobs/62233).
   - AMD warns against job scams and advises applicants to apply directly through the [amd.com](https://www.amd.com/) Careers page.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://careers.amd.com/careers-home/jobs/57679">Triton Compiler Engineer in San Jose, California | Advanced Micro Devices, Inc</a>: AMD | Careers Home is hiring a Triton Compiler Engineer in San Jose, California. Review all of the job details and apply today!</li><li><a href="https://careers.amd.com/careers-home/jobs/62233">Triton Compiler Senior Engineer in Cambridge, United Kingdom | Advanced Micro Devices, Inc</a>: AMD | Careers Home is hiring a Triton Compiler Senior Engineer in Cambridge, United Kingdom. Review all of the job details and apply today!
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

bigfoot1144: Any progress so far?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1354006565128241194)** (2 messages): 

> `gpumode leaderboard, MI250 node, AMD Instinct MI250 evaluation` 


- **GPUMODE Leaderboard: MI250 as a Stopgap?**: A member proposed leveraging the **GPUMODE leaderboard**, featuring an **MI250 node**, as a temporary solution.
   - They suggested applying for access to the [AMD Instinct MI250 evaluation program](https://www.amd.com/en/products/accelerators/instinct/eval-request.html).
- **Requesting Access to AMD Instinct MI250**: A user shared a link to the [AMD Instinct MI250 evaluation request page](https://www.amd.com/en/products/accelerators/instinct/eval-request.html).
   - This suggestion was in the context of finding available hardware for testing or benchmarking.


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1353939648380145675)** (5 messages): 

> `TileLang Compatibility with Torch AOTexport, TileLang compilation for AMD, Custom Triton Kernels` 


- **TileLang eyes Torch AOTexport compatibility**: A member inquired about **TileLang's** compatibility with **Torch AOTexport** and inference in **C++**, given a codebase with custom **Triton kernels**.
   - Another member confirmed that **TileLang** compiles kernels into **CUDA C source code**, accessible via `kernel.get_kernel_source()`, paving the way for potential AOT integration.
- **TileLang Targets HIP for AMD**: A member asked about TileLang's compilation target for **AMD** GPUs.
   - The response indicated that **TileLang** compiles to **HIP source code** for AMD architectures.


  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1354079283991150673)** (3 messages): 

> `Iterative Pruning, Double Pruning Weights, Pruning Ratio Calculations` 


- **Pruning Weights Twice Causes Problems**: A user reported that when applying pruning to an already pruned layer using `torch.nn.utils.prune`, there's no direct option to prune only non-pruned weights.
   - The user notes that applying a pruning ratio creates a global mask that affects the entire tensor, potentially re-pruning already pruned weights, leading to an incorrect final pruning ratio.
- **Simple Iteration does not fix Pruning Weights Twice**: A member asked if applying a ratio of **0.2** in the first iteration, then **0.4** in the second iteration on the already pruned weights, achieves the desired outcome.
   - The original poster responded that masks respect already defined masks, so you need calculate the new pruning ratio on the already pruned layer (or network) to arrive at the desired ratio.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1354157815333261434)** (2 messages): 

> `lce_forward_deprecated vs lce_forward` 


- **`lce_forward_deprecated` vs `lce_forward`**: The user asked what the differences are between `lce_forward_deprecated` and `lce_forward` and the reason to deprecate the old one.
   - No response was given.
- **Reasons for Deprecation**: There was no response detailing reasons for deprecation, so the reasons remain unknown.
   - More information is needed.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1353914790724112404)** (3 messages): 

> `Open Source ML platform building` 


- **Plans to Build Open Source ML Platform Spark Excitement**: A member expressed a desire to build out more features for their **open source machine learning platform**.
   - Another member responded with encouragement to share progress on the platform.
- **Open Source ML Platform Gets Encouragement**: A member wants to develop their **open source ML platform** further.
   - Another member cheered them on and encouraged them to share their progress.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1353988440676634746)** (5 messages): 

> `fp16 MatMul for Gemma3, Gemma3 Residuals, CUDA Execution Time Benchmarks, Inferless on Product Hunt` 


- **Gemma3 leverages fp16 MatMul**: It's observed that **fp16 matmul** *might* be suitable for **Gemma3**, however, it's crucial to cast the outputs back to **bf16**.
- **Gemma3's Residuals Are Tricky**: Quantization of **fp16 weights** faces challenges with **Gemma3**, necessitating a fix like the one proposed in [this Hugging Face transformers PR](https://github.com/huggingface/transformers/pull/36832).
- **CUDA Benchmark Insights Emerge**: A blog post detailing CUDA execution time benchmarks emphasizes the use of **CUDA Events over CPU timers** and excluding memory transfers unless relevant.
   - The author concludes: *'If you want to benchmark then you should use an actual production grade profiler'*. and notes that the majority of resources online simply tell you to do x or y, but I want to see **_the numbers_**.
- **Inferless Goes Live on Product Hunt**: **Inferless**, a serverless compute platform for deploying ML models, has launched on [Product Hunt](https://www.producthunt.com/posts/inferless), offering **$30 compute** for new signups.
   - They said that their goal is to *Deploy any machine learning models in minutes* and offer *ultra-low cold starts*.



**Link mentioned**: <a href="https://github.com/huggingface/transformers/pull/36832">gemma3 fp16 fix by mobicham · Pull Request #36832 · huggingface/transformers</a>: What does this PR do?Fixes float16 inference with Gemma 3 models by simply clipping the activations. The residual addition step should also be clipped for more accurate outputs. Without this fix, ...

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1353828756220936292)** (10 messages🔥): 

> `ARC-AGI-2 Benchmark, Reasoning-Gym Puzzles, verL and vLLM0.8.1, Codegen Updates, RL Research Directions` 


- ****ARC-AGI-2** Benchmark Announced**: The **ARC-AGI-2**, an unsaturated frontier AGI benchmark that challenges AI reasoning systems, was announced, aiming to measure AI reasoning systems with a grand prize for achieving **85%** efficiency at about **$0.42/task**.
   - Current performance benchmarks show base LLMs at **0%** and reasoning systems achieving less than **4%** success.
- **verL Supports vLLM0.8.1**: verL now supports **vLLM0.8.1**, with ongoing efforts to set up the image on a cluster for local inference, and updates are expected soon on the **Codegen** side.
   - A member is excited to share a pretty hefty update regarding **CodeGen** soon.
- **Cool Open Questions and Research Directions in RL**: A member shared [a link](https://docs.google.com/spreadsheets/d/1s_ZDKtOoGqi1FtTyPeeS0h_0d96jOFJ-TuzwjWnnCPo/edit?gid=1478931401#gid=1478931401) to a Google Sheets document outlining cool open questions and research directions in RL.
   - The document covers topics like demand hypothesis, status quo, 10x improvements, and proposals for solving big, open problems in the world.
- **RL Reward Shaping Framework Urged**: A framework for **RL reward shaping** was proposed to simplify quick experiments for determining the right approach, suggesting that instead of messy tweaks, a framework to manage weights and loops in one go would be beneficial.
   - The member stated that in the near future, everyone will be doing RL but there's no good framework for it, suggesting a **Bayesian optimization framework** as a potential solution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/arcprize/status/1904269307284230593">Tweet from ARC Prize (@arcprize)</a>: Today we are announcing ARC-AGI-2, an unsaturated frontier AGI benchmark that challenges AI reasoning systems (same relative ease for humans).Grand Prize: 85%, ~$0.42/task efficiencyCurrent Performanc...</li><li><a href="https://x.com/arcprize/status/1904269307">Tweet from Bill Engvall (@billengvall)</a>: Gotta figure out how to send twit pics</li><li><a href="https://docs.google.com/spreadsheets/d/1s_ZDKtOoGqi1FtTyPeeS0h_0d96jOFJ-TuzwjWnnCPo/edit?gid=1478931401#gid=1478931401">Lossfunk - ideas / research directions</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1353952879131496609)** (1 messages): 

> `Flash Attention Layout, Tensor Layout, Performance Optimization` 


- **Flash Attention Layout Optimization Explored**: A member inquired why **Flash Attention's layout** isn't ordered as **(batch_size, num_heads, N, d)** instead of **(batch_size, N, num_heads, d)** for potentially faster performance.
   - The question suggests an optimization angle, questioning whether reordering the tensor layout could improve computational speed during attention mechanisms.
- **Tensor Layout Impact on Performance**: The discussion centers on how different **tensor layouts** can affect the efficiency of operations, particularly in the context of **Flash Attention**.
   - Different memory access patterns can have a significant impact on performance, making this a relevant optimization consideration.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1353842717272178758)** (6 messages): 

> `Conv2d compilation errors, CUDA compilation issues, PyTorch C++ extension problems, load_inline issues` 


- **Users encounter CUDA compilation errors during conv2d submission**: A user encountered a `RuntimeError` during submission to conv2d, stemming from a `subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1` error during the extension module build, with a [long error traceback in the logs](https://fake.link/errorlog).
   - The root cause appears to be that the CUDA source code couldn't compile properly, as indicated by the error message: *Error building extension 'conv2d_module'*.
- **Confirmation of module issues by other users**: Another user suggested that the issue might be related to a wrong or missing module, specifically mentioning uncertainty about `load_inline` function in `/root/submission.py`, line 191.
   - The original poster acknowledges the feedback and plans to investigate the module and `load_inline` function further, with no further information.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1353876682926194688)** (3 messages): 

> `H100 benchmarks, T4 vectorsum, A100 grayscale` 


- **H100 gets Grayscale Benchmark**: Benchmark submission with id **2988** to leaderboard `grayscale` on GPUS: **H100** using Modal runners succeeded!
- **T4 excels in Vectorsum task**: Leaderboard submission with id **3005** to leaderboard `vectorsum` on GPUS: **T4** using Modal runners succeeded!
- **A100 Aces Grayscale test**: Test submission with id **3006** to leaderboard `grayscale` on GPUS: **A100** using Modal runners succeeded!


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1353817937567285373)** (40 messages🔥): 

> `DeepSeek as moderation bot, Numerical RAG with Databricks, Fine-tuning open-source LLMs, VAD tool language agnostic, Hugging Face AgentX competition` 


- **DeepSeek Debated as Discordian Decider**: A member inquired whether [DeepSeek](https://deepseek.com/) is suitable as a moderation bot.
   - Another member responded affirmatively but suggested that a smaller **3B LLM** might suffice at a cost of *5 cents per million tokens*.
- **Numerical Nirvana: RAG Edition**: A member is building a **RAG** on numerical structured data from **Databricks** and wants the LLM to understand the query, create a query, run it, and then reply back in natural language.
   - They requested suggestions for tutorials on this approach, potentially saving the world from **2025.02.24-150819**.
- **Hugging Face Launches AgentX Competition**: Hugging Face is hosting the **AgentX – LLM Agents MOOC Competition** in conjunction with the [Advanced LLM Agents MOOC](https://llmagents-learning.org/sp25), calling for trailblazers ready to push the boundaries of **AI and Agents**.
   - The **AgentX Competition** is open to the public and culminates in an in-person Demo Day at the Agents Summit this August at **UC Berkeley**.
- **Beginner Asks How to Fine-Tune CUDA on Windows**: A member sought an absolute beginner's guide for fine-tuning a model that works in **Windows**, preferably with **CUDA** support.
   - Another member said this was impossible since installing **PyTorch** and the **CUDA Toolkit** on Windows is *hell*, and linked to guides [Step-by-Step-Setup-CUDA-cuDNN](https://github.com/imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility) and [Installing-pytorch-with-cuda-support-on-Windows](https://www.gpu-mart.com/blog/Installing-pytorch-with-cuda-support-on-Windows).
- **Hosted Dataset Faces HTTPRequest Hurdles**: A member is experiencing **HTTPRequest** errors while fine-tuning **ViT** with a dataset (15GB, 102k datapoints) hosted in 24 shards (500MB each).
   - They asked for help understanding the cause of the error in [this discord channel](https://discord.com/channels/879548962464493619/1339556954162462851).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/1354052436217823334">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://discordapp.com/channels/879548962464493619/13540">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://huggingface.co/chat/).">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://rdi.berkeley.edu/agentx/">AgentX</a>: AgentX is hosted by RDI at UC Berkeley.</li><li><a href="https://aikval25.kattis.com/contests/aikval25/problems/windchill">Windchill &ndash; Kattis, AI-olympiadens Kval 2025</a>: no description found</li><li><a href="https://huggingface.co/spaces/opencompass/Open_LMM_Reasoning_Leaderboard">Open LMM Reasoning Leaderboard - a Hugging Face Space by opencompass</a>: no description found</li><li><a href="https://huggingface.co/spaces?q=leaderboard&sort=trending">Spaces - Hugging Face</a>: no description found</li><li><a href="https://archive.ph/2025.02.24-150819/https://medium.com/data-scientists-from-future/fine-tuning-open-source-language-models-a-step-by-step-guide-a38bed8df923">Fine-Tuning Open-Source Language Models: A Step-by-Step Guide | by Vi&#x2026;</a>: no description found</li><li><a href="https://github.com/imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility">GitHub - imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility: This repository provides a step-by-step guide to completely remove, install, and upgrade CUDA, cuDNN, and PyTorch on Windows, including GPU compatibility checks, environment setup, and installation verification.</a>: This repository provides a step-by-step guide to completely remove, install, and upgrade CUDA, cuDNN, and PyTorch on Windows, including GPU compatibility checks, environment setup, and installation...</li><li><a href="https://www.gpu-mart.com/blog/Installing-pytorch-with-cuda-support-on-Windows">How to Install Pytorch with CUDA support on Windows</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/training">Fine-tuning</a>: no description found</li><li><a href="https://huggingface.co/learn/nlp-course/chapter3/1">Introduction - Hugging Face NLP Course</a>: no description found</li><li><a href="https://huggingface.co/docs/autotrain/v0.8.24/index">AutoTrain</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/v4.49.0/perf_train_gpu_one">Methods and tools for efficient training on a single GPU</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

ynvers256: Today I'm learning Renforcement learning and make research about Eureka
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1354165894841893035)** (1 messages): 

> `Aider + Zed, Codium's Windsurf, TabNine, Cursor` 


- **Users Inquire About Aider, Windsurf, Tabnine and Cursor**: A member inquired about experiences with **Aider** in conjunction with the **Zed editor**, **Codium's Windsurf**, and **TabNine**, seeking user feedback.
   - The user explicitly requested comparisons of these tools to **Cursor**.
- **Comparison Request: Code Editors and AI Assistants**: The primary purpose of the message was to gather comparative insights on various code editors and AI assistant tools.
   - The user aims to understand the strengths and weaknesses of each tool relative to **Cursor**, potentially for making an informed decision on which to use.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1353862454014382120)** (14 messages🔥): 

> `Audio Extraction Tool in Rust, Achievement Token System, Music Generation System` 


- **Rust Tool Extracts Audio Blazingly Fast**: A new [tool](https://github.com/egorsmkv/extract-audio) has been released to extract audio files from **parquet** or **arrow files** generated by the **Hugging Face datasets library**, with a [Colab demo](https://colab.research.google.com/drive/1prztEZIf8nNFUSaptY8Jv16VO8Crjnzb?usp=sharing).
   - The developer aims to provide **blazingly fast speeds** for audio dataset extraction.
- **Token System Powers Kid-Friendly App**: The developer is implementing a **token system** for achievements within their kid-friendly app, which will allow kids to earn tokens and unlock **games** and **sanitized image generation**.
   - The system already has a **font size slider** and safety measures to ensure **kid-friendly content**, with plans to create **printable achievement certificates**.
- **Synthesized Clippy with Lipsync avatars**: The developer is working on incorporating a **music generation system**, alongside options for users to select their own **lipsync avatars** as a "Clippy"-like assistant.
   - They are revamping **wav2lip** and experimenting with **latensync** to create a reasonably fast and high-quality talking UI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/egorsmkv/extract-audio">GitHub - egorsmkv/extract-audio: Extract audio files from a parquet or arrow file generated by Hugging Face `datasets` library.</a>: Extract audio files from a parquet or arrow file generated by Hugging Face `datasets` library. - egorsmkv/extract-audio</li><li><a href="https://colab.research.google.com/drive/1prztEZIf8nNFUSaptY8Jv16VO8Crjnzb?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1354012576534495232)** (2 messages): 

> `AutoCAD drawing generation, Metric scale at object location` 


- **AutoCAD Drawing Generation by Prompt**: A member inquired about generating **AutoCAD** drawings based on input prompts, seeking methods for automated design creation.
   - This suggests interest in using **AI** or other programmatic tools to translate textual descriptions into CAD models, potentially streamlining design workflows.
- **Measuring Metric Scale Sans Reference**: A member posed a challenge of obtaining a **metric scale** at an object's location without using a reference object or distance measurements.
   - This seeks innovative solutions for estimating size or scale in a scene without traditional spatial cues, perhaps involving **image analysis** or other contextual clues.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1354080726320021587)** (1 messages): 

> `Unstructured Data to JSON Conversion, LLM Fine-tuning Datasets` 


- **Users Seek Unstructured Data to JSON Converter Websites**: A member is looking for websites that convert **unstructured data** into **structured data** like **JSON** for **LLM fine-tuning datasets**.
- **LLM Fine-tuning Dataset Conversion**: The user specifically requires a tool to transform unstructured information into a structured format suitable for training Large Language Models.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1353824302524530848)** (1 messages): 

> `Gradio Deep Links, Gradio 5.23` 


- **Gradio Adds Deep Linking Delight!**: **Gradio 5.23** introduces support for **Deep Links**, enabling direct linking to specific generated outputs like images or videos.
   - An example link to a **blue jay image** generated by Flux is provided: [https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek](https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek).
- **Upgrade to Gradio 5.23 Now!**: Users are instructed to upgrade to the latest version, **Gradio 5.23**, via `pip install --upgrade gradio` to access the new **Deep Links** feature.
   - An attached image showcases the new feature, and can be found [here](https://cdn.discordapp.com/attachments/1014577787039924226/1353824302855622746/image.png?ex=67e46022&is=67e30ea2&hm=d0e0e82ce95fbb6745775ca3274bbce8c92061de43b4f725f643d61076ed06f8&).



**Link mentioned**: <a href="https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek">black-forest-labs/FLUX.1-schnell</a>: no description found

  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1353973947770863646)** (4 messages): 

> `Llama-3.2 and LlamaIndex.ai, Ollama setup, BAAI/bge-base-en-v1.5, Custom Tool help` 


- **Llama-3.2 integrates with LlamaIndex.ai**: A member experimented with **Llama-3.2** using [this tutorial](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/), noting it demonstrates how to build agents with **LlamaIndex**, starting with a basic example and adding **Retrieval-Augmented Generation (RAG)** capabilities.
- **Ollama aids local LLM setup**: A member used **Ollama** as a tool to set up **LLMs** locally, following the [README](https://github.com/jmorganca/ollama) for installation.
   - To download the Llama3 model the command `ollama pull llama3.1` was used, and members noted that a machine with at least **~32GB of RAM** is needed.
- **BAAI embeddings model used**: The member used [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) as their embedding model.
   - The member had to `pip install llama-index-llms-ollama llama-index-embeddings-huggingface` to integrate with **Ollama** and **Huggingface**.
- **Custom Tool help requested**: A member requested help with editing a *my_custom_tool* function to build and test their own tool, asking how to edit it within **Hugging Face**.
   - Another member recommended going to *files* then clicking on the *app.py* file and then clicking edit within **Hugging Face**.



**Link mentioned**: <a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Using Local LLMs) - LlamaIndex</a>: no description found

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1353805793954103457)** (39 messages🔥): 

> `New MCP Mod, Nexus context management system for AI coding assistants, Atom of Thoughts for Claude, Deepseek V3 with AOT, Running Multiple MCP Servers` 


- **New Mod joins MCP Community!**: The MCP community welcomes a new moderator, [who has actively participated](https://github.com/evalstate) since the beginning and contributed to various projects.
   - He is planning to organize MCP events and help grow the community further.
- ****Nexus** Saves the Day for Context-Constrained AI Coders!**: A member shared [Nexus](https://www.reddit.com/r/mcp/comments/1jj3iuq/nexus_a_system_for_managing_context_and_improving/), a system to address context management challenges with AI coding assistants, particularly in large software projects, aimed at reducing **token costs** and improving **code accuracy**.
   - Nexus addresses the limited context windows of **LLMs**, which leads to inaccurate code generation and high token costs.
- ****Atom of Thoughts** Mesmerizes Claude Users**: A member recommended **Atom of Thoughts** for **Claude**, describing it as *incredible*, following a discussion about using Anthropic's 'think tool'.
   - Another member shared images of **Deepseek V3** working with **AOT**.
- **Serving Multiple Servers Simultaneously**: Members discussed how to run multiple MCP servers with user-defined ports, suggesting the use of **Docker** and mapping ports.
   - They also pointed to the ability to configure ports via the `FastMCP` constructor in the [python-sdk](https://github.com/modelcontextprotocol/python-sdk/blob/4e11f2890b30be59ca67e5198cb5ede8f401c3a2/src/mcp/server/fastmcp/server.py#L56).
- **GPT-4o-mini's Tool-Calling Hallucinations**: A member reported that **gpt-4o-mini** hallucinated tool call requests for a non-existent `process_text` function, even without any tool definitions provided.
   - The request later changed to `text_processing`, which was also not found in the workspace.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/mcp/comments/1jj3iuq/nexus_a_system_for_managing_context_and_improving/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://github.com/MissionSquad/nexus">GitHub - MissionSquad/nexus: The Nexus System: AI-Assisted Software Development Paradigm</a>: The Nexus System: AI-Assisted Software Development Paradigm - MissionSquad/nexus</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/4e11f2890b30be59ca67e5198cb5ede8f401c3a2/src/mcp/server/fastmcp/server.py#L56>">python-sdk/src/mcp/server/fastmcp/server.py at 4e11f2890b30be59ca67e5198cb5ede8f401c3a2 · modelcontextprotocol/python-sdk</a>: The official Python SDK for Model Context Protocol servers and clients - modelcontextprotocol/python-sdk
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1353805686282256496)** (23 messages🔥): 

> `Speech MCP, gotoHuman MCP Server, Apple MCP tools, VNC control via Claude` 


- ****MCP** for Speech Interaction Surfaces**: A member shared their main MCP for speech interaction with audio visualization: [speech-mcp](https://github.com/Kvadratni/speech-mcp), a Goose MCP extension.
   - This allows voice interaction with **audio visualization**.
- ****gotoHuman** MCP Server Seeks Approvals**: The gotoHuman team presented an MCP server to request **human approvals** from agents and workflows: [gotohuman-mcp-server](https://github.com/gotohuman/gotohuman-mcp-server).
   - The server allows for easy human review of LLM actions, defining the approval step in **natural language** and triggering a webhook after approval.
- ****Apple MCP** Tools Released**: A member introduced a collection of **apple-native tools** for the MCP protocol: [apple-mcp](https://git.new/apple-mcp).
   - A demo of this local MCP server using **VNC control** can be found in [this step-by-step video](https://x.com/DhravyaShah/status/1892694077679763671).
- **Control Desktops via **VNC** and Claude**: A co-founder shared their side-project that provides **VNC** control of remote macOS desktops via the **Claude** desktop app: [mcp-remote-macos-use](https://github.com/baryhuang/mcp-remote-macos-use).
   - A **YouTube** demo is available [here](https://www.youtube.com/watch?v=--QHz2jcvcs), where the use case is to connect Blender MCP with MCP Omni to connect CLI with OpenAI models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://www.youtube.com/watch?v=--QHz2jcvcs">Claude Mcp Remote MacOs Use demo (subtitle)</a>: github repo: github.com/baryhuang/mcp-remote-macos-use</li><li><a href="https://github.com/gotohuman/gotohuman-mcp-server">GitHub - gotohuman/gotohuman-mcp-server</a>: Contribute to gotohuman/gotohuman-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/Kvadratni/speech-mcp">GitHub - Kvadratni/speech-mcp: Speech MCP: A Goose MCP extension for voice interaction with audio visualization</a>: Speech MCP: A Goose MCP extension for voice interaction with audio visualization - Kvadratni/speech-mcp</li><li><a href="https://git.new/apple-mcp">GitHub - Dhravya/apple-mcp: Collection of apple-native tools for the model context protocol.</a>: Collection of apple-native tools for the model context protocol. - Dhravya/apple-mcp
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1353871431959576696)** (7 messages): 

> `NotebookLM, Versatile Bot Project, Interactive mode, Chat Episode Prompt, Delivery pacing` 


- **Podcast Host with NotebookLM**: A member asked for a hack to use **NotebookLM** as a podcast host with the user as a guest, to engage in conversation about a given topic.
   - Another member asked where to find the **Chat Episode Prompt** to enable this.
- **Dive into Versatile Bot Project**: A member launched the [Versatile Bot Project](https://github.com/shun0t/versatile_bot_project), which includes a **Chat Episode prompt document** designed for AI hosts to discuss arbitrary topics in **Interactive mode**.
   - This mode allows users to join the episode while the AI hosts are talking, enabling conversations on specified topics.
- **Control AI Host Delivery Pacing**: A member inquired about changing the pacing of the AI host's delivery, such as faster or slower reading speeds.
   - Another member provided a template with parameters like *Energetic pace*, *clear articulation*, and target **words/minute** to control the AI's delivery speed, with source document type influencing the AI's guidance.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1353812676769480795)** (47 messages🔥): 

> `Google Cloud Platform Billing, Workspace Data Export Tool, NotebookLM Plus subscription benefits, Mind Map Feature rollout, Multilingual Podcast` 


- **GCP Billing Confusion Clarified**: A user enabled **Google Cloud Platform** to access the **Data Export** tool, but was unsure about billing; another user clarified that enabling billing doesn't guarantee charges.
   - The user confirmed they initiated the Data Export from the admin console, accessing the archive via console.cloud.google.com.
- **Data Export Caveats Exposed**: Users discovered that the option to choose the data destination during export is limited by their **Workspace edition** and the exported data is stored in a **Google-owned bucket** and will be deleted in **60 days** as explained on [Google Support](https://support.google.com/a/answer/14338836?sjid=14118684210403272528-EU&hl=en).
- **Mind Map Availability Miff**: Users reported missing **Mind Map** features in NotebookLM, and it was confirmed to be a **gradual rollout**.
   - Some users speculated that the delayed rollout is due to bug fixes, with one user saying that the pace of the roll out is *like a snails pace of an actual rollout*.
- **Customize Box Concealment Confirmed?**: A user asked if the **customize box** is no longer available in the free version of NotebookLM.
   - Another user responded that the **customize box** is still showing up in the free account, not plus.
- **Multilingual podcast missing**: A user requested a **multilingual feature** for NotebookLM, specifically for the **podcast** feature.
   - Another user noted that NLM chat is already multilingual, but the podcast feature is currently only available in English.



**Link mentioned**: <a href="https://support.google.com/a/answer/14338836?sjid=14118684210403272528-EU&hl=en">Export your users' data - Google Workspace Admin Help</a>: no description found

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1353830953004568710)** (19 messages🔥): 

> `Universal Translator, Mozilla's Transformer Lab, GPU Tokenization, Gemini 2.5 Pro` 


- **Universal Translator Incoming?**: A member speculates that a **universal translator** is only five years away given **ChatGPT's** ability to understand and translate languages.
   - Another member then shares a [YouTube link](https://www.youtube.com/watch?v=K1RbD7aAXtc) asking *what model is singing* in the video.
- **Mozilla's Transformer Lab Sparks Interest**: A member asks if anyone has seriously looked at **Mozilla's Transformer Lab** project, a potential way to bring training and fine-tuning to regular users on regular hardware, and shares a link to the [GitHub repo](https://github.com/transformerlab/transformerlab-app).
   - Another member confirmed that `Transformer Lab is proud to be supported by Mozilla through the Mozilla Builders Program`, and clarified that it is *supported* but not *created* by them.
- **GPU Tokenization Claims**: A member noted that during tokenization, **LM Studio** pushes a single CPU thread to full throttle, but is curious whether the process of tokenizing is 100% on the GPU.
   - Another user replied that *tokenizing has nothing to do with the GPU*, but then countered himself, observing that playing with **flash attention** and **cache settings** for k and v has a big effect on the tokenizing time.
- **Gemini 2.5 Pro Gets Trialed**: A member asks whether others have tried **Gemini 2.5 Pro**.
   - Another member replied that they tried it and it was able to correctly answer a logic puzzle that **Gemini 2.0 Flash Thinking** could not, and shares a link to use it for free on [aistudio](https://www.hopeless.fr/share/msedge_O0y9jZHBZV.png).



**Link mentioned**: <a href="https://github.com/transformerlab/transformerlab-app">GitHub - transformerlab/transformerlab-app: Open Source Application for Advanced LLM Engineering: interact, train, fine-tune, and evaluate large language models on your own computer.</a>: Open Source Application for Advanced LLM Engineering: interact, train, fine-tune, and evaluate large language models on your own computer. - transformerlab/transformerlab-app

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1353840864043995278)** (23 messages🔥): 

> `VRAM limits on GPUs, Docker container overhead with GPUs, 3090 Ti speed, M4 Max power consumption with 32B models, ROCm support for AMD GPUs` 


- **GPU VRAM Limit Tweaks Explored**: A user is trying to decrease the reserved **VRAM** on their **16 GB GPUs** from **0.5 GB** to **0.2 GB** for LLM processing, but another member warned against fully utilizing **VRAM** to avoid system lockups.
   - It may be possible to get to **0.3 GB** but it *won't be reliable, as any additional sight on VRAM will make setup to crumble*.
- **Docker Overhead with GPUs**: A user accesses each of their **8 GPUs** through its own **Docker** container, and questioned system memory overhead.
   - One member said that *Docker should contain basically no internal system, besides bare minimum*, but the CUDA core still should be loaded within each instance.
- **3090 Ti Fully Loaded**: One user showcased their **3090 Ti** being fully loaded, showing *pure speed* in a [screenshot](https://cdn.discordapp.com/attachments/1153759714082033735/1354073319133155429/image.png).
   - They reported speeds of **~20 tokens/s** without flash and slowing down after **4-5k tokens**, and **~30 tokens/s** with flash.
- **M4 Max Power Spike with 32B Thinking Models**: A user noted that **32B-class thinking models** drive power higher on their **M4 Max** (up to **140W**) compared to same or larger sized models (up to **120W**).
   - The user guesses this is due to *more aggressive memory access*, because *the gpus seem to be running roughly the same*.
- **ROCm Support Updates for AMD GPUs**: A user asked about support for the latest AMD GPUs and another user mentioned that it's *only through Vulkan*, lacking **ROCm** support in master llama.cpp releases.
   - Another member clarified that the ROCm llama.cpp engine is updated along with CUDA and Vulkan, but noted that support for certain AMD GPUs like the **90xx** may be a separate question.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1353816255278092359)** (17 messages🔥): 

> `Data Retention, Security, Data Privacy, Data Usage Policy, Zero Data Retention (ZDR)` 


- **Cohere Clarifies Data Privacy Policies**: In response to a user inquiry, the Cohere team shared links to their [Privacy Policy](https://cohere.com/privacy) and [Data Usage Policy](https://cohere.com/data-usage-policy), emphasizing that users and customers should avoid uploading personal information when using the services.
- **Cohere Emphasizes Data Control Options**: Cohere offers a **SaaS platform** providing users direct control over their data through a [dashboard](https://dashboard.cohere.com/data-controls), as well as **Zero Data Retention (ZDR)** support upon request (via email to support@cohere.com).
- **Cohere's Deployment Options**: Cohere is available on major cloud providers like **OCI**, **Bedrock**, **Sagemaker**, and **Azure Cloud**, ensuring requests remain within the cloud environment; on-prem solutions are also available, detailed on their [deployment options page](https://cohere.com/deployment-options).
- **Cohere Achieves Security and Compliance Standards**: Cohere is **SOC II** and **GDPR compliant**, adhering to industry standards for data security and privacy, with more details available in their [security policy](https://cohere.com/security).
- **Users Gain Control Over Data Usage**: Users can manage their data settings on the [data controls dashboard](https://dashboard.cohere.com/data-controls), preventing their data from being used for prompt generation or fine-tuning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dashboard.cohere.com/data-controls">Login | Cohere</a>: Login for access to advanced Large Language Models and NLP tools through one easy-to-use API.</li><li><a href="https://cohere.com/security">Security | Cohere</a>: Ensure ultimate AI security and privacy with Cohere&#x27;s enterprise-grade security protocols, robust access controls, and private deployment options. </li><li><a href="https://cohere.com/privacy">Privacy Policy | Cohere</a>: Cohere Inc. (“Cohere”) values and respects your privacy. We have prepared this privacy policy to explain the manner in which we collect, use and disclose personal information through our Website locat...</li><li><a href="https://cohere.com/deployment-options">Deployment Options - SaaS, Cloud API, Virtual Private Cloud (VPC), On Premise | Cohere</a>: Our solutions provide industry-leading data privacy and security and are designed to meet the diverse needs of organizations seeking to harness the power of generative AI. Whether you’re a start-up or...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1354002232085057567)** (16 messages🔥): 

> `Cohere API streaming, Cohere embedding generator, Cohere tokenization` 


- **Cohere API supports response streaming**: The Cohere API supports response streaming, which can improve user experience by allowing them to see text appearing on the client end as it's generated, as documented in the [Cohere's Chat Stream API reference](https://docs.cohere.com/reference/chat-stream).
- **Tokenizing text for CohereEmbeddingGenerator**: A user building a **CohereEmbeddingGenerator** client in .NET asked about tokenizing text before embedding, and was advised to use the `/embed` endpoint which returns the number of tokens used, or download the tokenizer manually from [Cohere's public storage](https://storage.googleapis.com/cohere-public/tokenizers/embed-english-v3.0.json).
   - The user was directed to download the tokenizer listed for their model by making a GET request to `api.cohere.com/v2/models/embed-english-v3.0`,  grabbing the tokenizer from the `tokenizer_url` and then manually tokenizing using a library such as the HF tokenizer.



**Link mentioned**: <a href="https://docs.cohere.com/reference/chat-stream">Chat with Streaming — Cohere</a>: Generates a text response to a user message. To learn how to use the Chat API and RAG follow our Text Generation guides.Follow the Migration Guide for instructions on moving from API v1 to API v2.

  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1354175936861573120)** (2 messages): 

> `` 


- **No Relevant Discussions**: No relevant discussions were found in the provided messages. The messages consisted of simple greetings.
- **Channel Inactivity**: The channel appears to be mostly inactive, with only brief greetings exchanged between users.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1354119470205108268)** (2 messages): 

> `NLP project, Text summarization tool, Introduction of Sage` 


- **Sage Introduces Herself**: A new member named Sage introduced themself to the community.
   - They mentioned working on an **NLP project** for their university final year: building a **text summarization tool**.
- **Sage seeks guidance on Text Summarization Tool**: Sage is building a **text summarization tool** as an NLP project.
   - They hope to learn from the community and contribute in return, as they are currently facing difficulties.


  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1354137784583065741)** (1 messages): 

> `torchtune v0.6.0, Tensor Parallel, Phi 4, Multinode training` 


- **TorchTune v0.6.0 is Out!**: TorchTune just released **v0.6.0** with various new capabilities.
   - Release notes can be found [here](https://github.com/pytorch/torchtune/releases/tag/v0.6.0).
- **Tensor Parallel Capabilities**: **Tensor Parallel** is now supported for bigger distributed training and inference recipes!
   - This enhancement allows for more efficient handling of large-scale models and datasets.
- **Microsoft's Phi 4 supported**: Builders for **Phi 4**, the latest model from Microsoft, have been added.
   - More information about **Phi 4** can be found on the [Microsoft Tech Community Blog](https://techcommunity.microsoft.com/blog/aiplatformblog/introducing-phi-4-microsoft’s-newest-small-language-model-specializing-in-comple/4357090).
- **Multinode training is live!**: **Multinode training** is now supported, facilitating distributed training across multiple nodes.
   - Get started with multinode training [here](https://pytorch.org/torchtune/stable/tutorials/multinode.html).


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1353827988663177326)** (14 messages🔥): 

> `DeepSeek-V3, Quantization Aware Training, MoEs in torchtune` 


- **DeepSeek Drops V3 Model, Skips Readme**: Members noticed the [DeepSeek-V3 model](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) was released *without a readme*, joking that the **DeepSeek AI team** had become *unhinged*.
   - The new model boasts a **chat interface**, **Hugging Face integration**, and links to their **Discord**, **Wechat**, and **X** accounts.
- **TorchTune's MoEs Spark Daydreams**: One member made a *hidden reminder about **MoEs** addition in **torchtune***, pondering if it would require **8-9 TB of VRAM** and *a cluster of 100 H100s or H200s* to train.
   - They jokingly noted they'd *have to move a few boxes in the attic first* to make space.
- **Optimizer State Preserved after Quantization Aware Training**: A member asked about how **Quantization Aware Training (QAT)** affects the optimizer state, linking to the [relevant code in *torchtune*](https://github.com/pytorch/torchtune/blob/57c8d6b50d1462cc437d57991dca7f8acb599678/recipes/qat_distributed.py#L790).
   - Another member confirmed that *optimizer state is preserved after the switch to QAT*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/57c8d6b50d1462cc437d57991dca7f8acb599678/recipes/qat_distributed.py#L790">torchtune/recipes/qat_distributed.py at 57c8d6b50d1462cc437d57991dca7f8acb599678 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1353817270228484158)** (20 messages🔥): 

> `CUDA overhead, Cursed Submodule, vLLM + GRPO, r1-zero` 


- **Banish CPU CUDA Launch Overhead with Graph Capture**: To reduce GPU idle time, members discussed that launching CUDA operations from CPU has non-negligible overhead and that capturing GPU operations as a graph and launching them as a single operation can consolidate the computational graph.
   - One member asked if this is what compile does.
- **Is a /cursed submodule needed to improve Performance 10x?**: Members debated the creation of a `/cursed` submodule for *cursed code* that drastically improves performance, like manual CUDA device specification.
   - A member stated that he is using an approach only suitable for smaller models, where each process has its vLLM instance that generates data, instead of the more common approach of having a centralized vLLM generating process.
- **vLLM & GRPO Integration: smaller models vs. larger models**: A member has a working version for distributed setups, but stopped iterating on it after being notified that some work is already done internally.
   - The current approach is supposedly better for smaller models (up to 8B), whereas the internal approach would be better for large models (>=70B).
- **r1-zero Recipe for Async Training**: A member shared a [link to r1-zero](https://github.com/joecummings/r1-zero/blob/main/scripts/runnable_recipe_ray_vllm_weight_sync.py) for async training of reasoning models, emphasizing it's still a work in progress.
   - They also mentioned plans to integrate a version of this recipe into torchtune soon, focusing on cleaning up and allowing non-HF models with vLLM.
- **Layer GPU Address Mapping via HTTP Calls**: A member suggested modifying **vLLM** to launch with a mapping of each layer's GPU address.
   - The goal is to run a vLLM process on the side and interact with it via HTTP calls; someone is already *working on it*.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1353825891112845412)** (31 messages🔥): 

> `LocalDocs Backup, Privacy Considerations for Chat Data, Local LLM vs API for Message Processing, LocalDocs DB Import, Lost LocalDocs DB` 


- **LocalDocs DB Backup for Safety**: Members discussed backing up the `localdocs.db` file to avoid data loss, especially if the original documents are lost or inaccessible.
   - One member suggested that GPT4All uses the `*.db` file with the highest number (e.g., `localdocs_v3.db`), and renaming them might allow for import/export, though this is unconfirmed.
- **Privacy Laws Challenge Chat Data Analysis**: One member raised concerns about privacy laws, particularly in the EU, and the need to ensure compliance when processing chat data.
   - The discussion highlighted the importance of verifying permissions and the format of the chat messages (plaintext or convertible) before feeding them into an LLM.
- **LLM API vs Local LLM for Chat Processing**: A member inquired about whether to use a paid API like **Deepseek** or **OpenAI**, or to run a local LLM for processing incoming group chat messages to calculate satisfaction rates, extract keywords, and summarize messages.
   - Another member suggested that if the messages are relatively small (under **100MB**), a local machine with a good GPU might suffice, especially if using smaller models for labeling and summarization.
- **LocalDocs DB Importing Challenges**: Members discussed the possibility of importing a `localdocs.db` file, but noted that the file contains encrypted/specially encoded text, making it difficult for a generic LLM to parse without an embedding model.
   - One member who lost their localdocs.db was experiencing painfully slow CPU indexing and was hoping to get around this problem.
- **Win11 Update Wipes LocalDocs**: A member experienced their `localdocs.db` becoming empty after a Windows 11 update and was struggling to re-index the local documents on CPU.
   - Drive letter changes due to the update were considered as a possible cause, with a suggestion to move files to the C drive to avoid such issues.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1354127968749879368)** (2 messages): 

> `LlamaCloud MCP Server, Build an MCP server in Python` 


- **LlamaCloud as an MCP Marvel**: [LlamaCloud](https://www.llamaindex.ai/) can be an **MCP server** to any compatible client and this [demo](https://t.co/t8yteZLg19) shows that in action.
   - It shows how **MCP** is very popular.
- **Pythonistas Produce Portable Pythonic Paradigm**: A member shows how to build your own **MCP server** using **LlamaIndex** to provide tool interfaces of any kind to any MCP client, with ~35 lines of Python connecting to **Cursor AI**.
   - Members implemented **Linkup web search** and [this](https://t.co/kj6UfDj0TU) project.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1353886649397809203)** (25 messages🔥): 

> `Claude MCP support, Multi-agent performance with LlamaIndex, Agent types in LlamaIndex, Automatic LLM evaluations` 


- **LlamaIndex Adds Claude MCP Compatibility**: A member provided a simplified example of integrating **Claude MCP** with **LlamaIndex**, showcasing how to expose a localhost and port for MCP clients like **Claude Desktop** or **Cursor** using `FastMCP` and `uvicorn` in a [code snippet](https://link.to.snippet).
- **Agents slow? AgentWorkflow for Speed**: A user reported slow performance with **LlamaIndex MultiAgentic** setups using **Gemini 2.0** with 12 tools and 3 agents; a suggestion was made to use `AgentWorkflow` and the `can_handoff_to` field for [controlled agent interaction](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#multi-agent-systems-with-agentworkflow).
- **LlamaIndex Agents Confuse Newbies**: A member expressed confusion about the different agent types in **LlamaIndex** and when to use them, and a refactor on docs is coming.
   - A team member noted that `core.agent.workflow` should generally be used, with **FunctionAgent** for **LLMs** with function/tool APIs and **ReActAgent** for others, suggesting the [Hugging Face course](https://huggingface.co/learn/agents-course/en/unit2/llama-index/agents) for additional help.
- **Automatic LLM Evals No Prompts!**: A founder is validating an idea for **OSS automatic evaluations** with a single API and no evaluation prompts, using proprietary models for tasks like Hallucination and Relevance in under 500ms, with a plan to offer an end-to-end solution including models, hosting, and orchestration tools, explained in more detail on the [autoevals.ai website](https://www.autoevals.ai).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.autoevals.ai">Home</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/en/unit2/llama-index/agents">Using Agents in LlamaIndex - Hugging Face Agents Course</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index">GitHub - run-llama/llama_index: LlamaIndex is the leading framework for building LLM-powered agents over your data.</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#multi-agent-systems-with-agentworkflow">Multi-agent workflows - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb">llama_index/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb at main · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1353894887120175186)** (4 messages): 

> `AI IDE Evaluation, SAEs on CoTs, DeepSeek-V3-0324 showcase, Gemini 2.5 Pro` 


- **New member joins interpretability efforts with SAEs focus**: A new member, Sam, with a background in physics and geometry, expresses interest in contributing to interpretability efforts, particularly using **SAEs** on **CoTs**.
   - Sam shares that their background is in *physics + geometry* and they have *data science* experience, including deep learning.
- **AI IDE Evaluation Repo Emerges**: A member is seeking feedback on evaluating **AI IDEs**, highlighting inconsistent performance between IDEs like Cursor and Windsurf despite using similar models, sharing a [repo](https://github.com/grahamannett/ai-ide-compare) for comparison.
   - The current evaluation focuses on 'greenfield' projects, using predefined prompts and assessing generated code/tasks through metrics like lines of code and files, with potential for incorporating more in-depth evaluations later.
- **DeepSeek-V3-0324 generates p5.js program**: A user quoted AK (_akhaliq) who highlights that **DeepSeek-V3-0324** successfully wrote a p5.js program showing a ball bouncing inside a spinning hexagon, affected by gravity and friction, bouncing realistically off the walls.
   - The model even improvised features like ball reset and randomization based on a prompt asking for sliders to adjust parameters and side count buttons, as showcased in [this tweet](https://x.com/teortaxesTex/status/1904342699756433859).
- **Google unveils Gemini 2.5 Pro**: Google released **Gemini 2.5 Pro**, touting it as the *world's most powerful model* with unified reasoning capabilities, long context, and tool usage as seen in [this tweet](https://x.com/OfficialLoganK/status/1904580368432586975).
   - The model is available experimentally and for free in **Google AI Studio + API**, with pricing details to follow soon.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OfficialLoganK/status/1904580368432586975">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Introducing Gemini 2.5 Pro, the world&#39;s most powerful model, with unified reasoning capabilities + all the things you love about Gemini (long context, tools, etc)Available as experimental and for ...</li><li><a href="https://x.com/teortaxesTex/status/1904342699756433859">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: V3-0324 did this in one shot. It improvised most of the features, like ball reset and randomize – I only asked for “sliders to adjust parameters” and side count buttons.Ball-posting is tedious. Defaul...</li><li><a href="https://github.com/grahamannett/ai-ide-compare">GitHub - grahamannett/ai-ide-compare</a>: Contribute to grahamannett/ai-ide-compare development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1354023705629429771)** (12 messages🔥): 

> `SkyLadder short-to-long context window transition, Data-constrained pretraining for math, Composable Generalization` 


- **SkyLadder: Short-to-Long Context is the Optimal Path**: A paper on ArXiv proposes **SkyLadder**, a simple yet effective approach that implements a **short-to-long context window transition** for LLM pretraining, yielding consistent gains of up to **3.7%** on common tasks ([2503.15450](https://arxiv.org/abs/2503.15450)).
   - The authors pre-train **1B** and **3B** parameter models on **100B tokens**, demonstrating that **SkyLadder** preserves strong standard benchmark performance while matching or exceeding baseline results on long context tasks.
- **Human Thought: The Key to Data-Constrained Pretraining?**: A new approach suggests that explicitly modeling and inferring the **latent thoughts** that underlie the text generation process can significantly improve pretraining data efficiency, especially for math ([2503.18866](https://arxiv.org/abs/2503.18866)).
   - The paper empirically demonstrates the effectiveness of inferring latent thoughts through synthetic data approaches, outperforming training on the same amount of raw data (**5.7% -> 25.4%**).
- **Composable Generalization Through Hypernetworks!**: A paper reformulates multi-head attention as a **hypernetwork**, revealing that a composable, low-dimensional latent code specifies key-query specific operations, enabling transformers to generalize to novel problem instances ([2406.05816](https://arxiv.org/abs/2406.05816)).
   - One member liked that *for a single pair of q, k indices, the authors interpret activations along the head-number dimension as a latent code (of dimension n_heads) specifying the task / context*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.05816">Attention as a Hypernetwork</a>: Transformers can under some circumstances generalize to novel problem instances whose constituent parts might have been encountered during training, but whose compositions have not. What mechanisms un...</li><li><a href="https://arxiv.org/abs/2503.18866">Reasoning to Learn from Latent Thoughts</a>: Compute scaling for language model (LM) pretraining has outpaced the growth of human-written texts, leading to concerns that data will become the bottleneck to LM scaling. To continue scaling pretrain...</li><li><a href="https://arxiv.org/abs/2503.15450">SkyLadder: Better and Faster Pretraining via Context Window Scheduling</a>: Recent advancements in LLM pretraining have featured ever-expanding context windows to process longer sequences. However, our pilot study reveals that models pretrained with shorter context windows co...</li><li><a href="https://arxiv.org/abs/2503.18908">FFN Fusion: Rethinking Sequential Computation in Large Language Models</a>: We introduce FFN Fusion, an architectural optimization technique that reduces sequential computation in large language models by identifying and exploiting natural opportunities for parallelization. O...</li><li><a href="https://arxiv.org/abs/2106.06295">Going Beyond Linear Transformers with Recurrent Fast Weight Programmers</a>: Transformers with linearised attention (&#39;&#39;linear Transformers&#39;&#39;) have demonstrated the practical scalability and effectiveness of outer product-based Fast Weight Programmers (FWPs) fro...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1354035224857673739)** (1 messages): 

> `Chinchilla Scaling Formula, Impact of Suboptimal Hyperparameters, Learning Rate Effects` 


- **Digging into **Chinchilla Scaling Formula****: A member inquired how suboptimal hyperparameter settings affect the **Chinchilla scaling formula** for optimal model training across different scales, referencing the [Chinchilla paper](https://arxiv.org/abs/2203.15556).
   - Specifically, they asked which parameters (**E**, **A**, **B**, **alpha**, or **beta**) in the formula would be altered if the **learning rate** was set too low (e.g., 1/10 of the optimal value).
- **Learning Rate's Ripple Effects**: The inquiry focuses on the theoretical consequences of a suboptimal **learning rate** on the scaling behavior predicted by the **Chinchilla formula**.
   - It aims to understand if a consistently low **learning rate** would primarily affect the **error term (E)**, the scaling coefficients (**A**, **B**), or the exponents (**alpha**, **beta**) governing the relationship between model size, training data, and performance.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1353847794858987610)** (1 messages): 

> `Self-organizing AI, AI Building Blocks` 


- **All AI Use Similar Building Blocks?**: A member introduced the idea that all these systems could use very similar building blocks.
   - This led them to the concept of a **self-organizing AI** that learns different potential configurations.
- **AI Self-Configuration Speculation**: A member posited a new direction for AI, suggesting that AIs can self-organize.
   - They added this self-organization could allow AIs to learn different potential configurations dynamically.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1353833840694788237)** (5 messages): 

> `gpt-neox CI status, lm_eval upgrade` 


- **gpt-neox CI Needs Fixing**: The EleutherAI/gpt-neox repo's CI is failing and needs some fixes because the volunteer who was maintaining them was hired elsewhere fulltime, though local tests are being run per PR.
   - The tests that passed locally include `pytest tests -m cpu`, and all tests passed except for a few requirements-related failures when running `pytest --forked --cov-report term --cov=megatron tests`.
- **lm_eval Upgrade PR Ready for Review**: A member has drafted a PR to update the evaluation logic to `lm_eval==0.4.8`, the latest version, and proposes resolving failing tests in another PR as they seem unrelated and also fail on main, linked here: [PR 1348](https://github.com/EleutherAI/gpt-neox/pull/1348).
   - A member suggests that the test failures might be due to an incorrectly set up environment or inconsistent versioning of dependencies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1348">Update Evaluation Logic to Latest `lm_eval` (0.4.8) and Support Automatic Benchmark Evals w/o Validation Set by Kyle1668 · Pull Request #1348 · EleutherAI/gpt-neox</a>: I&amp;#39;m training a model where I want to train on the entire datasets. I do not want to split the dataset into train/val/test. I want to evaluate on a set of benchmarks, one of which was introduce...</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1349">[Throw Away] Sanity Check CI by Kyle1668 · Pull Request #1349 · EleutherAI/gpt-neox</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1354045126258982994)** (15 messages🔥): 

> `Mojo for Website vs Rust, AES in Mojo Progress, SIMD for AES, Rust vs Go for Backend` 


- **Mojo Unsuitable for Websites Compared to Rust**: A member inquired about using **Mojo** for websites versus **Rust**, but was advised against it due to **Mojo's** lack of specification for cryptographically sound code and a lacking IO story.
   - The suggestion was to use **Rust** with its production-ready libraries and frameworks, and its faster async capabilities, especially for applications requiring authentication or HTTPS.
- **Hardware-Backed AES in Mojo on Hold**: A member mentioned having a hardware-backed **AES** implementation in **Mojo**, but it doesn't run on older Apple silicon Macs and is not a full TLS implementation.
   - The developer is holding off on further **AES** work until a cryptographer is willing to write the software half, emphasizing the risks of non-experts implementing cryptographic functions.
- **SIMD for AES Implementations Explored**: Discussion involved using **SIMD** for **AES** and other algorithms, with a member noting that **x86** has **vaes** and similar features for **SIMD AES 128**.
   - It was mentioned that **ARM** has **SVE AES**, which is similar but not as well supported, highlighting hardware-level optimizations available for cryptographic functions.
- **Rust API to call into MAX**: Despite the challenges, a member expressed reluctance towards **Rust** due to the perception that it's not suitable for fast writing, seeking an easier solution for backend development, and the suggestion was to have the **Rust API call into it** and pass arguments along.
   - They were considering using **Python** but found it slow and unstable, leading to a plan to create a **Rust** project callable via **FFI**.
- **Go as a Middle Ground**: As an alternative, a member suggested **Go** as a decent middle ground that is also production ready, between Python/Mojo and Rust.
   - However, another member expressed concerns that too many microservices could make the project very big.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1354201595406975076)** (2 messages): 

> `CUDA-free, PTX for targeting NVIDIA GPUs` 


- **PTX Powers 'CUDA-Free' Mojo on NVIDIA**: Mojo directly generates **PTX** (Parallel Thread Execution) code to target NVIDIA GPUs, bypassing the need for CUDA.
   - This approach avoids dependencies on **cuBLAS**, **cuDNN**, and **CUDA C**, streamlining the development process.
- **Details on CUDA Free**: Confirmed by bradlarson, the team generates PTX directly and lowers from there.
   - There is *no* dependency on cuBLAS, cuDNN, or CUDA C.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1354048890420330568)** (9 messages🔥): 

> `Text Summarization with DSPy, SIMBA Optimizer, Output Refinement vs Assertions, BestOfN Module, Refine Module` 


- **Summarization Task Tackled with DSPy**: A member is exploring using **DSPy** for a text summarization task with **300** examples from an expert and is testing it out on a simple metric.
   - They are wondering if the optimizer can be more effective if it sees where exactly the summarization differed, and if there's a better approach to optimizing a prompt for text summarization.
- **SIMBA Optimizer Supports Granular Feedback**: A member suggested using the experimental optimizer `dspy.SIMBA` for the summarization task, which allows for providing feedback on differences between the generated summary and the ground truth.
   - The feedback can be returned via `dspy.Prediction(score=your_metric_score, feedback="stuff about the ground truth or how the two things differ")` to guide the optimization.
- **Output Refinement Doc Shared**: A member shared a link to a [DSPy tutorial on Output Refinement](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/) explaining `BestOfN` and `Refine` modules designed to improve prediction reliability by making multiple LM calls with different parameter settings.
   - The tutorial elaborates on how both modules stop when they have reached `N` attempts or when the `reward_fn` returns an award above the `threshold`.
- **BestOfN Module improves with Temperature Tweaks**: The `BestOfN` module runs a given module multiple times with different temperature settings to get an optimal result.
   - It returns either the first prediction that passes a specified threshold or the one with the highest reward if none meet the threshold.
- **Refine Module not as Composable?**: A member inquired if `Refine` is going to subsume assertions, and whether it's as granular and composable, since it wraps an entire module.
   - Another member responded that the composability can be managed by adjusting the module size, allowing for more explicit control over the scope.



**Link mentioned**: <a href="https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/">Output Refinement - DSPy</a>: The framework for programming—rather than prompting—language models.

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1354166265857572955)** (3 messages): 

> `ROCm Support, OpenCL Front End, AMD GPUs with Tinygrad` 


- **Legacy AMD GPUs get Tinygrad Boost**: Using **OpenCL frontend**, older AMD GPUs not supported by ROCm, such as those in a 2013 Mac Pro, *can* potentially run with Tinygrad.
   - The success may depend on the specific driver and the level of OpenCL support available on the system, so users should verify their *custom driver* compatibility.
- **ROCm Alternative for Older AMD**: ROCm does not support older AMD GPUs, but the **OpenCL frontend** in tinygrad might offer a workaround.
   - Success will vary based on specific driver versions and the extent of OpenCL support; experimentation is needed to confirm compatibility.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1353866401005309963)** (1 messages): 

> `Windsurf Creators Club, Vibe Coding Channel, Windsurf v1.5.8 Release` 


- **Windsurf launches Creators Club**: Windsurf is rewarding community members for content creation, offering **$2-4 per 1k views** and more details can be found at the [Windsurf Creators Club](https://whop.com/windsurf/).
- **Windsurf introduces 'Vibe Coding' channel**: A new channel has been created for 'vibe coders' to *enter the flow state*, chat, discuss, and share tips/tricks.
- **Windsurf v1.5.8 Released with Patch Fixes**: **Windsurf v1.5.8** is now released with patch fixes including cascade/memories fixes, Windsurf Previews improvements, and cascade layout fixes; An image of this release was also shared.



**Link mentioned**: <a href="https://whop.com/windsurf/)">no title found</a>: no description found

  

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
