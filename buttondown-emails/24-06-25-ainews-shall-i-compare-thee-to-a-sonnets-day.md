---
id: a7147fb1-bfa9-47d9-b734-cbb42c629bd8
title: '[AINews] Shall I compare thee to a Sonnet''s day?'
date: '2024-06-26T00:39:44.720143Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-sonnet
---

<!-- buttondown-editor-mode: plaintext -->**Claude 3.5 Sonnet is all you need.**

> AI News for 6/24/2024-6/25/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**415** channels, and **2614** messages) for you. 
Estimated reading time saved (at 200wpm): **260 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

 ![image.png](https://assets.buttondown.email/images/5bb77dd0-5fd1-4526-a833-52919ac96f78.png?w=960&fit=max) 

In realms of code, **Claude Sonnet** [ascends](https://twitter.com/lmsysorg/status/1805329822748655837),

A digital bard in **silicon** attire.

Through **Hard Prompts**' maze, its prowess transcends,

Yet skeptics question its confident fire.

**LMSYS** [crowns](https://twitter.com/lmsysorg/status/1805329826951348504) it silver, not far from gold,

Its **robust** mind tackles tasks with grace.

But whispers of doubt, like shadows, unfold:

Can **Anthropic's** child truly keep this pace?

In **Glif**'s [domain](https://twitter.com/fabianstelzer/status/1805326248261910552), it births **Wojak** dreams,

A **meme**-smith working at lightning speed.

Five minutes craft what impossible seems,

**JSON**'s extraction, a powerful deed.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Claude 3.5 Sonnet from Anthropic**

- **Impressive performance**: Claude 3.5 Sonnet secured the #1 spot in Coding Arena, Hard Prompts Arena, and #2 Overall, surpassing Opus at lower cost and competitive with GPT-4o/Gemini 1.5 Pro. [@lmsysorg](https://twitter.com/lmsysorg/status/1805329822748655837)
- **Overtakes GPT-4o**: Sonnet achieves #2 in "Overall" Arena, overtaking GPT-4o. [@lmsysorg](https://twitter.com/lmsysorg/status/1805329826951348504)
- **Robust in "Hard Prompts"**: Sonnet is also robust in the "Hard Prompts" Arena, which has a specific selection criteria. [@lmsysorg](https://twitter.com/lmsysorg/status/1805329824770400402)
- **Attitude and instruction-following critique**: Some suggest Sonnet's attitude implies capabilities it may not have, and that Anthropic's instruction-tuning is not as strong as OpenAI's. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1805316991848267832)

**Glif and Wojak Meme Generator**

- **Fully automated meme generator**: A Wojak meme generator was built in Glif in 5 min using Claude 3.5 for JSON generation, ComfyUI for Wojak images, and JSON extractor + Canvas Block to integrate. [@fabianstelzer](https://twitter.com/fabianstelzer/status/1805326248261910552)
- **JSON extractor block showcase**: This demonstrated the utility of Glif's new JSON extractor block for getting an LLM to generate JSON and split it into variables. [@fabianstelzer](https://twitter.com/fabianstelzer/status/1805326958986936679)
- **Edgy outputs from Claude**: Some of Claude 3.5's meme generator outputs were surprisingly edgy. [@fabianstelzer](https://twitter.com/fabianstelzer/status/1805326958986936679)

**Artifacts and Niche App Creation** 

- **Enabling otherwise unwritten software**: Artifacts makes it possible to quickly create niche apps, internal tools, or fun projects that would otherwise never be developed. [@alexalbert__](https://twitter.com/alexalbert__/status/1805261958134055409)
- **Example dual monitor visualizer**: Claude made a useful app in <5 minutes to visualize how dual monitors would fit on a desk - not groundbreaking but valuable given the speed of creation. [@alexalbert__](https://twitter.com/alexalbert__/status/1805261958134055409)

**Fusion Energy and Nuclear Fission**

- **Fusion not a near-term game changer**: Contrary to tech optimism, viable fusion today would barely impact energy economics in the next 100 years. [@fchollet](https://twitter.com/fchollet/status/1805343413669446022) 
- **Fission as existing clean energy solution**: Nuclear fission already provides near-unlimited clean energy, with 1970s plants cheaper to build and operate than hypothetical fusion ones. [@fchollet](https://twitter.com/fchollet/status/1805343413669446022)
- **Fuel cost a minor factor**: ~100% of fission electricity cost is from plants (80%) and transmission (20%), not fuel. Fusion maintaining 150M K plasma also won't be free to build/operate. [@fchollet](https://twitter.com/fchollet/status/1805343413669446022)

**AI Adoption and Productivity**

- **75% of workers using AI**: For desk jobs, it's becoming rare to find people not integrating AI into their work. The transition to AI-assisted productivity is underway. [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1805245591993160087)
- **Incremental productivity gains matter**: Even small productivity boosts from AI are highly valuable for busy people and startups. [@scottastevenson](https://twitter.com/scottastevenson/status/1805293727964938705)

**Together Mixture-of-Agents (MoA)**

- **MoA implemented in 50 LOC**: Together implemented their Mixture-of-Agents (MoA) approach in just 50 lines of code. [@togethercompute](https://twitter.com/togethercompute/status/1805289022241259803)

**Retrieval Augmented Generation (RAG) Fine-Tuning**

- **RAG fine-tuning outperforms larger models**: Fine-tuned Mistral 7B models using RAG match or beat larger models like GPT-4o & Claude 3 Opus on popular open source codebases, with 150x lower cost & 3.7x faster speed on Together. [@togethercompute](https://twitter.com/togethercompute/status/1805340734918476076)
- **Performance boost on codebases**: RAG fine-tuning improved performance on 4 out of 5 tested codebases. [@togethercompute](https://twitter.com/togethercompute/status/1805340738529771609)
- **Synthetic datasets used**: The models were fine-tuned on synthetic datasets generated by Morph Code API. [@togethercompute](https://twitter.com/togethercompute/status/1805341892978393344)

**Extending LLM Context Windows**

- **KVQuant for 10M token context**: KVQuant quantizes cached KV activations to ultra-low precisions to extend LLM context up to 10M tokens on 8 GPUs. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382644659896515) 
- **Activation Beacon for 400K context**: Activation Beacon condenses LLM activations to perceive 400K token context with limited window, trainable in <9 hrs on 8xA800 GPU. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382628742234179)
- **Infini-attention for 1M sequence length**: Google's Infini-attention uses compressive memory and local/long-term attention to scale a 1B LLM to 1M sequence length. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382639152775392)
- **LongEmbed for 32K context**: Microsoft's LongEmbed uses parallel windows, reorganized position IDs, interpolation to extend embedding model context to 32K tokens without retraining. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382655946768548)
- **PoSE for 128K context**: PoSE manipulates position indices in fixed window to mimic longer sequences, enabling 4K LLaMA-7B to handle 128K tokens. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382661223203256)
- **LongRoPE for 2M context**: Microsoft's LongRoPE extends pre-trained LLM context to 2M tokens while preserving short context performance, without long text fine-tuning. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382661223203256)
- **Self-Extend for long context**: Self-Extend elicits LLMs' inherent long context ability without fine-tuning by mapping unseen to seen relative positions via FLOOR. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382676859470113)
- **Dual Chunk Attention for 100K context**: DCA decomposes attention into intra/inter-chunk to let LLaMA-70B support 100K token context without continual training. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382682555089364)

**Many-Shot In-Context Learning**

- **Significant performance boosts**: Google finds major gains from many-shot vs few-shot in-context learning, even with AI-generated examples. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943)
- **Machine translation and summarization improvements**: Many-shot ICL helps low-resource language translation and nears fine-tuned summarization performance. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943)
- **Reinforced ICL with model rationales**: Reinforced ICL using model-generated rationales filtered for correctness matches or beats human rationales on math/QA. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943) 
- **Unsupervised ICL promise**: Unsupervised ICL, prompting only with problems, shows promise especially with many shots. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943)
- **Adapting to new label relationships**: With enough examples, many-shot ICL can adapt to new label relationships that contradict pre-training biases. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943)

**Miscellaneous**

- **Temporal dithering at 120 FPS**: Temporal dithering for color depth/supersampling is invisible at 120 FPS for most. 2D VR windows can exceed display resolution if 120 FPS jittered. [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1805295838211571878)
- **First-mover effect**: Existential proofs drive rapid catch-up. Sonnet-3.5 now slightly above once-leading GPT. 4-5 Sora clones at 70-80% quality in 4 months. [@DrJimFan](https://twitter.com/DrJimFan/status/1805265388256837842)
- **240T token dataset**: A 240T token dataset, 8x larger than previous SOTA, now available for LLM training. FineWeb's 15T is 48 TB. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805389473892778185)
- **iOS 18 motion cues**: iOS 18 adds on-screen dots that move with car to reduce phone motion sickness. [@kylebrussell](https://twitter.com/kylebrussell/status/1805211971731533848)
- **Open-source and corporate interests**: Difficult for open-source to be truly open when used strategically for corporate interests. [@fchollet](https://twitter.com/fchollet/status/1805315872988479788)

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Developments and Advancements**

- **Addressing AI inequality**: In an interview with Business Insider, Anthropic CEO Dario Amodei suggests that solutions [**beyond universal basic income are needed to tackle AI-driven inequality**](https://www.businessinsider.com/anthropic-ceo-dario-amodei-universal-basic-income-ubi-ai-inequality-2024-6). Microsoft AI CEO Mustafa Suleyman predicts that [GPT-6, expected in 2 years, will be able to follow instructions and take consistent actions](https://x.com/tsarnick/status/1805104305139232975), with some comparing the hype around GPT models to that of iPhones.

- **Powering the future of computing**: Bill Gates has unveiled a [**revolutionary nuclear reactor design aimed at powering the future of computing in Wyoming**](https://www.youtube.com/watch?v=LbWNXX4aLM8). Meanwhile, a demonstration showcases the potential of running [large AI models like a 3.3B BITNET on resource-constrained devices like a 1GB RAM retro handheld](https://streamable.com/gwt5fm).

**AI Models, Frameworks, and Benchmarks**

- **Anthropic's Claude making strides**: Anthropic's Claude 3.5 Sonnet model has [surpassed OpenAI's GPT-4o on the LMSYS Arena benchmark](https://x.com/lmsysorg/status/1805329822748655837). A user also showcased a [fractal explorer created by Claude, capable of displaying and zooming into four different fractals](https://v.redd.it/dhvzpd5h7h8d1).

- **New model releases**: The [Dolphin-2.9.3-Yi-1.5-34B-32K model has been released](https://i.redd.it/yq78y02vqi8d1.png), and the latest Chrome Canary build is now [capable of running the Gemini model locally](https://twitter.com/mortenjust/status/1805190952358650251). A user has also provided a [review of various models for summarization and instruction-following tasks](https://www.reddit.com/r/LocalLLaMA/comments/1dnavrt/update_model_review_for_summarizationinstruct_1gb/).

**AI Ethics, Regulation, and Societal Impact**

- **Frustration with AI companies**: A user expressed [frustration with OpenAI's delay in releasing the voice feature for GPT-4o](https://www.reddit.com/r/OpenAI/comments/1dn7dwq/im_sick_of_waiting_for_chatgpt_4o_voice_and_i/), leading them to lose respect for the company and switch to Anthropic's Claude AI. The importance of companies keeping promises and the impact on user trust was discussed.

- **The "Wild West" of AI**: It was suggested that we are currently in a ["Wild West" phase of AI development](https://www.reddit.com/r/singularity/comments/1dnxd9o/we_are_in_the_wild_west_of_the_preai_era/), where everything is rapidly evolving, compared to the prologue of a video game. The potential [impact of AI alignment on creative works](https://www.reddit.com/r/singularity/comments/1dnqe2i/what_are_some_of_the_best_books_movies_and_games/) was also discussed.

- **Skepticism and memes**: A [meme poked fun at LLM skeptics](https://i.redd.it/wbhy32608m8d1.png) who continue to doubt AI capabilities even as models advance, with comments suggesting skeptics may shift arguments to whether AI has a "soul" once human-level performance is reached.

**AI Applications and Use Cases**

- **Automation and creative industries**: Apple plans to [automate 50% of its iPhone final assembly line](https://9to5mac.com/2024/06/24/iphone-supply-chain-automation-workers/), replacing human workers with machines. Record labels have used AI tools [Udio and Suno to recreate versions of famous songs](https://www.reddit.com/r/singularity/comments/1dnlext/record_labels_were_able_to_basically_recreate/), raising questions about copyright and the music industry.

- **Photorealistic AI images and coding**: A user showcased [impressive photorealistic AI-generated images](https://www.reddit.com/gallery/1dntyba), with comments suggesting techniques like stock photo checkpoints and realism LoRAs to achieve natural results. The [DeepseekCoder-v2 model was praised for its coding performance](https://www.reddit.com/gallery/1dncebg).

- **Web UI automation**: Claude 3.5 was reported as the [first LLM reliably used for web UI automation and interaction](https://www.reddit.com/r/singularity/comments/1dnmatp/claude_35_is_the_first_llm_i_was_able_to_reliably/), outperforming GPT-4o for tasks like accessibility and front-end testing.

**AI Research and Development**

- **Educational resources and upcoming releases**: Andrej Karpathy's GitHub repository ["Let's build a Storyteller"](https://github.com/karpathy/LLM101n) aims to educate the community about building AI models. Excitement was expressed for the [upcoming release of Ray Kurzweil's book "The Singularity is Nearer"](https://www.reddit.com/r/singularity/comments/1dnbgkx/ray_kurzweils_book_is_being_released_in_a_few/).

- **New models and tools**: Salesforce released [Moirai-1.1, an updated time series foundation model](https://www.reddit.com/r/LocalLLaMA/comments/1dnajuy/salesforce_releases_moirai11_time_series/) for diverse forecasting tasks. The open-source project [WilmerAI was introduced](https://www.reddit.com/r/LocalLLaMA/comments/1dnsfh9/sorry_for_the_wait_folks_meet_wilmerai_my_open/) to maximize local LLM potential through prompt routing and multi-model workflow management. [Rensa, a high-performance MinHash implementation](https://www.reddit.com/r/LocalLLaMA/comments/1dn7erd/rensa_a_high_performance_minhash_implementation/), was also announced for fast similarity estimation and deduplication.

**Miscellaneous**

- **Brain cells and AI**: The use of [human brain cells to power computers for R&D](https://www.reddit.com/r/singularity/comments/1dnwyuu/human_brain_cells_now_running_computers_for_rd/) was discussed, with comparisons to the Matrix and questions about implications for AGI/ASI development. It was clarified that these are cells, not entire brains, and lack the complexity for consciousness.

- **Multimodal AI and solar energy**: Interest was expressed in [Meta's multimodal AI model, Chameleon](https://www.reddit.com/r/LocalLLaMA/comments/1dnm1v1/have_any_of_you_tried_metas_multimodal_chameleon/), noting a lack of community discussion. An Economist article predicted that [solar energy, with its increasing affordability, will become a major energy source](https://www.economist.com/interactive/essay/2024/06/20/solar-power-is-going-to-be-huge).

- **Unique AI projects**: The font [llama.ttf, which also functions as a language model](https://fuglede.github.io/llama.ttf/), was introduced. A [forensic analysis of the system prompt used in Anthropic's Claude 3.5 Sonnet](https://tyingshoelaces.com/blog/forensic-analysis-sonnet-prompt) introduced the concept of "Artifacts" in AI prompts.

---

# AI Discord Recap

> A summary of Summaries of Summaries

## Claude 3 Sonnet

**1. LLM Advancements and Benchmarking**

- **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** from Meta has rapidly risen to the top of leaderboards like **ChatbotArena**, outperforming models like **GPT-4-Turbo** and **Claude 3 Opus** in over 50,000 matchups.

- New models like **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** from IBM enhance instruction following for code tasks, while **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** boasts **236B parameters**. 

- Skepticism surrounds certain benchmarks, with calls for credible sources like Meta to set realistic LLM assessment standards.

**2. Optimizing LLM Inference and Training**

- **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** promises a 4x reduction in communication overhead for large model training on GPUs.

- The **[vAttention](https://arxiv.org/abs/2405.04437)** system dynamically manages KV-cache memory for efficient LLM inference without PagedAttention.

- **[QServe](https://arxiv.org/abs/2405.04532)** introduces **W4A8KV4 quantization** to boost cloud-based LLM serving performance on GPUs.

- Techniques like **[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** explore parallel token decoding for reduced inference latency.

**3. Open-Source AI Frameworks and Community Efforts** 

- **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** supports diverse dataset formats for instruction tuning and pre-training LLMs.

- **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** powers a new course on building agentic RAG systems with Andrew Ng.

- **[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)** is open-sourced, claiming to be the best LLM for "unsexy data tasks".

- **[Modular](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)** teases Mojo's potential for Python integration and AI extensions like _bfloat16_.

**4. Multimodal AI and Generative Modeling Innovations**

- **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** focuses on elevated chat interactions, while **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** refines coding abilities.  

- The **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** model brings powerful AI chatbots to browsers via WebGPU.

- Combining **Pixart Sigma + SDXL + PAG** aims to achieve **DALLE-3**-level outputs, with potential for further refinement through fine-tuning.

- The open-source **[IC-Light](https://github.com/lllyasviel/IC-Light)** project focuses on improving image relighting techniques.

## Claude 3.5 Sonnet

1. **New LLMs Shake Up the Leaderboards**:

   - The [Replete-Coder-Llama3-8B model](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) has gained attention across multiple discords for its proficiency in over 100 programming languages and advanced coding capabilities.

   - [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) with 236B parameters and [Hathor_Fractionate-L3-8B-v.05](https://huggingface.co/Nitral-AI/Hathor_Fractionate-L3-8B-v.05) were discussed for their performance in various tasks.

   - Skepticism about benchmarks was a common theme, with users emphasizing the need for real-world testing over leaderboard rankings.

2. **Open-Source Tools Empower AI Developers**:

   - [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) gained traction for supporting diverse dataset formats in LLM training.

   - [LlamaIndex](https://www.latent.space/p/llm-training-2024) was highlighted for its integration with DSPy, enhancing RAG capabilities.

   - The release of [llamafile v0.8.7](https://www.phoronix.com/news/Llamafile-0.8.7-Released) brought faster quant operations and bug fixes, with hints at potential Android compatibility.

3. **Optimization Techniques Push LLM Boundaries**:

   - The [Adam-mini optimizer](https://arxiv.org/abs/2406.16793) sparked discussions across discords for its ability to reduce memory usage by 45-50% compared to AdamW.

   - [Sohu's AI chip](https://x.com/Etched/status/1805625693113663834) claims to process 500,000 tokens per second with Llama 70B, though the community expressed skepticism about these performance metrics.

4. **AI Ethics and Security Take Center Stage**:

   - A [Remote Code Execution vulnerability (CVE-2024-37032)](https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032) in the Ollama project raised concerns about AI security across multiple discords.

   - Discussions on [AI lab security](https://x.com/jordanschnyc/status/1805340489391997177) highlighted the need for enhanced measures to prevent risks like "superhuman hacking" and unauthorized access.

   - The AI music generation lawsuit against Suno and Udio, as reported by [Music Business Worldwide](https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/), sparked debates on copyright and ethical AI training across communities.

## Claude 3 Opus

**1. New LLM Releases and Benchmarking**

- **Replete-Coder-Llama3-8B** model impresses with coding proficiency in 100+ languages and uncensored training data ([Hugging Face](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B)).
- Discussions on the reliability of benchmarks, with some arguing they don't reflect real-world performance (Unsloth AI Discord).
- **DeepSeek-V2** outperforms GPT-4 on some tasks in the **AlignBench** and **MT-Bench** benchmarks ([Twitter announcement](https://x.com/deepseek_ai/status/1787478986731429933)).

**2. Optimizing LLM Performance and Efficiency**

- **Adam-mini optimizer** reduces memory usage by 45-50% compared to AdamW with similar or better performance ([arXiv paper](https://arxiv.org/abs/2406.16793)).
- Quantization techniques like **AQLM** and **QuaRot** enable running large models on single GPUs, e.g., **Llama-3-70b** on RTX3090 ([AQLM project](https://github.com/Vahe1994/AQLM)).
- **Dynamic Memory Compression (DMC)** boosts transformer efficiency, potentially improving throughput by up to 370% on **H100 GPUs** ([DMC paper](https://arxiv.org/abs/2403.09636)).

**3. Open-Source AI Frameworks and Collaborations**

- **Axolotl** supports various dataset formats for LLM instruction tuning and pre-training ([Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47)).
- **LlamaIndex** integrates with a new course on building agentic RAG systems by Andrew Ng ([DeepLearning.AI course](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)).
- **Mojo** language hints at future Python integration and AI-specific extensions like _bfloat16_ (Modular Discord).
- **StoryDiffusion**, an open-source alternative to Sora, is released under MIT license ([GitHub repo](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)).

**4. Multimodal AI and Generative Models**

- **Idefics2 8B Chatty** and **CodeGemma 1.1 7B** models focus on chat interactions and coding abilities, respectively ([Twitter](https://twitter.com/sanhestpasmoi/status/1787503160757485609) [posts](https://twitter.com/reach_vb/status/1786469104678760677)).
- **Phi 3** brings powerful AI chatbots to browsers using WebGPU ([Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)).
- Combining **Pixart Sigma**, **SDXL**, and **PAG** aims to achieve **DALLE-3**-level outputs (Latent Space Discord).
- **IC-Light**, an open-source project, focuses on image relighting techniques ([GitHub repo](https://github.com/lllyasviel/IC-Light)).

## GPT4O (gpt-4o-2024-05-13)


1. **Performance Improvements and Technical Fixes**:
   - **PyTorch Tensor Alignment Issue Gets Attention**: Users discussed aligning PyTorch tensors for efficient memory usage and referenced code and documentation to address issues like `torch.ops.aten._weight_int4pack_mm` [source code](https://github.com/pytorch/pytorch/pull/110914).
   - **LangChain Enhancements**: Members praised [LangChain Zep](https://www.getzep.com/) integration, which provides persistent AI memory, summarizing conversations for effective long-term use.
   - **LazyBuffer Bug Identified in Tinygrad**: A problem with 'LazyBuffer' not having an attribute 'srcs' in Tinygrad was documented, suggesting fixes like `.contiguous()` and using Docker for CI debug [Dockerfile here](https://github.com/Qazalin/containers/blob/main/tinygrad/Dockerfile#L1C1-L14C1).

2. **Ethical and Legal Challenges in AI**:
   - **AI Music Generators Sued for Copyright Infringement**: Major record companies are suing Suno and Udio for unauthorized training on copyrighted music, raising questions on ethical AI training practices [Music Business Worldwide report](https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/).
   - **Carlini Defends His Attack Research**: Nicholas Carlini defended his research on AI model attacks, stating that they highlight crucial AI model vulnerabilities [blog post](https://nicholas.carlini.com/writing/2024/why-i-attack.html).
   - **Probllama's Security Breach**: Rabbithole's security disclosure revealed critical vulnerabilities due to hardcoded API keys, potentially enabling widespread misuse across services like ElevenLabs and Google Maps [full disclosure](https://rabbitu.de/articles/security-disclosure-1).

3. **New Releases and AI Model Innovations**:
   - **EvolutionaryScale’s Breakthrough with ESM3**: The ESM3 model simulates 500M years of evolution, earning $142M in funding and aiming to achieve new heights in programming biology [funding announcement](https://x.com/pdhsu/status/1805563282746446116).
   - **Gradio's New Feature Set**: The latest release, Gradio v4.37, introduced a revamped chatbot UI, dynamic plots, and GIF support, alongside performance improvements for a better user experience [changelog](https://www.gradio.app/changelog).
   - **Rising AI Models on OpenRouter**: New AI models like [AI21's Jamba Instruct](https://openrouter.ai/models/ai21/jamba-instruct) and NVIDIA's [Nemotron-4 340B](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct) were added to the platform, integrating diverse capabilities for various applications.

4. **Dataset Management and Optimization**:
   - **Addressing RAM Issues in Dataset Loading**: Techniques like using `save_to_disk`, `load_from_disk`, and enabling `streaming=True` were discussed to mitigate memory issues when handling large datasets in AI models.
   - **Minhash Optimization Performance Boost**: A member boasted a 12x performance improvement for minhash calculations using Python, sparking interest and collaboration for further optimization [GitHub link](https://github.com/beowolx/rensa).

5. **Conferences, Events, and Community Engagement**:
   - **AI Engineer World's Fair Highlights**: Excitement builds as engineers anticipate the AI Engineer World's Fair with keynotes and engaging talks, including insights from the LlamaIndex team [event details](https://twitter.com/llama_index/status/1805366745877856363).
   - **Detecting Bots and Fraud in LLMs**: An event on June 27 will feature Unmesh Kurup from hCaptcha discussing strategies to counteract LLM-based bots and fraud detection in modern AI security [event registration](https://lu.ma/y6hz8cod?utm_source=diskord).
   - **OpenAI's ChatGPT Desktop App for macOS**: The new app allows macOS users to access ChatGPT with enhanced features, marking a significant step in AI usability and integration [ChatGPT for macOS](https://openai.com/chatgpt/mac/).

---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Brain Over Brawn in Python**: In a heated discussion on numerical precision, Python users shared code snippets for large float calculations that often result in an `OverflowError`. Solutions revolved around alternative methods to compute high-power floats without precision loss.

- **Memory Mayhem with AI Datasets**: A user grappling with memory limitations while loading datasets even with 130GB of RAM got tips on disk storage techniques. Options like `save_to_disk`, `load_from_disk`, and the `streaming` flag were suggested to alleviate the issue.

- **Model Miniaturization Mysteries**: Conversation turned to quantization as a method for running hefty AI models on modest hardware, balancing performance and precision.

- **Graphviz Glitches in Git**: Users trying to use `graphviz` with Hugging Face spaces faced `PATH` errors, and offered up wisdom on system configurations to fix the issue.

- **Skills, Not Fields, Foster Opportunities**: Among skills discussions, a user emphasized the value of project involvement over specific technical fields when considering career opportunities in tech.

- **Excitement Over LLM JSON Structuring**: A Langchain Pydantic Basemodel user sought advice for structuring documents in JSON to avoid table structure confusion, sparking enthusiasm among peers.

- **Cybersecurity Strategies on Standby**: With a June 27 event on bot and fraud detection announced, the community is gearing up to learn advanced tactics from hCaptcha's ML Director.

- **Tokenization Talk Turns Contentious**: [Apehex](https://huggingface.co/blog/apehex/tokenization-is-a-dead-weight) stirred up the pot with an argument against tokenization, advocating for direct Unicode encoding. This generated a buzzing discussion on the trade-offs between various encoding approaches.

- **Personalizable Maps and Media-friendly Start Pages**: Creative coders showcased their works like the [Cityscape Prettifier](https://github.com/C1N-S4/cityscape-prettifier), which crafts stylized city maps, and the browser start page extension [Starty Party](https://startyparty.dev/) designed for media enthusiasts.

- **Progress in the Paper Landscape**: Members of the reading group sought out and recommended research on topics like contamination in coding benchmarks while others hinted at imminent code releases tied to updated papers.

- **Troubleshooting Tools in Vision**: Users found `hf-vision/detection_metrics` error-prone due to dependency snags and discussed ongoing issues documented on GitHub, such as the problem mentioned in [this issue](https://github.com/huggingface/evaluate/issues/111#issuecomment-1855872972).

- **Looking for LLM Expertise on Tabular Data**: A query was raised about open-source projects capable of conversing about tendencies within tabular data, without delving into modeling or predictions. Meanwhile, a community member expressed an intention to contribute to a [PR about RoBERTa-based scaled dot product attention](https://github.com/huggingface/transformers/pull/30510), despite facing repository access barriers.

- **Gradio Amps Up Chatbots and Plots**: The release of Gradio v4.37 brought a reimagined chatbot UI and dynamic plots, along with the ability to nest components like galleries and audio in chat. GIF support got a nod too, as detailed in Gradio's [changelog](https://www.gradio.app/changelog).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Aligning PyTorch Tensors**: A user sought advice on how to **align PyTorch tensors** in memory, which is critical for efficiently loading tensor pairs using `float2` due to alignment issues.

- **Understanding Dequantization in PyTorch**: In a bustling discussion, engineers dissected the function `torch.ops.aten._weight_int4pack_mm`, referring to [GitHub source code](https://github.com/pytorch/pytorch/pull/110914) for better understanding dequantization and matrix multiplication, and bemoaned the lack of informative autogenerated documentation.

- **Quantum Quake Challenge**: A **13kb JavaScript** remake of Quake, called Q1K3, was presented through a [YouTube making-of video](https://www.youtube.com/watch?v=F9MLpuvPDio), along with ways to [play the game](https://phoboslab.org/q1k3/) and further discussion provided in a [blog post](https://phoboslab.org/log/202).

- **Generating Trouble at HF**: Issues with `HFGenerator` post-cache logic update in the transformers library were highlighted, prompting a need for a rewrite to fix problems with changing prompt lengths causing recompilation when using `torch.compile`.

- **Software Meets Hardware**: Engineers shared a breakthrough with a **Windows build cuDND fix merged**, discussed stability challenges while training on **H100 with cuDNN**, mused over **AMD GPU support**, highlighted a **PR for on-device reductions** to limit data transfers, and talked roadmaps including **Llama 3 support and v1.0** aiming for optimizations like rolling checkpoints and a **StableAdamW optimizer**.

- **Evaluating AMD's Future**: An article assessing the performance of AMD's upcoming **MI300x** was linked, indicating interest in the direction of AMD's GPU developments.

- **PyTorch Device Allocation Investigated**: A technical fix for a **PyTorch tensor device call issue** was proposed with reference to a line in `native_functions.yaml`, which might help in resolving device call mismatches in tensors.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama3-8B Packs a Punch in Over 100 Languages**: Engineers are discussing the [Replete-Coder Llama3-8B model](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B), touting its prowess in advanced coding across a multitude of languages and its unique dataset that eschews duplicates.
- **Benchmarks Under Microscope**: The reliability of benchmarks sparked debate, with recognition that benchmarks can often misrepresent practical performance; this implies a need for more holistic assessment methods.
- **Optimizer That Lightens the Load**: The *[Adam-mini optimizer](https://arxiv.org/abs/2406.16793)* has captured attention for its potential to deliver AdamW-like performance with significantly decreased memory usage and increased throughput.
- **Ollama's Achilles' Heel Patched**: Discussion around the [CVE-2024-37032 vulnerability](https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032) in the Ollama project emphasizes a swift response and the urgency for users to update to the remediated version.
- **GPUs in Tandem**: For those experiencing multi-GPU snags with Unsloth, the consensus involves practical workarounds such as limiting CUDA devices, with insights found on [GitHub issue 660](https://github.com/unslothai/unsloth/issues/660), while challenges with model fine-tuning are being tackled with novel techniques like model merging.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Confusion Over Perplexity's Pro Features**: Users voiced concerns over **Perplexity AI's** features, with the primary issue being the random switch of UI language from English to other languages, and the confusion between Pro Search and standard search functionalities. Reports also surfaced regarding problems with generating download links for users with a PRO subscription, and questions were raised about whether a Pro plan includes the "Pages" feature for international content localization.

- **Starliner Woes and Local News Highlights Hit YouTube**: A [YouTube video](https://www.youtube.com/embed/xUsxGDrwzls) was discussed, highlighting issues with the **Starliner spacecraft** and the latest victory of the Panthers. Additionally, the appointment of **Samantha Mostyn** as Australia's new Governor General caught users' attention.

- **Perplexity API Fails to Deliver Complete Output**: Users leveraging the **Perplexity API** reported it failed to include citations and images in its summarization, suggesting the use of code blocks as a workaround.

- **Seeking Pro Troubleshooting**: A member expressed disappointment in requiring Pro features for much-needed work and was directed to seek assistance from "f1shy" to potentially resolve the issue.

- **Technical Content Curated**: There was a mention of **Jina Reranker v2 for Agentic RAG**, referring to its ultra-fast, multilingual function-calling, and code search capabilities, which was noted as valuable information for the technical audience.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**RTX 3090 Can't Handle the Heat**: Users express frustration with an **RTX 3090 eGPU setup** failing to load larger models like **Command R (34b) Q4_K_S**, leading to suggestions for **exl2 format** utilization for improved VRAM use, despite a noted scarcity of tools and GUI options for exl2.

**Confusion Cleared on Different Llama Flavors**: Clarification was provided for **Llama 3 model variants**: the unlabeled **Llama 3 8B** is the base model, set apart from the *Llama 3 8B text* and *Llama 8B Instruct*, which are finetuned for specific tasks.

**Model Marvels and Mishaps**: Praise was given for **Hathor_Fractionate-L3-8B-v.05**'s creativity and **Replete-Coder-Llama3-8B**'s coding proficiency, while **DeepSeek Coder V2** was flagged for high VRAM demands, and **New Dawn 70b** was applauded for its role-play capabilities with contexts up to 32k.

**Tech Support Troubles**: Issues surfaced with **Ubuntu 22.04 network errors** in LM Studio, with possible remedies like disabling **IPv6**, and it was noted that **LM Studio does not currently support Lora adapters or image generation**.

**Hardware Banter and Bottlenecks**: A humorous exchange highlighted the chasm between the affordability of high-performance GPUs and their necessity for advanced AI work, with older rigs mockingly deemed as belonging to *"the 1800s"*.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI Music Generators Face Legal Troubles**: Major record companies including **Sony Music Entertainment** and **Universal Music Group** have initiated lawsuits against AI music generators Suno and Udio for copyright infringement, as coordinated by the **RIAA**. Discussions in the community centered around the ethics of AI training and considered the possibility of creating an open-source music model that avoids these Copyright issues. [Music Business Worldwide report](https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/).

- **Carlini Clarifies His Rationale for Attack Papers**: Nicholas Carlini responded to criticisms, particularly from Prof. Ben Zhao, with a [blog post](https://nicholas.carlini.com/writing/2024/why-i-attack.html) defending his reasons for writing attack research papers, which spark important dialogues on AI model vulnerabilities and community standards.

- **Glazing Over Controversial Content**: The Glaze channel was deleted amid speculations of cost, legal worries, or an attempt to erase controversial past statements, highlighting the ongoing tension between content moderation and free discussion in the AI research community.

- **Nightshade's Legal Haze**: The AI protection scheme called Nightshade was flagged for potential legal and ethical risks before its official release, reflecting the community's concerns with the complexities of deploying model protection measures. Details of these concerns can be found in the article "[Nightshade: Legal Poison Disguised as Protection for Artists](https://undeleted.ronsor.com/nightshade-legal-poison/)."

- **Controversy over Model Poisoning**: A contentious debate surrounded Prof. Zhao's endorsement of model poisoning as a legitimate strategy, underlining the divisive issue of tampering with AI models and the potential backlash from within the engineering community.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**ChatGPT App Lands on macOS**: The **ChatGPT desktop app** is now available for **macOS**, offering streamlined access via Option + Space shortcut and enhanced features for chatting about emails, screenshots, and on-screen content. Check it out at [ChatGPT for macOS](https://openai.com/chatgpt/mac/).

**Animated Discussion Over Token Size**: Engineers debated token context window sizes across models like **ChatGPT4**, with **ChatGPT4** offering **32,000 tokens for Plus users** and **8,000 tokens for free users**, while other models like **Gemini or Claude** provide larger capacities, with Claude reaching 200k tokens.

**Custom GPT Misconceptions Cleared**: Members clarified the differences between **CustomGPT's document attachment feature** and actual model training. CustomGPT doesn't offer persistent memory across chats but rather augments the model's knowledge with external documents.

**GPT Struggles Reported**: Discord users reported issues with GPT’s handling of large documents and the provision of incorrect information from uploaded files, along with performance hiccups and JSON output difficulties, highlighting a need for better handling of complex queries and outputs.

**AI Chips and Evolutionary Breakthroughs**: Shared excitement emerged around *[EvolutionaryScale’s ESM3](https://www.evolutionaryscale.ai/blog/esm3-release)*, simulatively reproducing 500M years of biological evolution, and *[Sohu’s AI chip](https://x.com/etched/status/1805625693113663834?s=46&t=tqMTDs9oHX6jLrqq8NrMdw)*, capable of outperforming current GPUs in running transformer models.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Artistic Flair Sells AI Art**: Skilled individuals with a background in art are finding success selling AI-generated art, illustrating that advanced prompting skills coupled with an existing art foundation might be key to commercial success.
- **Troubleshooting CUDA and PyTorch**: Engineers experienced issues with accessing a Github repository and encountered a *RuntimeError* pertaining to PyTorch and GPU compatibility, with the consensus advising a compatibility check between CUDA and PyTorch versions.
- **Skepticism Around Open Model Initiative**: The Open Model Initiative sparked divisive opinions among engineers, with some questioning its integrity on ethical grounds, despite its support by communities like reddit.
- **Concern Over Google Colab Usage**: Users are worried about potential restrictions on Google Colab due to heavy use of Stable Diffusion, suggesting alternatives like runpod, which costs about 30 cents an hour for similar usage.
- **Future of Stability.AI in Question**: Doubts were voiced about the longevity of Stability.AI in the competitive market if they don't address issues and reverse censorship with products like SD3, challenging their current and future market position.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Generative Hypernetworks Get LoRA-fied**: Hypernetwork discussions surfaced about generating **Low-Rank Adaptations (LoRAs)**, indicating hyperparametric flexibility and signaling a move towards more customizable AI models, particularly those targeting specificity with a rank of 1.

- **Nuances of "Nous"**: Clash of linguistics led to a clarification: "Nous" in Nous Research nods to intelligence (Greek origin), rather than the assumption of the French meaning "our," spotlighting the blend of collective passion and intellect within the community.

- **Security Alert: Probllama Vulnerability Exposed**: Twitter buzz highlighted a Remote Code Execution (RCE) vulnerability in **Probllama**, detailed in [this thread](https://x.com/sagitz_/status/1805261557481312623), and is now assigned **CVE-2024-37032**.

- **Enter the Llama-Verse with Coder Llama3-8B**: [Replete-AI/Replete-Coder-Llama3-8B](https.detoken://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) stormed into the AI scene, asserting prowess in over 100 programming languages and potential to reshape the coding landscape with its 3.9 million lines of curated training data.

- **LLM Study Unpacks Decision Boundaries**: An [arXiv paper](https://arxiv.org/abs/2406.11233) reveals non-smooth, intricate decision boundaries in LLMs' in-context learning, contrasting the expected behavior from conventional models such as Decision Trees. This study instigates new considerations for model interpretability and refinement.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **New AI Models Hit OpenRouter**: OpenRouter presented its 2023-2024 model lineup, introducing [AI21's Jamba Instruct](https://openrouter.ai/models/ai21/jamba-instruct), [NVIDIA's Nemotron-4 340B Instruct](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct), and [01-ai's Yi Large](https://openrouter.ai/models/01-ai/yi-large). However, they also reported an issue with incorrect data on the Recommended Parameters tab, assuring users that a fix is underway.

- **From Gaming to AI Control**: Developer **rudestream** showcased an AI integration for *Elite: Dangerous* which uses **OpenRouter's free models** to enable in-game ship computer automation. While the project is gaining attention, the developer is seeking further enhancements with **Speech-to-Text** and **Text-to-Speech** capabilities, as demonstrated via [GitHub](https://github.com/RatherRude/Elite-Dangerous-AI-Integration) and a [demo video](https://www.youtube.com/watch?v=nvuCwwixvxw).

- **Testing Delays and AI Development Reflections**: OpenRouter delayed an announcement post for further tests on the new Jamba model while a user inspired a discussion on the state of AI innovation, suggesting enthusiasts listen to François Chollet's [insights on AI’s future](https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/).

- **Jamba Instruct Model Glitches and Best Practices**: Users faced technical problems with **AI21’s Jamba Instruct** model; even after rectifying privacy settings, inconsistencies persisted. Separately, the community exchanged prompt engineering strategies, pointing to [Anthropic Claude’s guidelines](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) for reference.

- **The AI Personality Debate is Real**: Debates sparked on the neutrality of large language models (LLMs) with consensus tilting towards preferring less restrictive AI that engage in more original and dynamic conversations, as opposed to echoing neutral, "text wall" responses.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Typography Meets AI with llama.ttf**: Engineers explored [llama.ttf](https://fuglede.github.io/llama.ttf/?utm_source=changelog-news), an innovative font file that merges a large language model with a text-based LLM inference engine, harnessing HarfBuzz's Wasm shaper. This clever merger prompts discussions on unconventional uses of AI in software development.

- **Karpathy Kicks Off AI Fanfare**: [Andrej Karpathy](https://x.com/karpathy/status/1805328398920958214?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) stirred excitement with the announcement of the AI World's Fair in San Francisco, emphasizing the need for volunteers amidst the already sold-out event, signifying escalating interest in AI community gatherings.

- **MARS5 TTS Model Breakthrough**: The tech community introduced [MARS5 TTS](https://x.com/reach_vb/status/1805336863101620699), an avant-garde open-source text-to-speech model that promises unmatched prosodic control and the capability of voice cloning with minimal audio input, sparking interest in its underlying architecture.

- **EvolutionaryScale’s $142M Seed Shocks the Sector**: The announcement of [EvolutionaryScale's](https://x.com/pdhsu/status/1805563282746446116) colossal $142M fundraising round to support the development of their ESM3 model, intended to simulate half a billion years of protein evolution, highlights interests in marrying AI with biology.

- **Sohu's Speed Stuns Nvidia**: Discussions revolved around [Sohu](https://x.com/Etched/status/1805625693113663834), the newest AI chip on the block that claims to outstrip Nvidia's Blackwell by processing 500,000 tokens per second with Llama 70B. This catalyzed debates on benchmarking methodologies and whether these claims stack up in real-world scenarios.

- **Podcasting the Future of AI**: The **Latent Space** podcast teasers brought excitement with a [preview of the AIEWF conference](https://x.com/latentspacepod/status/1805468252828844052) and discussions on **DBRX** and **Imbue 70B**, shaping up debates around the current landscape of Large Language Models (LLMs) and innovating AI media content [[Listen here](https://www.latent.space/p/llm-training-2024)].



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Catch LlamaIndex on Tour**: The LlamaDate team will be at the AI Engineer World's Fair, with a [keynote](https://twitter.com/llama_index/status/1805366745877856363) by @jerryjliu0 on the Future of Knowledge Assistants happening on Wednesday, 26th. Don't miss it! 

- **RAG Receives a DSPy Boost**: LlamaIndex has bolstered RAG capabilities through a collaboration with DSPy, optimizing retriever-agent interaction with superior data handling. Full details of the enhancement can be found in their announcement [here](https://twitter.com/llama_index/status/1805622004030284036).

- **Dimensional Puzzle Solved in PGVectorStore**: A matchup error spotted by a user, triggered by an embedding dimension mismatch with the bge-small model, was ironed out once the `embed_dim` was set correctly to maintain consistency.

- **RAG Architecture Unveiled**: Resources on RAG's inner workings were shared, directing users to diagrams and detailed documentation on [concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) and [agent workflows](https://docs.llamaindex.ai/en/stable/understanding/agent/basic_agent/), along with a foundational [paper](https://arxiv.org/abs/2312.10997) on the subject.

- **Prompt Templating Potential with vllm**: The dialogue on prompt templates in vllm clarified the use of `messages_to_prompt` and `completion_to_prompt` function hooks for integrating few-shot prompting into LlamaIndex modules.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Git Logs for Efficient Changelog Peeking**: Engineers discovered that using "git log -S" allows for searching history of specific code changes, valuable when navigating the [Mojo changelog](https://github.com/modularml/mojo/blob/1b79ef249f52163b0bafbd10c1925bfc81ea1cb3/docs/changelog.md), especially since documentation rebuilds eliminate searchable history older than three months.

**Mojo and MAX Interconnected Potential**: Discussions indicated that while Mojo currently may not support easy simultaneous use with Torch, a future integration aims to harness both Python and C++ capabilities. Additionally, for AI model serving, the MAX graph API serde is in development, promising future support for custom AI models with frameworks like Triton.

**MAX 24.4 Embraces MacOS and Local AI**: With the [release of MAX 24.4](https://www.modular.com/blog/whats-new-in-max-24-4-max-on-macos-fast-local-llama3-native-quantization-and-gguf-support), MacOS users can now leverage the toolchain for building and deploying Generative AI pipelines, introducing support for local models like Llama3 and native quantization.

**SIMD & Vectorization Hot Topics for Mojo**: Engineers are examining SIMD and vectorization within Mojo, where hand-rolled SIMD, LLVM's loop vectorizer status, and features like SVE support surface as critical considerations. These discussions spurred recommendations to submit features or PRs for better alignment to SIMD standards.

**Nightly Compiler Updates Drive Mojo Optimizations**: Issues and enhancements are flowing with Mojo nightly versions `2024.6.2505` and `2024.6.2516`, where performance gains via list autodereferencing and better reference handling in dictionaries are emphasized. Troubleshooting highlights include compile-time boolean expression dealing, with reference to [specific commits](https://github.com/modularml/mojo/commit/57ab0baf2352abee9e83e30967634c2be357afd5).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LingOly Benchmark Under Scrutiny**: Engineers discussed the potential flaws in the [LingOly benchmark](https://arxiv.org/abs/2406.06196v2), questioning its scope and scoring, particularly the risks of memorization when test sets are public.
- **Celebrating Rise of the Ethical AI Makers**: The community recognized Mozilla's [Rise25 Awards](https://blog.mozilla.org/en/mozilla/mozilla-announces-finalists-for-the-2nd-annual-rise25-awards), commending honorees for contributions to ethical and inclusive AI.
- **The MoE Advantage in Parameter Scaling**: Sparse parameters in Mixture of Experts (MoE) emerge as a preferred scaling route, challenging the deepening of architectures.
- **Backdoor Threats in Federated Learning and AI**: Discussions focused on the potential for adversarial backdoor attacks in federated learning and the implications for open weights models, referring to research in [this paper](https://arxiv.org/abs/2206.10341).
- **Importance of Initializations in AI Highlighted**: A member cites *"Neural Redshift: Random Networks are not Random Functions"* in a discussion about the underestimated structural role of initializations in neural networks, directing to [AI koans](http://www.catb.org/esr/jargon/html/koans.html) for levity.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI Welcomes Multi App**: [Multi has announced](https://multi.app/blog/multi-is-joining-openai) it will become part of OpenAI, aiming to explore collaborative work between humans and AI, offering services until July 24, 2024, with post-termination data deletion plans detailed.
  
- **Apple Bets on ChatGPT over Llama**: [Apple turns down Meta's AI partnership offer](https://archive.is/uUv1L), favoring an alliance with OpenAI's ChatGPT and Alphabet's Gemini, mainly over privacy practice concerns with Meta.

- **Rabbithole's Hardcoded Key Hazard**: [A codebase security breach](https://rabbitu.de/articles/security-disclosure-1) at rabbitude has exposed hardcoded API keys, risking unauthorized access to a plethora of services including ElevenLabs and Google Maps, and prompting discussions on potential misuse.

- **Nvidia's Status Quo Shattered**: [Market shifts reflect a realization](https://www.ft.com/content/7332b1f8-cf7c-4bfa-82f4-88d0deb23f98) that Nvidia isn’t the sole giant in the GPU landscape; Imbue AI's release of a toolkit for 70B parameter models is received with both skepticism and interest.

- **AI Lab Security Needs Dire Attention**: Insights from an interview with Alexandr Wang underlined the pressing need for stringent security in AI labs, hinting at how AI poses risks potentially more significant than nuclear weapons through avenues like "superhuman hacking."



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Llama3-8B Coder AI Shakes Up the Community**: The [Replete-Coder-Llama3-8B model](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) has impressed engineers with its proficiency in over 100 languages and advanced coding capabilities, though it's not tailored for vision tasks.

**Technical Triumphs Tangled With Quirks**: Engineers found success using `claude-3-5-sonnet-20240620` for code executions after troubleshooting flags, but compatibility and function support issues point to the need for refined model configurations.

**Vision Feature Frustration Persists**: Despite concerted efforts, users like daniel_farinax struggle with sluggish processing times and CUDA memory errors when employing vision capabilities locally, spotlighting the cost and complexity of emulating OpenAI's vision functions.

**Limited Local Vision Functionality Sparks Debate**: Users attempt to activate vision features such as `--local --vision` with minimal success, revealing a gap in Llama3's capabilities and the desire for more accessible and efficient local vision task execution.

**Single AI Content Sidenote**: A lone remark about the unsettling nature of AI-generated videos suggests an underlying concern for user m.0861, though not expanded into a broader discussion within the engineering community.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **ChatOllama Wrangling Simplified**: Engineers experimenting with **Ollama** can utilize an experimental wrapper aligning its API with OpenAI Functions, as demonstrated in a [notebook](https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/). For efficient addition of knowledge to chatbots, engineers advised using "add_documents" method of a vector database with **FAISS** for indexing without full reprocessing.

- **Asynchronous API Puzzles**: Members discussed how to handle concurrent requests to OpenAI's **ChatCompletion** endpoint, with the need for an asynchronous solution to notify multiple users simultaneously, differing from **GPT-4**'s batch requests.

- **Stepping Up Streaming**: To optimize response times with **Ollama**, users are advised to import `ChatOllama` and utilize its `.stream("query")` method, a trick recommended for speedier token-based outputs.

- **Memory for the Long Haul**: [Zep](https://www.getzep.com/), discussed as a potential solution for long-term memory in AI, integrates with LangChain to maintain persistent conversation summaries and retain critical facts effectively.

- **Flaunting AI Fitness and Business Savvy**: *Valkyri*e project amalgamates **NVIDIA, LangChain, LangGraph, and LangSmith tools** in an AI Personal Trainer, detailed on [GitHub](https://github.com/pannaf/valkyrie). A separate innovation spotlights a Python script to scrape Kentucky business leads on Instagram, complete with a [Google Sheet](https://docs.google.com/spreadsheets/d/1IYiaqHm_PmX5FdhZTIolxQhE3pWJ5T9X/edit?gid=87752022#gid=87752022) of data and a [YouTube tutorial](https://youtu.be/3xZlvR3aPQI) for Lambda integration in Visual Agents.

- **Framework Fit or Folly**: Decision-making for AI framework integration into apps was distilled in a [YouTube video](https://youtu.be/uG0cs8AlnHw), dissecting critical features of **GPT-4o, Gemini, Claude, and Mistral**, and the roles of setups like **LangChain** in development workflows.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Claude-3.5-Sonnet Buzz Fizzles Out**: Speculations about **Claude-3.5-Sonnet** diminish as insiders confirm a lack of privileged information regarding its development, pointing to publicly available details only.
  
- **Cohere Clamps Down on Rerank Model Stats**: Cohere maintains secrecy around the parameter size of its rerank models, leaving community members in the dark despite inquiries.

- **Global AI Minds, Gather**: **[Expedition Aya](https://sites.google.com/cohere.com/expedition-aya/home)** has been announced, a six-week event by Cohere aiming to foster worldwide collaborations in building multilingual AI models, complete with API credits and prizes for participants.

- **Preambles under the Microscope**: Cohere's **[Command R default preamble](https://docs.cohere.com/docs/preambles)** gains clarity through discussions and shared resources, revealing how it shapes model interactions and expectations.

- **Tune in for Cohere Dev Talk**: The *Cohere Developer Office Hours* encouraged eager devs to deep dive into the functionalities of Command R+, with a call to join the conversation at the following [Discord invitation link](https://discord.gg/CsqyQFhEXT?event=1248301309233336350).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad "LazyBuffer" Bug Spotted**: Users pinpointed a `'LazyBuffer' object has no attribute 'srcs'` error in the [tinygrad Tensor library](https://github.com/tinygrad/tinygrad/issues/5134); George Hotz acknowledged the bug in `lazy.py` and the need for thorough testing and a patch.

- **Clip() Workaround Proposed**: A workaround for the "LazyBuffer" bug involved substituting `.contiguous()` for `realize` during the usage of `.clip()` in tinygrad, a tweak that sidestepped the issue.

- **Docker for CI Debugging**: To address CI discrepancies on Macs, a member suggested using a [Linux environment via Docker](https://github.com/Qazalin/containers/blob/main/tinygrad/Dockerfile#L1C1-L14C1), which has a history of effectively solving similar issues.

- **Bounty Hunt for Qualcomm Drivers**: There's a $700 bounty for developing a Qualcomm GPU driver, details discussed referencing a certain [tweet](https://x.com/__tinygrad__/status/1805317200200581282?t=0bk72a1BFj_jqFAJgwqLjA&s=19), with suggestions to refer to `ops_amd.py` for guidance and use an Android phone with Termux and tinygrad for the setup.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Anticipation for Multimodal Models**: There's concern among members that the *llm3 multimodal* might be released before the 72 billion parameter model finishes training in mid-July, taking roughly 20 days with each epoch lasting 5 days.
  
- **Boosting Optimization with Adam-mini**: The [Adam-mini optimizer](https://arxiv.org/abs/2406.16793) paper on arXiv has caught members' attention for reducing memory usage by 45% to 50% when compared to AdamW, by decreasing the number of individual learning rates.

- **Custom LR Schedulers on HF Radars**: A user sought advice on creating a cosine learning rate (LR) scheduler using Hugging Face, keen on implementing a minimum LR greater than zero to fine-tune model training.

- **Accelerating Minhash with Python**: A member boasted a 12x performance enhancement in minhash calculations using Python, sparking interest and inviting collaborative feedback to further improve this optimization.

These were the highlights within the **OpenAccess AI Collective** that captured the guild's most significant discussions and technical interests.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Tokenizer Tussle on Torchtune**: A discrepancy between tokenizer configurations for **Phi-3-mini** and **Phi-3-medium** might affect Torchtune performance, with the former including a beginning-of-string token (`"add_bos_token": true`) and the latter not (`"add_bos_token": false`).
- **Troubleshooting TransformerDecoder**: Engineers run into a runtime size mismatch error in `TransformerDecoder` parameters, such as `attn.q_proj.weight`, signaling potential configuration or implementation issues with **Phi-3-Medium-4K-Instruct**.
- **Phi-3-Medium-4K-Instruct Compatibility Quagmire**: Ongoing errors suggest that **Phi-3-Medium-4K-Instruct** support within **Torchtune** is incomplete, needing additional tweaks for full compatibility.
- **Crafting a Custom Tokenizer Solution**: To resolve tokenizer discrepancies, members propose the creation of a dedicated `phi3_medium_tokenizer` by adapting the `phi3_mini_tokenizer` config and setting `add_bos = False`.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Beowulf's Big Speed Breakthrough**: A member announced a significant **speed improvement** for **beowulfbr's** efficiency tool, making it **12 times faster** than the datasketch.

- **Simon Says, 'Streamline Your Commands!'**: **Simon Willison** shared his talk on integrating **Large Language Models with command-line interfaces**, featuring a [YouTube video](https://www.youtube.com/watch?v=QUXQNi6jQ30) and an [annotated version of his presentation](https://simonwillison.net/tags/annotatedtalks/).

- **Innovative Dataset Generation Method Unveiled**: A new method for generating high-quality datasets for LLM instruction finetuning was highlighted. It is described as fully automated, requiring no seed questions and capable of running locally, with details shared in the linked [post](https://x.com/rasbt/status/1805217026161401984).

- **Synthetic Aperture Encoding with Linus Lee's Prism**: The guild discussed **Linus Lee's** work on **Prism** for finetuning, expressing interest in his approach to creating more interpretable models for humans, as detailed in his blog [post](https://thesephist.com/posts/prism/).

- **Private Model, Gradio Trouble**: A member encountered an error when attempting to create a **Gradio space** with a privately fine-tuned model via **AutoTune**, necessitating an `hf_token` due to the model's private status.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile v0.8.7 Goes Live**: The release of [llamafile v0.8.7](https://www.phoronix.com/news/Llamafile-0.8.7-Released) introduces *faster quant operations* and bug fixes, with hints at upcoming Android compatibility.
  
- **Get Set for July AI Talks and Tools**: Two key events, [Jan AI](https://discord.com/events/1089876418936180786/1251002752239407134) and [AutoFix by Sentry.io](https://discord.com/events/1089876418936180786/1245836053458190438), along with the [AI Foundry Podcast Roadshow](https://discord.com/events/1089876418936180786/1253834248574468249) are set to engage the community this month.
  
- **Mozilla AI Hits the Conference Circuit**: Members will present at the [World's Fair of AI](https://www.ai.engineer/worldsfair) and moderate at the [AI Quality Conference](https://www.aiqualityconference.com/) while Firefox Nightly paves new paths with optional AI services detailed in their [Nightly blog](https://discord.com/channels/1089876418936180786/1254858795998384239).

- **Read Up on the Latest ML Paper Picks**: The curated selection of recent machine learning research is now available, offering insights and discussions from the community.

- **Enhancing New User Experience for Llamafile**: Suggestions have been made to provide a step-by-step **llamafile and configuration guide** for novices, and discussions are ongoing about Firefox potentially integrating a built-in local inference feature for easier on-device inference.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Racy AI Enters the Beta Stage**: [Honeybot.ai](https://honeybot.ai/), an **AI-generated adult content platform**, announced the commencement of their beta phase, stating that the service is free for individuals over 18 years of age.

- **Project Activity Under Scrutiny**: A user raised concerns regarding the active status of a project, noting the prevalence of spam as an indicator that the project may no longer be active.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Bot Battlegrounds: Detecting Digital Deceivers**: An upcoming event titled *"Detecting Bots and Fraud in the Time of LLMs"* will unravel strategies to identify and mitigate the impact of LLM-based bots in automation and security. Set for June 27, 2023, the discussion will tackle the evolution of bots, as well as the current detection methodologies utilized by experts.

- **Meet the AI Sentinel – Unmesh Kurup**: With the prevalence of sophisticated LLMs, **Unmesh Kurup**, leading the ML team at Intuition Machines/hCaptcha, will be the keynote speaker at the digital event, breaking down advanced security systems to discern between bots and human interaction. Engineers and specialists in the field can [register for free](https://lu.ma/y6hz8cod?utm_source=diskord) to gain insights from Kurup’s extensive experience in AI/ML.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Datasette - LLM (@SimonW) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1255261278994305024)** (1 messages): 

```html
- **Argilla 2.0 boosts dataset annotation**: [Argilla 2.0](https://x.com/argilla_io/status/1805250218184560772) announced with new Python SDK for dataset integration and a flexible UI for data annotation. The update promises to "create high-quality datasets more efficiently."
- **Microsoft's Florence models crush benchmarks**: Microsoft released [Florence](https://x.com/osanseviero/status/1803324863492350208), a vision model for tasks like captioning and OCR with models sized 200M and 800M, MIT-licensed. "*Fine-tune Florence-2 on any task*" with a new [notebook and walkthrough](https://x.com/mervenoyann/status/1805265940134654424) on DocVQA dataset.
- **Generate GGUF quants in seconds**: New [support added](https://x.com/reach_vb/status/1804615756568748537) for "Generate GGUF quants in less than 120 seconds" including automatic uploads to the hub and support for private and org repos. Over 3500 model checkpoints created.
- **Embedding models guide for AWS**: A comprehensive guide on how to [train and deploy embedding models](https://www.philschmid.de/sagemaker-train-deploy-embedding-models) on AWS SageMaker using Sentence Transformers and fine-tuning the BGE model for financial data. Training takes ~10 minutes on a ml.g5.xlarge instance at around $0.2.
- **Ethics and Society newsletter on data quality**: The latest [Ethics and Society newsletter](https://huggingface.co/blog/ethics-soc-6) highlights the importance of data quality. Collaboration with the ethics regulars led to a detailed discussion on this crucial theme.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/argilla_io/status/1805250218184560772)">Tweet from Argilla (@argilla_io)</a>: 📢 Another big announcement: Argilla 2.0 rc!  What does it mean for AI builders?  🤺 Unified framework for feedback collection  🐍 New Python SDK to work with datasets, including a new @huggingface da...</li><li><a href="https://x.com/osanseviero/status/1803324863492350208)">Tweet from Omar Sanseviero (@osanseviero)</a>: Microsoft just silently dropped Florence  👀Vision model that can tackle many vision tasks (captioning, detection, region proposal, OCR) 🤏Small models (200M and 800M) with ~quality to models 100x lar...</li><li><a href="https://x.com/mervenoyann/status/1805265940134654424)">Tweet from merve (@mervenoyann)</a>: Fine-tune Florence-2 on any task 🔥  Today we release a notebook and a walkthrough blog on fine-tuning Florence-2 on DocVQA dataset @andi_marafioti @skalskip92   Keep reading ⇓</li><li><a href="https://x.com/reach_vb/status/1804615756568748537)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Generate GGUF quants in less than 120 seconds! ⚡  &gt; Added support for imatrix quants &gt; GGUF-split support for larger quants &gt; Automatic upload to hub &gt; Support for private and org repos  U...</li><li><a href="https://x.com/osanseviero/status/1804136001465442530)">Tweet from Omar Sanseviero (@osanseviero)</a>: Microsoft just silently (again!) dropped Instruction Pre-Training!  👀Augment pretraining datasets generating instructions 🦙A Llama 3 8B with comparable performance to 70B! 🔥General+domain models (m...</li><li><a href="https://x.com/vanstriendaniel/status/1804078257488495099)">Tweet from Daniel van Strien (@vanstriendaniel)</a>: Instruction pre-training is a new approach that enhances LLM pretraining by using instruction-response pairs from an instruction synthesizer instead of raw data.  Explore this method in this @gradio S...</li><li><a href="https://x.com/danieldekok/status/1804224598721830954)">Tweet from Daniël de Kok (@danieldekok)</a>: 🐬More Marlin features coming to the next @huggingface TGI release: support for using existing GPTQ-quantized models with the fast Marlin matrix multiplication kernel.  ⚡This feature is made possible ...</li><li><a href="https://x.com/eustachelb/status/1805262952913858919)">Tweet from Eustache Le Bihan (@eustachelb)</a>: Distil-Whisper goes multilingual!! 🤗  The French distilled version of Whisper is here! 🇫🇷 As accurate as large-v3, faster than tiny. The best of both worlds! 🚀  Check out the details below ⬇️</li><li><a href="https://x.com/_philschmid/status/1805593591223398832)">Tweet from Philipp Schmid (@_philschmid)</a>: Embedding models are crucial for successful RAG applications, but they&#39;re often trained on general knowledge! Excited to share an end-to-end guide on how to Train and Deploy open Embeddings models...</li><li><a href="https://x.com/FrG_FM/status/1803703761119871122)">Tweet from F-G Fernandez (@FrG_FM)</a>: Xavier & @osanseviero presenting the robotics initiatives of @huggingface 🤗 (including LeRobot led by none other than @RemiCadene) at #AIDev by @linuxfoundation  Looking forward to the day when we re...</li><li><a href="https://x.com/RisingSayak/status/1805521415543697582)">Tweet from Sayak Paul (@RisingSayak)</a>: Were you aware that we have a dedicated guide on different prompting mechanisms to improve the image generation quality? 🧨  Takes you through simple prompt engineering, prompt weighting, prompt enhan...</li><li><a href="https://x.com/evijitghosh/status/1805312283628761446)">Tweet from Avijit Ghosh (@evijitghosh)</a>: The quarterly @huggingface Ethics and Society newsletter is out! Had so much fun collabing on this with @frimelle and supported by the ethics regulars. The theme for this quarter&#39;s newsletter is t...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1254876632255172749)** (436 messages🔥🔥🔥): 

- **Fun with Floating-Point**: Users debated the practicalities of float vs integer types in Python, leading to various code iterations to handle large float computations (e.g., `pi**pi**pi`). One user pointed out a common issue: *"OverflowError: (34, 'Result too large')*" when using `math.pow`.

- **RAM Troubleshooting for AI Models**: A user struggled to load datasets without running out of memory, despite having 128GB of RAM. Solutions proposed included using `save_to_disk`, `load_from_disk`, and enabling `streaming=True`.

- **Discussions on Quantization**: Members explained quantization as a method to run large AI models on lower-end hardware. It reduces the precision of the model's parameters, which may affect performance but allows models to operate within memory constraints.

- **Git Usage Concerns**: Users discussed inefficiencies and errors related to using `graphviz` on Hugging Face spaces, troubleshooting an error about missing executables and suggesting potential fixes. One helpful solution involved confirming whether graphviz was correctly added to the system's `PATH`.

- **Career and Learning Path Advice**: Users discussed which tech skills were most employable, debating fields like cybersecurity vs data science. Advice was given: *"more than any particular field, getting involved in real projects can be hugely helpful."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>: no description found</li><li><a href="https://llava-vl.github.io/">LLaVA</a>: no description found</li><li><a href="https://huggingface.co/Azazelle/L3-RP_io/tree/main">Azazelle/L3-RP_io at main</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=eId6K8d0v6o">How to Install WSL2 on Windows 11 (Windows Subsystem for Linux)</a>: In this video tutorial, we&#39;ll show you how to install WSL2 on Windows 11, allowing you to run Linux commands and tools directly within the Windows environmen...</li><li><a href="https://huggingface.co/Azaz">azaz (Z)</a>: no description found</li><li><a href="https://huggingface.co/posts/nroggendorff/357091156426242">@nroggendorff on Hugging Face: &quot;@osanseviero your move&quot;</a>: no description found</li><li><a href="https://tenor.com/view/huh-cat-cat-huh-small-cat-huh-what-gif-2593177363967991691">Huh Cat GIF - Huh Cat Cat huh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/pluh-veilbound-gif-4315180366045476816">Pluh Veilbound GIF - Pluh Veilbound - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/wat-gif-5001440307614895833">Wat GIF - Wat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=-4Oso9-9KTQ">ChatGPT Explained Completely.</a>: For 16 free meals with HelloFresh PLUS free shipping, use code KYLEHILL16 at https://bit.ly/41QHRYfChatGPT is now the fastest-growing consumer app in human h...</li><li><a href="https://tenor.com/view/white-cat-eating-salad-meme-blonde-woman-crying-gif-26971383">White Cat Eating Salad GIF - White Cat Eating Salad Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://untitled.stream/library/track/JcLqx6Kwgwn39scJDdMuX">SHITLORD (POOPMASTER DISS)</a>: mikusss&#x27;s track on [untitled]</li><li><a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch#the-big-table-of-tasks">transformers/examples/pytorch at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://tenor.com/view/old-man-frustrated-dog-gif-15305478">Old Man Frustrated GIF - Old Man Frustrated Dog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://untitled.stream/library/track/w5TYpULbkmTPsnWPLQ5WH">Altman</a>: mikusss&#x27;s track on [untitled]</li><li><a href="https://github.com/beowolx/rensa">GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets</a>: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets - beowolx/rensa</li><li><a href="https://www.pythonanywhere.com/">Host, run, and code Python in the cloud: PythonAnywhere</a>: no description found</li><li><a href="https://wiki.civitai.com/wiki/Civitai_API#GET_/api/v1/models-versions/by-hash/:hash">Civitai API - Civitai Wiki</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1255059213109956760)** (2 messages): 

```html
- **Challenges with Langchain Pydantic and LLM**: A member is trying to use **Langchain Pydantic Basemodel** to structure document data into JSON with additional insights. They are facing issues as the LLM misinterprets the data due to tabular structures and seek evaluation strategies or better methods.

- **Expression of Interest in the Topic**: Another member indicated their interest in the topic by stating, "I am interested ...".
```
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1254913112554078309)** (3 messages): 

- **Attend the bot and fraud detection event**: An upcoming event on "Detecting Bots and Fraud in the Time of LLMs" is scheduled for June 27, 2023, at 10 a.m. PDT. The keynote speaker Unmesh Kurup, Director of ML at Intuition Machines/hCaptcha, will share insights into advanced detection strategies ([Register here](https://lu.ma/y6hz8cod?utm_source=diskord)).

- **Check out T2V-Turbo on HuggingFace**: A member shared a link to [T2V-Turbo](https://huggingface.co/spaces/TIGER-Lab/T2V-Turbo) on HuggingFace's Spaces. They noted it offers a refreshing and impressive experience.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/TIGER-Lab/T2V-Turbo">T2V Turbo - a Hugging Face Space by TIGER-Lab</a>: no description found</li><li><a href="https://lu.ma/y6hz8cod?utm_source=diskord">A Million Turing Tests per Second: Detecting bots and fraud in the time of LLMs · Luma</a>: The Data Phoenix team invites you to our upcoming webinar, which will take place on June 27th at 10 a.m. PDT. Topic: A Million Turing Tests per Second:…
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1255017509107535893)** (159 messages🔥🔥): 

- **Tokenization isn't practical, says [Apehex](https://huggingface.co/blog/apehex/tokenization-is-a-dead-weight)**: Apehex argues in their article that tokenization methods are ineffective and suggests using neural networks to encode sequences of Unicode characters directly. A detailed discussion ensued, covering technical aspects like embedding, model size, and potential issues with floating-point accuracy.
- **Personalize city maps with [Cityscape Prettifier](https://github.com/C1N-S4/cityscape-prettifier)**: Deuz_ai_80619 shared a GitHub project that allows users to create beautiful, personalized city maps using Flask, Prettymaps, and Python, turning OpenStreetMap data into stylish visualizations.
- **Startpage extension for media lovers [Starty Party](https://startyparty.dev/)**: Desmosthenes introduced a new start page extension for browsers focused on media and content, available for installation at [marketing.startyparty.dev](https://marketing.startyparty.dev).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/apehex/tokenization-is-a-dead-weight">Is Tokenization Necessary?</a>: no description found</li><li><a href="https://github.com/C1N-S4/cityscape-prettifier">GitHub - C1N-S4/cityscape-prettifier: Make beautiful, personalized city maps with ease. With Flask, Prettymaps, and Python, turn OpenStreetMap data into elegant, stylish visualizations. Ideal for designers, developers, and urban aficionados.</a>: Make beautiful, personalized city maps with ease. With Flask, Prettymaps, and Python, turn OpenStreetMap data into elegant, stylish visualizations. Ideal for designers, developers, and urban aficio...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1255094329580585073)** (4 messages): 

- **Looking for papers on contamination**: A member requested recommendations for papers on contamination, particularly for coding benchmarks. They shared three relevant papers from their reading list: [annotate tests by month](https://arxiv.org/abs/2403.07974), [general saturation in ML benchmarks](https://www.semanticscholar.org/paper/Mapping-global-dynamics-of-benchmark-creation-and-Ott-Barbosa-Silva/43ae69101c302628b9f7186ec5f35f99bb89d5d6), and [robustness in coding benchmarks](https://arxiv.org/abs/2212.10264).

- **Exploring Hilbert curve for 2D to 1D conversion**: A member inquired about using the Hilbert curve for scanning 2D images into 1D. They noted its advantage of not having jumps and working well for square images of different sizes.

- **Concerns on unordered path information loss**: Another member cautioned that the suggested method of using the Hilbert curve could result in information loss. They deemed the unordered path as not reasonable and mentioned a follow-up question on a different platform.

- **Paper update and code release**: A member announced they are preparing to update their paper and will release the code in the upcoming days. This indicates progress towards sharing their research findings publicly.
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1255204935075758101)** (3 messages): 

- **`hf-vision/detection_metrics` struggles with dependencies**: A member encountered an error when trying to use `hf-vision/detection_metrics` in `evaluate`, prompted by an ImportError due to missing dependencies. They noted that no such package exists, or they might be missing something.

- **Detection metrics feature flagged as problematic**: The same member pointed out that the issue with `hf-vision/detection_metrics` is documented in the Hugging Face GitHub issues, specifically in [this issue comment](https://github.com/huggingface/evaluate/issues/111#issuecomment-1855872972).

- **`evaluate` fails to locate `detection_util`**: It was discovered that `evaluate` could not find `detection_util` because it is located inside a folder within the space, which causes the tool to malfunction.

**Link mentioned**: <a href="https://github.com/huggingface/evaluate/issues/111#issuecomment-1855872972.">Add COCO evaluation metrics · Issue #111 · huggingface/evaluate</a>: I&#39;m currently working on adding Facebook AI&#39;s DETR model (end-to-end object detection with Transformers) to HuggingFace Transformers. The model is working fine, but regarding evaluation, I&#39...

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1255054037364244551)** (3 messages): 

- **Query on LLMs for Tabular Data Interaction**: A member asked if there are open-source **LLM projects** or products that specialize in inference on tabular data, specifically stored as CSV or pandas DataFrames. They are interested in interacting with a chat bot to ask questions about trends in the data without needing modeling or prediction.

- **Interest in Contributing to a GitHub PR**: A member expressed interest in helping with a [PR related to RoBERTa-based models](https://github.com/huggingface/transformers/pull/30510), focusing on adding support for Scaled Dot Product Attention (SDPA). They faced an issue due to lack of access to the original repository and sought advice on how to contribute.


**Link mentioned**: <a href="https://github.com/huggingface/transformers/pull/30510">[RoBERTa-based] Add support for sdpa by hackyon · Pull Request #30510 · huggingface/transformers</a>: What does this PR do? Adding support for SDPA (scaled dot product attention) for RoBERTa-based models. More context in #28005 and #28802. Before submitting   This PR fixes a typo or improves the do...

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1255207180878221352)** (1 messages): 

- **Gradio v4.37 is here**: The latest release, Gradio v4.37, features a **redesigned chatbot UI**, dynamic plots and GIF support, and significant performance upgrades. It also boasts improved customizability and numerous bug fixes for a smoother user experience.
- **Exciting new features announced**: The **new chatbot UI** supports embedding components like gr.Gallery and gr.Audio directly in the chat, while gr.Image now supports GIFs. Check out the full details in [Gradio's changelog](https://www.gradio.app/changelog).
  

---



### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1255191730324766882)** (3 messages): 

- **Need to align PyTorch tensor in memory**: A member asked if *"there is any way to enforce that a PyTorch tensor is memory-aligned to an amount of bytes"* for loading tensors in pairs using `float2`. They are encountering issues with alignment.
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1254916229521608704)** (11 messages🔥): 

- **Function dequantization clarified with source code link**: Users discussed the function `torch.ops.aten._weight_int4pack_mm`, with a helpful link to the [source code on GitHub](https://github.com/pytorch/pytorch/pull/110914). This function performs dequantization and matrix multiplication with an identity matrix.
- **Docs for autogen function are not helpful**: Users pointed out that the autogenerated documentation for the function was essentially blank and not informative at all (*"documentation is blank 🤣"*).
- **8-bit Adam collaboration thread**: A thread was initiated for collaboration on 8-bit Adam optimizations. Key questions included the use of dynamic quantization schemes and whether dequantization and adam-step operations are fused together in a single kernel.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html">Function at::_weight_int4pack_mm &mdash; PyTorch main documentation</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/pull/110914">Quant: add weight int4pack mm kernel by yanboliang · Pull Request #110914 · pytorch/pytorch</a>: Adding the weight int4pack mm CUDA kernel. The kernel comes from the tinnygemm project which developed by Jeff Johnson.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1255179530121904321)** (3 messages): 

- **Cute Demake Quake Impresses**: Members shared a [YouTube video titled "Q1K3 – Making Of"](https://www.youtube.com/watch?v=F9MLpuvPDio), featuring a tribute to Quake created in **13kb of JavaScript** for the js13kGames contest 2021. They also provided a [link to play the game](https://phoboslab.org/q1k3/) and a [blog post](https://phoboslab.org/log/202).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=F9MLpuvPDio]">Q1K3  – Making Of</a>: A tribute to Quake in 13kb of JavaScript, made for the js13kGames contest 2021.Play here: https://phoboslab.org/q1k3/Blog Post: https://phoboslab.org/log/202...</li><li><a href="https://www.youtube.com/watch?v=F9MLpuvPDio">Q1K3  – Making Of</a>: A tribute to Quake in 13kb of JavaScript, made for the js13kGames contest 2021.Play here: https://phoboslab.org/q1k3/Blog Post: https://phoboslab.org/log/202...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1255013900718702592)** (4 messages): 

- **HFGenerator broken after cache logic change**: A member reported an issue with `HFGenerator`, noting that while the native `model.generate(input_ids)` function works well, the former has been problematic since transformers updated the cache logic.
- **mobicham confirms need for rewrite**: A member acknowledged this issue, stating, *"I need to rewrite, will do it this week,"* and mentioned potential problems with `model.generate` when using `torch.compile`, particularly the recompile behavior due to changing prompt lengths.
- **Verifying output without compiling**: Discussions included verifying outputs without using `torch.compile`, suggesting that the focus was primarily on functional correctness rather than performance optimization.
- **Model-specific issues and alternatives**: The conversation shifted to model-specific concerns, mentioning potential issues with Llama2-7B and comparing it with Llama3, where `axis=1` settings and various configurations were provided for reference: *"Llama3-8b-instruct GPTQ (gs=64): 66.85, AWQ (gs=64): 67.29, HQQ (axis=1, gs=64): 67.4."*
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1254873802899460218)** (402 messages🔥🔥): 

- **Windows build cuDNN fix merged**: After some troubleshooting, a fix for windows cuDNN build breakage was merged ([#639](https://github.com/karpathy/llm.c/pull/639)). Issues were tied to macro redefinitions and needed adjustments like adding `WIN32_LEAN_AND_MEAN`.
  
- **Exploring H100 training instability with cuDNN**: Training on H100 with cuDNN showed instability, particularly in bf16 training, which did not occur when cuDNN was turned off. Investigations point towards possible differences in tile sizes for cuDNN flash attention.
  
- **AMD GPU support in focus**: Members discussed incorporating support for AMD GPUs. An AMD fork of the repository, [anthonix/llm.c](https://github.com/anthonix/llm.c), is currently maintained, but wider interest in AMD GPUs is still developing.
  
- **PR for on-device reductions is under review**: There is a pull request aimed at reducing GPU ↔ CPU transfers by moving more calculations on-device ([#635](https://github.com/karpathy/llm.c/pull/635)). It includes micro-optimizations such as avoiding recalculations in validation steps.
  
- **Discussion on Llama 3 support and v1.0 roadmap**: Planning toward a v1.0 release, focusing on treating GPT-2/3 support separately and introducing Llama 3 in follow-up versions. **PRs for rolling checkpoints** and **StableAdamW optimizer** are key components ([#636](https://github.com/karpathy/llm.c/pull/636)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpath">karpath - Overview</a>: GitHub is where karpath builds software.</li><li><a href="https://github.com/karpathy/llm.c/pull/640">Add missing MULTI_GPU compiler flag by chinthysl · Pull Request #640 · karpathy/llm.c</a>: To run the file sharing NO_USE_MPI build.</li><li><a href="https://github.com/anthonix/llm.c">GitHub - anthonix/llm.c: LLM training in simple, raw C/HIP for AMD GPUs</a>: LLM training in simple, raw C/HIP for AMD GPUs. Contribute to anthonix/llm.c development by creating an account on GitHub.</li><li><a href="https://x.com/Etched/status/1805625693113663834">Tweet from Etched (@Etched)</a>: Meet Sohu, the fastest AI chip of all time.  With over 500,000 tokens per second running Llama 70B, Sohu lets you build products that are impossible on GPUs. One 8xSohu server replaces 160 H100s.  Soh...</li><li><a href="https://github.com/karpathy/llm.c/pull/636">rolling checkpoints by karpathy · Pull Request #636 · karpathy/llm.c</a>: checkpoints are either MINOR or MAJOR and minor checkpoints get deleted with a rolling window. This is an optimization that will allow us to save state more often, but preserve disk space overall. ...</li><li><a href="https://github.com/karpathy/llm.c/pull/633">Socket server/client interface by chinthysl · Pull Request #633 · karpathy/llm.c</a>: Dummy PR to make use of the distributed interface in PR #632</li><li><a href="https://github.com/karpathy/llm.c/pull/637">add outlier detector, test for it, and start tracking z score of loss by karpathy · Pull Request #637 · karpathy/llm.c</a>: still TODO:  track grad norm in addition to loss (do we have to split the gpt2_update function) add argparse option to monitor loss and grad norm for outliers (e.g. z &gt; 3) add handling of instabili...</li><li><a href="https://github.com/karpathy/llm.c/pull/635">On-device reductions by ngc92 · Pull Request #635 · karpathy/llm.c</a>: Moves loss calculation to backward, and ensures  we can do more on-device reductions and fewer host&lt;-&gt;device transfers. Also enables a micro-optimization, that validate does not calculate dlogit...</li><li><a href="https://github.com/warner-benjamin/optimi/blob/4542d04a3974bb3ac9baa97f4e417bda0432ad58/optimi/stableadamw.py#L28>).">optimi/optimi/stableadamw.py at 4542d04a3974bb3ac9baa97f4e417bda0432ad58 · warner-benjamin/optimi</a>: Fast, Modern, Memory Efficient, and Low Precision PyTorch Optimizers - warner-benjamin/optimi</li><li><a href="https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/">Scaling Language Model Training to a Trillion Parameters Using Megatron | NVIDIA Technical Blog</a>: Natural Language Processing (NLP) has seen rapid progress in recent years as computation at scale has become more available and datasets have become larger. At the same time, recent work has shown&#82...</li><li><a href="https://community.flexera.com/t5/FlexNet-Publisher-Forum/FlexNet-11-19-5-build-on-Visual-Studio-2015/m-p/306967">FlexNet 11.19.5 build on Visual Studio 2015</a>: Hi all,   I am trying to build my app with FlexNet 11.19.5. I am facing some compiler issues (Visual Studio 2015):c:\program files (x86)\windows kits\8.1\include\shared\ws2def.h(100): warning C4005: '...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 messages): 

iron_bound: https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1254909421511770227)** (1 messages): 

- **Debugging PyTorch tensor device call**: A member discussed a potential fix for a device call issue in PyTorch by referring to a specific [line in native_functions.yaml](https://github.com/pytorch/pytorch/blob/18fdc0ae5b9e9e63eafe0b10ab3fc95c1560ae5c/aten/src/ATen/native/native_functions.yaml#L7680). They suggested trying `BitnetTensor(intermediate).to(device=tensor.device)` instead of the original code.

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/blob/18fdc0ae5b9e9e63eafe0b10ab3fc95c1560ae5c/aten/src/ATen/native/native_functions.yaml#L7680,">pytorch/aten/src/ATen/native/native_functions.yaml at 18fdc0ae5b9e9e63eafe0b10ab3fc95c1560ae5c · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1254877097969582150)** (131 messages🔥🔥): 

- **Replete-Coder Llama3-8B model debuts**: The new [Replete-Coder Llama3-8B model](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) was discussed heavily, being highlighted for its advanced coding capabilities in over 100 languages, and its uncensored, fully deduplicated training data. It is supported by TensorDock for cloud compute rental.
- **Skepticism about benchmarks**: Users discussed the reliability of benchmarks, emphasizing that they can be overfitted and do not always represent real-world performance. One member stated, “...benchmarks tell you very little… it needs eyes on for a bigger problem scope.”
- **Adam-mini optimizer**: The [new Adam-mini optimizer](https://arxiv.org/abs/2406.16793) claimed to achieve on-par or better performance than AdamW with 45% to 50% less memory and 49.6% higher throughput, which sparked discussions about potential implementation and comparative benefits to existing optimizers.
- **Probllama vulnerability in Ollama project**: A critical remote code execution vulnerability, [CVE-2024-37032](https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032), was discussed. The issue was fixed swiftly, and users stressed the importance of upgrading to the latest version for security.
- **General chat about model training and fine-tuning**: Members shared insights on optimizing memory and throughput during model training, discussing the use of different context lengths and expressing interest in fine-tuning models like Yi-1.5-34B while considering constraints like GPU capacities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the number of learning rates in...</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>: no description found</li><li><a href="https://x.com/faridofanani96/status/1804079517193113850">Tweet from Mochamad Farido Fanani (@faridofanani96)</a>: The Pangu Model 5.0 is here, first divided into four series:  Pangu E series: 1-9 billion parameter level, applied to mobile phones and PCs Pangu P series: 10 billion -90 billion parameters, suitable ...</li><li><a href="https://huggingface.co/google-bert/bert-base-uncased">google-bert/bert-base-uncased · Hugging Face</a>: no description found</li><li><a href="https://x.com/_clashluke/status/1805522937744597316">Tweet from Lucas Nestler (@_clashluke)</a>: @arankomatsuzaki Adam-Mini (https://arxiv.org/abs/2406.16793, left) is identical to NVIDIA&#39;s NovoGrad (https://arxiv.org/pdf/1905.11286, right) up to a constant factor. The major difference is tha...</li><li><a href="https://x.com/danielhanchen/status/1805608733671833974">Tweet from Daniel Han (@danielhanchen)</a>: I&#39;ll be at The Marriott Marquis, SF 9AM PST today for our @aiDotEngineer World&#39;s Fair workshop! Head over to Golden Gate Ballroom B.  Excited to chat and meet in person if you&#39;re in SF! I&...</li><li><a href="https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032">Probllama: Ollama Remote Code Execution Vulnerability (CVE-2024-37032) – Overview and Mitigations | Wiz Blog</a>: Wiz Research discovered CVE-2024-37032, an easy-to-exploit Remote Code Execution vulnerability in the open-source AI Infrastructure project Ollama. </li><li><a href="https://ai.meta.com/research/cicero/diplomacy/">no title found</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Replete-Coder-Llama3-8B-exl2">bartowski/Replete-Coder-Llama3-8B-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Replete-Coder-Llama3-8B-GGUF">bartowski/Replete-Coder-Llama3-8B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1254887346701336636)** (251 messages🔥🔥): 

```html
- **Checkpoints and Finetuning**: "Use save checkpoints and continue finetuning from the checkpoints" is suggested, with a link to the [Unsloth wiki](https://github.com/unslothai/unsloth/wiki) for more detailed instructions.
- **Multi GPU Issues**: Users reported runtime errors when trying to run Unsloth on multi-GPU setups and discussed potential workarounds, including limiting CUDA devices and downgrading to a previous Unsloth version. A relevant link to [GitHub issue 660](https://github.com/unslothai/unsloth/issues/660) was shared.
- **Vision Models and OCR**: GPT4o's performance in OCR was discussed, with some users skeptical about LLAVA models achieving similar results. An alternative suggestion was [openedai-vision](https://github.com/matatonic/openedai-vision).
- **Experimentation with LLaMA Models**: Users shared difficulties and potential solutions when fine-tuning "unsloth/Phi-3-mini-4k-instruct" and other issues with datasets and training setups. A workaround involving model merging for better results on Hugging Face was suggested.
- **Training Statistics and Callbacks**: Discussion on how to track loss and other metrics during training, with recommendations for using wandb, TensorBoard, and custom callbacks in Hugging Face. A link to [TensorBoardCallback documentation](https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.TensorBoardCallback) was provided.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ibm/MoLFormer-XL-both-10pct">ibm/MoLFormer-XL-both-10pct · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on">How do I select which GPU to run a job on?</a>: In a multi-GPU computer, how do I designate which GPU a CUDA job should run on?  &#xA;&#xA;As an example, when installing CUDA, I opted to install the NVIDIA_CUDA-&amp;lt;#.#&amp;gt;_Samples then ran ...</li><li><a href="https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.integrations.TensorBoardCallback">Callbacks</a>: no description found</li><li><a href="https://github.com/matatonic/openedai-vision">GitHub - matatonic/openedai-vision: An OpenAI API compatible API for chat with image input and questions about the images. aka Multimodal.</a>: An OpenAI API compatible API for chat with image input and questions about the images. aka Multimodal. - matatonic/openedai-vision</li><li><a href="https://github.com/unslothai/unsloth/issues/660">CUDA_VISIBILE_DEVICES not functioning · Issue #660 · unslothai/unsloth</a>: I saw error message when I am trying to do supervised fine tuning with 4xA100 GPUs. So the free version cannot be used on multiple GPUs? RuntimeError: Error: More than 1 GPUs have a lot of VRAM usa...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1254874126611513435)** (182 messages🔥🔥): 

```html
<ul>
    <li><strong>Language Switching Bug Annoys Users</strong>: Multiple users reported a bug where the UI language on Perplexity randomly changed to languages other than English, despite settings indicating English. One user noted, "It says English, but it's in Spanish."</li>
    <li><strong>Pro Search Features Confuse Users</strong>: Users asked about differences between Pro Search and standard search, expressing confusion over features supposedly available to standard users as well. Another user wished for clarity, noting that new multi-step processes felt slower.</li>
    <li><strong>File Download Issues Plague PRO Users</strong>: A user reported problems with generating accessible download links for uploaded files, despite having a PRO subscription. The response indicated the lack of a "code interpreter" in Perplexity.</li>
    <li><strong>Perplexity Pro Functionality in Question</strong>: Users from Brazil faced issues with Perplexity fetching searches from localized sources, instead returning results primarily in English. One user from Argentina questioned if subscribing to the Pro plan would unlock the “Pages” feature.</li>
    <li><strong>API Summarization Fails to Impress</strong>: A user working with the Perplexity API noted that it failed to return citations and images. Another advised to ask Perplexity to create code blocks as a workaround for document generation.</li>
</ul>
```

**Link mentioned**: <a href="https://jina.ai/news/jina-reranker-v2-for-agentic-rag-ultra-fast-multilingual-function-calling-and-code-search/">Jina Reranker v2 for Agentic RAG: Ultra-Fast, Multilingual, Function-Calling &amp; Code Search</a>: Jina Reranker v2 is the best-in-class reranker built for Agentic RAG. It features function-calling support, multilingual retrieval for over 100 languages, code search capabilities, and offers a 6x spe...

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1254897118095736956)** (8 messages🔥): 

<ul>
<li><strong>Starliner Faces Crisis</strong>: Perplexity AI highlights <a href="https://www.youtube.com/embed/xUsxGDrwzls">a YouTube video</a> covering a range of topics, including issues with the Starliner spacecraft, Apple AI delays in Europe, and a significant acquisition by OpenAI.</li>
<li><strong>Panthers Win</strong>: A user shared a link to Perplexity AI showcasing the recent victory of the Panthers, promising more details about the event. <a href="https://www.perplexity.ai/search/Panthers-win-XXbt_oX7S_SW5q43L9XrNA">Read more</a>.</li>
<li><strong>Australia's New Governor General</strong>: The announcement states that Australia is set to appoint a new climate and gender advocate, Samantha Mostyn, as the Governor General. Get the full story <a href="https://www.perplexity.ai/page/Ms-Samantha-Mostyns-lVVHDtSRQeWmfvxzFMRk.Q">here</a>.</li> 
<li><strong>If I'm a...</strong>: A user poses an intriguing search question on Perplexity AI titled, <a href="https://www.perplexity.ai/search/If-Im-a-UwBIlYUgT6ms6xsVqQ.OGA">"If I'm a..."</a> encouraging others to explore the quest for self-identity.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/peoJ7ftVVqY">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/embed/xUsxGDrwzls">YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1255173219921694802)** (2 messages): 

- **Pro Features are necessary**: A member expressed that it's "unfortunate" because they need the "Pro features" for their work. The emoji 😦 highlights their disappointment.
- **Get help from f1shy**: Another member suggested contacting "f1shy" to resolve the issue. Their tone indicated that f1shy could provide the needed assistance.
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1254879432301350955)** (75 messages🔥🔥): 

- **RTX 3090 struggles with models**: A user reported disappointment with their **RTX 3090** eGPU setup, unable to load larger models like **Command R (34b) Q4_K_S** with high token contexts. Recommended to explore models in the **exl2 format** for better VRAM usage.
  
- **exl2 format remains limited**: A user noted the **small size and limited GUI options** for the exl2 format on GitHub. Suggested tools like **tabbyAPI and open-webui** were recommended for better performance.

- **Confusion over Llama 3 model labels**: Users discussed the differences between **Llama 3 8B text, Llama 8B Instruct**, and an unlabeled **Llama 3 8B**. Clarified that the unlabeled variant is the base model, not finetuned for specific tasks.

- **Support for AMD and Intel GPUs in LM Studio**: A user inquired about **Intel and AMD GPU support** for LM Studio; currently supported through **OpenCL** but lacks **RoCM and Vulkan** support. A shared link to [configuration instructions](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md) helped resolve some issues.

- **Inquiries about vision models**: Another user asked for **image generation capabilities** in ML Studio. Currently, **LM Studio** does not support image generation, with recommendations to use external tools like [Fooocus](https://github.com/lllyasviel/Fooocus) for those features.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">Vision Models (GGUF) - a lmstudio-ai Collection</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://github.com/lllyasviel/Fooocus">GitHub - lllyasviel/Fooocus: Focus on prompting and generating</a>: Focus on prompting and generating. Contribute to lllyasviel/Fooocus development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1254877779569148016)** (26 messages🔥): 

- **Hathor_Fractionate-L3-8B-v.05 praised for performance**: A user shared their positive experience with using **Hathor_Fractionate-L3-8B-v.05** on Hugging Face, highlighting its capabilities in creative writing and educational support. They emphasized the benefit of leaving output tensors and embeddings in F32 for improved writing quality.

- **Replete-Coder-Llama3-8B excels in coding tasks**: The **Replete-Coder-Llama3-8B** model is noted for its proficiency in over 100 coding languages and incorporation of security, vulnerability prevention, and advanced math abilities. It's trained with a substantial amount of uncensored coding instruction data, making it suitable for both general and specialized coding applications.

- **New Dawn 70b impresses in roleplay**: Members discussed their satisfaction with **New Dawn 70b** for complex role-playing scenarios. It demonstrated the ability to handle up to 32k context creatively before performance declined.

- **DeepSeek Coder V2 resource requirements**: A conversation around **DeepSeek Coder V2** highlighted its need for significant VRAM, specifically recommending 24GB for the lite version, and mentioned the optimal setup involves a combination of system RAM and GPU's VRAM for larger models.

- **Fantasy storywriting model recommendations**: For fantasy storytelling and roleplay, users were directed to try specific models including **bartowski/aya-23-8B-GGUF** and other recommendations on Discord. The importance of experimenting with different models to find the best fit for specific needs was emphasized.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/friends-ross-geller-david-schwimmer-drink-a-lot-gif-15947149">Friends Ross Geller GIF - Friends Ross Geller David Schwimmer - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/Nitral-AI/Hathor_Fractionate-L3-8B-v.05">Nitral-AI/Hathor_Fractionate-L3-8B-v.05 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Hathor_Stable-L3-8B-v0.5-GGUF">bartowski/Hathor_Stable-L3-8B-v0.5-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1254882262554509423)** (3 messages): 

- **LM Studio Network Error on Ubuntu 22.04**: A user reported a **"network error"** when trying to search models on HuggingFace using **LM Studio** on Ubuntu 22.04, although it still worked on Mac M1. They noted that commenting out the ser2net config file for port 3001 was the only change they made, which was used by the AnythingLLM web server.

- **Prompt to Add to Feature Requests**: Another user asked if the network error issue had already been added to **feature requests**, suggesting it might be a relevant addition.

- **IPv6 Potential Solution**: An IT expert suggested disabling **IPv6** on the affected Ubuntu box to potentially resolve the network error issue. They humorously noted that many problems are reported with the phrase "I haven't changed anything".
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1254876809938604084)** (7 messages): 

- **Old hardware struggles with AI workloads**: One member shared their difficulties running a language AI on a setup with *"16GB DDR4 and GTX 1060,"* mentioning significant lag despite various settings. Another member humorously noted that older setups *"might as well be in the 1800s"* compared to state-of-the-art cloud AI.
- **High cost of high-performance GPUs**: Members joked about the affordability of *"100,000 GPUS from NVIDIA that cost $40,000-80,000 each."* The conversation underscores the prohibitive costs associated with top-tier AI hardware, with one member adding, *"even if you don't have to ask, you probably still can't afford it."*
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/)** (1 messages): 

uniartisan_86246: I would like to ask if I can set the CPU threads when I am a server
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1254888695530782730)** (5 messages): 

- **Context length woes solved at 3000 tokens**: A user reported stable results at a 3000 token context window and wondered why it was not possible to go higher despite having 4GB RAM available.
- **Lora adapters unsupported in LM Studio**: A member inquired about using learned Lora adapters on GGUF models hosted by LM Studio. Another member responded that "Lora's are unsupported in LM Studio."
  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1255216237928255589)** (1 messages): 

- **Verbose system prompts yield better results**: The behavior handling of a chatbot is controlled through the system prompt. As stated, "*The more verbose the system prompt, the better the results.*"
  

---


### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1255175274597847161)** (11 messages🔥): 

- **Unsupported Gemini Nano model causes local run error**: A user attempted to run **Gemini Nano** on LM Studio locally but faced errors. Another clarified that **Gemini Nano** is unsupported, not meant for use with **llama.cpp** or LM Studio, and is not an official release.
- **LM Studio supports only GGUF models**: When asked if **Gemini Nano** can be quantized to **GGUF**, the response was negative. LM Studio is restricted to **GGUF models**, and hence **Gemini Nano** cannot be used.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1254883110403969075)** (81 messages🔥🔥): 

- **Record Companies Sue AI Music Generators**: Major record companies including **Sony Music Entertainment** and **Universal Music Group** are suing AI music generators such as Suno and Udio for "mass infringement of copyright." The **RIAA** is coordinating the lawsuits, claiming both AI models were trained on copyrighted music without authorization. [Source](https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/).

- **Criticism of AI Training Ethics in Music**: Members discussed how Suno and Udio could avoid copyright issues by processing captions to remove artist names and avoiding model overparameterization. The sentiment was that training on copyrighted material could be seen as fair use if it doesn't lead to memorization, but current AI practices deviate far from this ideal. [Example](https://www.404media.co/listen-to-the-ai-generated-ripoff-songs-that-got-udio-and-suno-sued/).

- **Potential for Open Source Music AI**: The lawsuit against Suno and Udio has led to discussions about creating an open-source, ethically built music model. This model would ideally use public domain or copyright-free songs and innovative architectures to minimize dependence on copyrighted material.

- **Model Training Best Practices**: Conversations included opinions that models trained on fewer parameters could avoid overfitting and memorization, which are primary causes of copyright infringement accusations. There were suggestions to follow proper training methodologies to avoid such pitfalls.

- **AI Image Caption Models**: Brief discussions on finding image captioning models that can exclude certain parts of an image (e.g., people). Although attention masking is proposed, practical implementations and results may vary, with some models like **LLaVA Llama3 8B** potentially having built-in capabilities to ignore specific elements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.404media.co/listen-to-the-ai-generated-ripoff-songs-that-got-udio-and-suno-sued/">Listen to the AI-Generated Ripoff Songs That Got Udio and Suno Sued</a>: The record industry has filed a list of thousands of songs it believes have been scraped without permission, and has recreated versions of famous songs using Udio and Suno.</li><li><a href="https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/">Major record companies sue Suno, Udio for ‘mass infringement’ of copyright</a>: Universal, Sony, Warner come together to sue AI music generators in new lawsuit&#8230;</li><li><a href="https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-gene">Major record companies sue Suno, Udio for ‘mass infringement’ of copyright</a>: Universal, Sony, Warner come together to sue AI music generators in new lawsuit&#8230;</li><li><a href="https://en.wikipedia.org/wiki/Sarah_Kubitschek">Sarah Kubitschek - Wikipedia</a>: no description found</li><li><a href="https://maxread.substack.com/p/my-kindle-thinks-im-stupid-now">My Kindle thinks I&#x27;m stupid now</a>: A journey into Kindle A.I. slop hell with Leah Beckmann</li><li><a href="https://tenor.com/0uH6.gif">Hitchhikers Guide To The Galaxy Leigh574 GIF - Hitchhikers Guide To The Galaxy Leigh574 Question - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1255113208088363101)** (27 messages🔥): 

- **Interest in open multi-modal models stoked**: One user asked if there’s a channel for collaboration on open multi-modal models like those discussed in [this post](https://laion.ai/notes/open-gpt-4-o/), focusing on technologies like GPT-4-OMNI from OpenAI.
- **Carlini defends attack papers**: A member shared a blog post by Nicholas Carlini ([link here](https://nicholas.carlini.com/writing/2024/why-i-attack.html)) where he explains his motivation behind writing attack papers and addresses criticism from Prof. Ben Zhao.
- **Glaze channel deleted due to controversy**: Users discussed that the Glaze channel was deleted, with some speculating it was due to the costs and legal concerns, while others suggested it was to remove past controversial statements.
- **Legal concerns on Nightshade**: A user provided a blog post explaining the potential legal and ethical risks of a protection scheme called Nightshade, pointing out that "[Nightshade](https://undeleted.ronsor.com/nightshade-legal-poison/)" poses significant concerns despite not having an official release yet.
- **Controversy over model poisoning**: It was noted that Prof. Zhao had issues primarily because he encouraged poisoning models, which caused significant backlash within the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://laion.ai/notes/open-gpt-4-o/">Call to Build Open Multi-Modal Models for Personal Assistants | LAION</a>: &lt;p&gt;Technologies like the recently introduced GPT-4-OMNI from OpenAI show again the potential which strong multi-modal models might have to positively transfo...</li><li><a href="https://undeleted.ronsor.com/nightshade-legal-poison/">Nightshade: Legal Poison Disguised as Protection for Artists</a>: As I stated in my previous article, generative AI has continued to be a contentious subject for many artists, and various schemes have appeared in order to ward off model training. The last article, h...</li><li><a href="https://nicholas.carlini.com/writing/2024/why-i-attack.html">
      Why I attack
    </a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1255250846426595428)** (1 messages): 

- **ChatGPT desktop app arrives on macOS**: The **ChatGPT desktop app** for **macOS** is now available to all users. Get [faster access](https://openai.com/chatgpt/mac/) to ChatGPT using the Option + Space shortcut to chat about email, screenshots, and anything on your screen.
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1254883207510495263)** (79 messages🔥🔥): 

- **Curious if LLMs support "Live" chats**: A user inquired if LLMs can initiate questions without being prompted, to which it was clarified as being instruction-related and a feature offered by Gemini.
  
- **Token context window limits debated**: Various users discussed the token context windows of different models like **ChatGPT4**, Claude, and Gemini, noting **ChatGPT's limitations at 8k for free users** and larger capacities available for paid or alternative models (e.g., Claude offers 200k tokens).

- **Distinguishing training and fine-tuning**: Discussion on the difference between **CustomGPT** and actual training via API, elaborating that CustomGPT involves attaching documents for extra knowledge rather than deeply training the model. *"It is not capable of remembering information from individual chats for new chats."* 

- **EvolutionaryScale and hardware advancements**: Shared *[EvolutionaryScale's announcement](https://www.evolutionaryscale.ai/blog/esm3-release)* of ESM3, a model simulating 500M years of evolution, and discussion about *[Sohu, a new specialized AI chip](https://x.com/etched/status/1805625693113663834?s=46&t=tqMTDs9oHX6jLrqq8NrMdw)* to run transformer models more efficiently.

- **Cellular intelligence and its implications for AI**: Discussion on **Michael Levin and Denis Noble's research** on cellular intelligence, and the possibility of mimicking these biological phenomena in AI models for advanced problem-solving abilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.02528">Scalable MatMul-free Language Modeling</a>: Matrix multiplication (MatMul) typically dominates the overall computational cost of large language models (LLMs). This cost only grows as LLMs scale to larger embedding dimensions and context lengths...</li><li><a href="https://x.com/alexrives/status/1805559211394277697">Tweet from Alex Rives (@alexrives)</a>: We have trained ESM3 and we&#39;re excited to introduce EvolutionaryScale.  ESM3 is a generative language model for programming biology. In experiments, we found ESM3 can simulate 500M years of evolut...</li><li><a href="https://ai.meta.com/blog/brain-ai-image-decoding-meg-magnetoencephalography/">no title found</a>: no description found</li><li><a href="https://x.com/etched/status/1805625693113663834?s=46&t=tqMTDs9oHX6jLr">Tweet from Etched (@Etched)</a>: Meet Sohu, the fastest AI chip of all time.  With over 500,000 tokens per second running Llama 70B, Sohu lets you build products that are impossible on GPUs. One 8xSohu server replaces 160 H100s.  Soh...</li><li><a href="https://x.com/etched/status/1805625693113663834?s=46&t=tqMTDs9oHX6jLrqq8NrMdw">Tweet from Etched (@Etched)</a>: Meet Sohu, the fastest AI chip of all time.  With over 500,000 tokens per second running Llama 70B, Sohu lets you build products that are impossible on GPUs. One 8xSohu server replaces 160 H100s.  Soh...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1254888302905921606)** (17 messages🔥): 

- **Context Windows: ChatGPT Struggles with Large Docs**: Users discussed difficulties with ChatGPT handling long documents. The context window is limited to **32k tokens for Plus users** and **8k tokens for GPT-3.5**, suggesting alternative models like **Gemini or Claude** for larger token needs.
- **JSON Output Issues**: A member sought assistance in receiving output from the assistant in valid JSON format. Despite providing instructions, they were unable to achieve valid JSON responses.
- **Math Equations Font Limitation**: Users noted ChatGPT's tendency to output mathematical equations in a specific "math font". Suggestions were made to specify **LaTeX format** for better results.
- **Performance Degradation**: There were complaints about **GPT-4’s recent performance** issues, including increased response times and reliability problems with analytical and historical research queries.
- **Desktop App Availability**: Members expressed frustration about the **desktop app being available only for macOS** with Apple Silicon, noting a Windows version is expected later this year.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1254890043420708954)** (4 messages): 

- **GPT struggles with file-based info**: A user expressed frustration over GPT providing incorrect answers when queried based on uploaded files. They asked for prompt suggestions to improve GPT's reliability in providing accurate information.
- **Streaming code iterations**: A user inquired about saving iterations of code written by GPT for extracting URLs from documents. Another member suggested using the code block expansion and icon to copy and reuse the code in future interactions or their own environment.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1254890043420708954)** (4 messages): 

- **Struggling with prompts for accurate info**: A member asked for recommendations on creating prompts to make GPTs provide reliable information when querying uploaded files. They shared that **GPT sometimes provides wrong answers** based on uploaded documents.
- **Save iteration code for future use**: Another member inquired about saving code iterations for extracting URLs from documents to reuse them through GPT's natural language interface. They were advised to copy the code by expanding the code block or tapping a specific icon and using it in future chats or personal environments.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1254885656837099674)** (90 messages🔥🔥): 

- **Adept prompts necessary for AI art sales**: Discussion highlights that people with advanced prompting skills, artistic backgrounds, and good scene composition have successfully made money selling AI-generated art. *"If you already had some art skills... and got good at this stuff too, otherwise odds are slim you'll sell anything."*
- **Issues with Github and CUDA tests**: Members experienced issues accessing a Github repository, which resolved as a temporary glitch. Another described a *"RuntimeError: Torch is not able to use GPU"* and received advice to check CUDA and PyTorch compatibility.
- **Open Model Initiative debate**: There was mixed reception to the announced Open Model Initiative, with some members doubting its authenticity and others supporting the idea. *"People on the reddit community want to hate it though because ethics were mentioned."*
- **Google Colab overuse concerns**: Users expressed worries about getting banned or flagged for excessive use of Stable Diffusion on Google Colab. *"They will eventually restrict your usage... running it on runpod is literally like 30 cents an hour."*
- **Stability.AI's future questioned**: Concerns were raised about the viability of Stability.AI if they do not fix and un-censor SD3. *"What do they have that can compete and make them money in the current market?"*

**Link mentioned**: <a href="https://civitai.com/articles/5884">Civitai Joins the Open Model Initiative | Civitai</a>: Today, we’re excited to announce the launch of the Open Model Initiative, a new community-driven effort to promote the development and adoption of ...

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1254910403771895952)** (4 messages): 

- **Replete-Coder-Llama3-8b aims for GPT dominance**: A member shared the new Replete-Coder-Llama3-8b model, stating it is "more than just a coding model" and capable of coding in over 100 languages. The model is supported by [TensorDock](https://tensordock.com) for cloud compute and is noteworthy for its uncensored and deduplicated training data of 3.9 million lines.
- **Request for Claude 3.5 support on Ollama**: A GitHub issue requests support for loading the open source Claude 3.5 model on [Ollama](https://github.com/ollama/ollama/issues/5235). The issue was highlighted by a member as part of an ongoing development discussion.
- **Cyborg exclusion raises eyebrows**: A member commented on a perceived "post-humanism bias" in the community, humorously noting the exclusion of cyborgs.
- **Performance improvements highlighted**: A final comment mentioned a significant performance boost, stating it has been "made 12x faster now." This underscores ongoing efforts to optimize existing systems.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>: no description found</li><li><a href="https://github.com/ollama/ollama/issues/5235">Claude 3.5 model · Issue #5235 · ollama/ollama</a>: Is it possible to support loading the open source Claude 3.5 model? Thanks.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1254903587541221397)** (5 messages): 

- **Decision Boundaries in LLMs are irregular**: A post shared a study exploring how the decision boundaries of in-context learning in LLMs compare to traditional models like Decision Trees and KNN. The research reveals *"unexpected irregularities and non-smoothness in LLMs' in-context decision boundaries."* [Read more on arXiv](https://arxiv.org/abs/2406.11233).

- **Sparse Attention for LLM Compression**: A GitHub link to the [official implementation](https://github.com/thu-nics/moa) of the paper "MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression" was shared. The project focuses on using mixtures of sparse attention to achieve automatic compression of large language models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/siyan_zhao/status/1805277462890492321?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Siyan Zhao (@siyan_zhao)</a>: Have you wondered how the decision boundary of in-context learning in LLMs compares to traditional models like Decision Trees and KNN? 🤔  Our research uncovers unexpected irregularities and non-smoot...</li><li><a href="https://github.com/thu-nics/moa">GitHub - thu-nics/MoA: The official implementation of the paper &lt;MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression&gt;</a>: The official implementation of the paper &lt;MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression&gt; - thu-nics/MoA
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1254880860596076557)** (64 messages🔥🔥): 

- **Hypernetwork generates LoRAs**: Discussions surfaced about a hypernetwork capable of generating  LoRAs with a rank of 1. This capability could imply advanced customization options for AI models.
  
- **Clarification on "Nous" in Nous Research**: A member explained that **"Nous"** in Nous Research means "our" in French, representing teamwork and collective passion. Another clarified that it is actually Greek, meaning *intelligence*.

- **Remote Code Execution vulnerability in Ollama**: A [tweet](https://x.com/sagitz_/status/1805261557481312623) highlighted a Remote Code Execution (RCE) vulnerability in one of the popular AI inference projects on GitHub, dubbed **Probllama (CVE-2024-37032)**.

- **Replete-Coder-Llama3-8B announced**: Replete-AI announced their new model, [Replete-Coder-Llama3-8B](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B), boasting general-purpose capabilities and exceptional coding proficiency in over 100 programming languages.

- **Performance claims for Llama 70B**: A Twitter discussion linked [here](https://twitter.com/Etched/status/1805625693113663834?s=19) about Llama 70B's performance claims—500,000 tokens per second—led to skepticism and analysis around probable configurations and realistic expectations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://glif.app/@fab1an/glifs/clxtc53mi0000ghv10g6irjqj">glif - WOJAK MEME GENERATOR by fab1an</a>: no description found</li><li><a href="https://tenor.com/view/cats-toby-the-cat-nod-yes-yes-yes-hooman-gif-17105827">Cats Toby The Cat GIF - Cats Toby The Cat Nod - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/bing-chilling-ice-cream-john-cena-chinese-bing-gif-23622676">Bing Chilling Ice Cream GIF - Bing Chilling Ice Cream John Cena - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cool-beans-thumbsup-gif-13344631">Cool Beans GIF - Cool Beans Thumbsup - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/sagitz_/status/1805261557481312623">Tweet from sagitz (@sagitz_)</a>: We found a Remote Code Execution (RCE) vulnerability in @Ollama - one of the most popular AI inference projects on GitHub. Here is everything you need to know about #Probllama (CVE-2024-37032) 🧵👇</li><li><a href="https://github.com/teknium1/Prompt-Engineering-Toolkit">GitHub - teknium1/Prompt-Engineering-Toolkit</a>: Contribute to teknium1/Prompt-Engineering-Toolkit development by creating an account on GitHub.</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Replete-Coder-Llama3-8B-exl2">bartowski/Replete-Coder-Llama3-8B-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Replete-Coder-Llama3-8B-GGUF">bartowski/Replete-Coder-Llama3-8B-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1254961530223460384)** (5 messages): 

- **Role assignment confirmed**: **Teknium** announced that new roles have been created, tagging multiple users to confirm the setup. **Interstellarninja** followed up by declaring it officially done and encouraged the team to move forward.
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1255079543778054215)** (3 messages): 

- **Unleash Jamba Instruct by AI21**: Check out the new [AI21: Jamba Instruct](https://openrouter.ai/models/ai21/jamba-instruct) model. This model has been added to OpenRouter's lineup for 2023-2024.

- **Navigate NVIDIA's Nemotron-4 340B**: OpenRouter introduces the [NVIDIA Nemotron-4 340B Instruct](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct) model. Available now, it's part of the 2023-2024 range.

- **Explore 01-ai/yi-large**: A new model, [01-ai/yi-large](https://openrouter.ai/models/01-ai/yi-large), is now live on OpenRouter. This release comes under the 2023-2024 collection.

- **Notice: Incorrect Data on Recommended Parameters Tab**: The Recommended Parameters tab on model pages is currently displaying incorrect data. A fix is in progress and an update will be shared soon.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/01-ai/yi-large)">Yi Large by 01-ai</a>: The Yi Large model was designed by 01.AI with the following usecases in mind: knowledge search, data classification, human-like chat bots, and customer service.  It stands out for its multilingual pro...</li><li><a href="https://openrouter.ai/models/ai21/jamba-instruct)">AI21: Jamba Instruct by ai21</a>: The Jamba-Instruct model, introduced by AI21 Labs, is an instruction-tuned variant of their hybrid SSM-Transformer Jamba model, specifically optimized for enterprise applications.  - 256K Context Wind...</li><li><a href="https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct)">NVIDIA Nemotron-4 340B Instruct by nvidia</a>: Nemotron-4-340B-Instruct is an English-language chat model optimized for synthetic data generation. This large language model (LLM) is a fine-tuned version of Nemotron-4-340B-Base, designed for single...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1255221743207972915)** (1 messages): 

- **AI takes on Elite: Dangerous!**: **rudestream** developed an **AI Integration for Elite: Dangerous** acting as a ship computer that reacts and responds to in-game events and player requests. Check out their project on GitHub [here](https://github.com/RatherRude/Elite-Dangerous-AI-Integration) and watch a demo video [here](https://www.youtube.com/watch?v=nvuCwwixvxw).
- **Calls for STT and TTS support**: The developer mentioned creating the project primarily using **free models from OpenRouter** but expressed a desire for support for **Speech-to-Text (STT) and Text-to-Speech (TTS) models**.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=nvuCwwixvxw">A Day in the Life of a Bounty Hunter | Elite: Dangerous AI Integration</a>: 🌟 Project on Github: https://github.com/RatherRude/Elite-Dangerous-AI-Integration       ( github.com/RatherRude/Elite-Dangerous-AI-Integration )💬 Join our ...

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1255008821697318933)** (63 messages🔥🔥): 

- **OR Announcement Post Delay Explained**: Users inquired about the removal of an announcement post, which was explained as being related to the latest Jamba model that required further testing. "*It's up but we needed to test it a bit more.*"

- **Is the Age of LLM Innovation Over?**: A user expressed concerns about a plateau in large language model advancements, noting it's been a year since GPT-4’s release. They recommended a [podcast by Francois Chollet](https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/) discussing the current state and future of AI.

- **AI21's Jamba Instruct Model Issues**: Several users reported errors while using the ai21/jamba-instruct model, causing frustration even after adjusting privacy settings. One user found success after resolving local caching issues and noted inconsistencies between chat and API usage.

- **Handling LLM Instructions**: Users discussed the best practices for handling instructions within LLM prompts, considering alternatives like XML tags for specific models. Fry69_61685 shared a useful resource, the [Anthropic Claude prompt engineering guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview), for further reading.

- **Debate on AI Model Neutrality and Originality**: A discussion arose around the restrictive nature of large corporate LLMs, which avoid taking definitive stances in philosophical or controversial topics. One user emphasized that "*talking to a text wall*" is less preferred compared to models capable of engaging in more dynamic and interesting dialogues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tiktokenizer.vercel.app/">Tiktokenizer</a>: no description found</li><li><a href="https://openrouter.ai/settings#privacy">Settings | OpenRouter</a>: Manage your accounts and preferences</li><li><a href="https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/">280 | François Chollet on Deep Learning and the Meaning of Intelligence &#8211; Sean Carroll</a>: no description found</li><li><a href="https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/).">280 | François Chollet on Deep Learning and the Meaning of Intelligence &#8211; Sean Carroll</a>: no description found
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1254888596528431234)** (57 messages🔥🔥): 

- **"llama.ttf" blends font file with LLM**: A member shared [llama.ttf](https://fuglede.github.io/llama.ttf/?utm_source=changelog-news), a font file incorporating a large language model and an inference engine. This innovative blend uses the HarfBuzz font shaping engine's Wasm shaper to perform text-based LLM inferences.

- **Karpathy advocates for AI World's Fair**: [Karpathy announced](https://x.com/karpathy/status/1805328398920958214?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) the AI World's Fair in SF, highlighting its uniqueness in the AI landscape. Volunteers are needed to manage high logistical demands due to sold-out expo slots and tickets.

- **MARS5 TTS model unveiled**: [MARS5 TTS](https://x.com/reach_vb/status/1805336863101620699) introduces an open-source text-to-speech model with advanced prosodic control, featuring voice cloning from less than 5 seconds of audio. It employs a two-stage architecture with AR and NAR models for precise audio output.

- **EvolutionaryScale raises $142M**: [EvolutionaryScale](https://x.com/pdhsu/status/1805563282746446116) secured $142M to develop its ESM3 model, able to simulate 500M years of evolution for generating new proteins. Key figures like Nat Friedman and Daniel Gross co-led this substantial seed round.

- **Sohu claims fastest AI chip title**: [Sohu](https://x.com/Etched/status/1805625693113663834) emerges as the fastest AI chip, purportedly outpacing Nvidia's Blackwell with 500,000 tokens per second on Llama 70B. Debate surfaces about the genuine performance metrics and comparison methods used in the benchmarks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fuglede.github.io/llama.ttf/?utm_source=changelog-new">llama.ttf</a>: no description found</li><li><a href="https://fuglede.github.io/llama.ttf/?utm_source=changelog-news">llama.ttf</a>: no description found</li><li><a href="https://www.youtube.com/live/EjgTv6aSeqk">PyTorch Documentary Virtual Premiere: Live Stream</a>: Join us for the official release of the PyTorch Documentary! Hear from key players in the project, from the early days to the present.</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Ys4rFt8XbMM">Transformers Explained From The Atom Up (Many Inaccuracies! I Am Fixing Them Now!)</a>: Have you ever wanted to learn how transformers work from the atom up? Well this is the place to learn! :)  Please follow me for more nanotech and AI videos. ...</li><li><a href="https://x.com/pdhsu/status/1805563282746446116?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Patrick Hsu (@pdhsu)</a>: Delighted to co-lead the $142M seed round in EvolutionaryScale with @natfriedman @danielgross @Lux_Capital, a new frontier AI lab that has now trained ESM3, a natively multimodal and generative langua...</li><li><a href="https://x.com/karpathy/status/1805328398920958214?s">Tweet from Andrej Karpathy (@karpathy)</a>: The @aiDotEngineer World&#39;s Fair in SF this week 🔥 https://www.ai.engineer/worldsfair  Reminded of slide #1 from my most recent talk:  &#34;Just in case you were wondering… No, this is not a norma...</li><li><a href="https://x.com/elevenlabsio/status/1805627120028184825?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from ElevenLabs (@elevenlabsio)</a>: Introducing the ElevenLabs Reader App. Listen to any article, PDF, ePub, or any text on the go with the highest quality AI voices.  Download now and have your life narrated: https://elevenlabs.io/text...</li><li><a href="https://x.com/sharongoldman/status/1805562061583253765?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sharon Goldman (@sharongoldman)</a>: NEW: @Meta axed his AI biology research team in 2023. Now @alexrives new startup, Evolutionary Scale, has raised $142 million to continue their work building large language models that generate recipe...</li><li><a href="https://x.com/karpathy/status/1805328398920958214?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Andrej Karpathy (@karpathy)</a>: The @aiDotEngineer World&#39;s Fair in SF this week 🔥 https://www.ai.engineer/worldsfair  Reminded of slide #1 from my most recent talk:  &#34;Just in case you were wondering… No, this is not a norma...</li><li><a href="https://x.com/pdhsu/status/1805563282746446116?">Tweet from Patrick Hsu (@pdhsu)</a>: Delighted to co-lead the $142M seed round in EvolutionaryScale with @natfriedman @danielgross @Lux_Capital, a new frontier AI lab that has now trained ESM3, a natively multimodal and generative langua...</li><li><a href="https://x.com/ylecun/status/1805581310548697360?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Yann LeCun (@ylecun)</a>: http://EvolutionaryScale.ai : an AI-for-proteomics startup that just came out of stealth. They are announcing ESM3 a 98B-paramter generative LLM for &#34;programming biology.&#34;  Using ESM3 and a si...</li><li><a href="https://x.com/Etched/status/1805625693113663834">Tweet from Etched (@Etched)</a>: Meet Sohu, the fastest AI chip of all time.  With over 500,000 tokens per second running Llama 70B, Sohu lets you build products that are impossible on GPUs. One 8xSohu server replaces 160 H100s.  Soh...</li><li><a href="https://x.com/cHHillee/status/1805696613480022238">Tweet from Horace He (@cHHillee)</a>: I&#39;m all for new chips, and it&#39;s great to see new competitors!  That being said, a couple of points that I think are misleading and I see people being confused by:  1. 500k tokens per second is...</li><li><a href="https://x.com/ludwigabap/status/1805571904654254560?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from ludwig (@ludwigABAP)</a>: my guy @jacobrintamaki  made a 20mn video that goes from atoms all the way to pytorch and transformer, &lt;100 views atm  also has a great speedrun video of &#34;from sand to GPU&#34;  can be a great ...</li><li><a href="https://x.com/reach_vb/status/1805336863101620699">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: MARS5 TTS: Open Source Text to Speech with insane prosodic control! 🔥  &gt; Voice cloning with less than 5 seconds of audio &gt; Two stage Auto-Regressive (750M) + Non-Auto Regressive (450M) model ar...</li><li><a href="https://tenstorrent.com/cards/">Cards</a>: Our first fast performance AI computer with PCIe Gen4.</li><li><a href="https://tenstorrent.com/">Tenstorrent</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=40790911">Anthropic Introduces Claude Projects | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1255028418949414992)** (4 messages): 

- **Surprise Pod Preview for AIEWF Conference**: A new episode of the **Latent Space** podcast previews the upcoming AIEWF conference and celebrates the 1-year anniversary of the "Rise of the AI Engineer" podcast. They also feature an interview with @RazRazcle to help launch his new podcast, High Agency. [Listen here](https://x.com/latentspacepod/status/1805468252828844052).

- **Return Guests on Latent Space**: Latent Space releases another surprise podcast featuring **Imbue and Databricks**. The episode discusses the launch of **DBRX** by Databricks and **Imbue 70B**, a new internal LLM claimed to outperform GPT-4 on various benchmarks while using significantly less data. [Listen here](https://www.latent.space/p/llm-training-2024).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1805468252828844052">Tweet from Latent Space Podcast (@latentspacepod)</a>: 🆕 pod alert: Special @aidotengineer Preview Edition!  https://www.latent.space/p/high-agency  We&#39;re so excited to flip the tables with @RazRazcle and help him launch his new podcast, High Agency....</li><li><a href="https://www.latent.space/p/llm-training-2024">State of the Art: Training &gt;70B LLMs on 10,000 H100 clusters</a>: Josh Albrecht, CTO of Imbue, and Jon Frankle, Chief AI Scientist of Databricks, dish on what it takes to train the largest models on the largest clusters... including fighting Infiniband Porch Pirates
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1254925728521388116)** (3 messages): 

- **Meet LlamaIndex in Person This Week**: LlamaIndex shared multiple chances to meet their team in person. *"Wednesday 26 - @jerryjliu0 is giving the closing keynote at AI Engineer World's Fair on the Future of Knowledge Assistants!"* Find more details in their [Twitter Post](https://twitter.com/llama_index/status/1805366745877856363).
- **Optimized RAG with LlamaIndex + DSPy**: LlamaIndex announced a new set of integrations with DSPy, combining DSPy’s PyTorch-esque syntax and optimization capabilities with LlamaIndex's data and orchestration tools for RAG/agents. Read the full announcement on their [Twitter Post](https://twitter.com/llama_index/status/1805622004030284036).

**Link mentioned**: <a href="https://t.co/0rhUE1kpGM">AI Engineer World&#x27;s Fair</a>: Join 2,000 software engineers enhanced by and building with AI. June 25 - 27, 2024, San Francisco.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1254881145770872985)** (52 messages🔥): 

- **Dimensions mismatch issue in PGVectorStore**: A user faced a DataError due to a dimension mismatch when using the bge-small embedding model with PGVectorStore. The issue was resolved after configuring embed_dim correctly and ensuring consistent dimensions across components.

- **Memory issues with HuggingFaceLLM**: A user encountered a ValueError related to memory constraints while using the HuggingFaceLLM with "meta-llama/Meta-Llama-3-8B". The solution suggested not offloading to disk due to performance issues and considering alternative models like ollama for local development.

- **Locating RAG architecture diagrams**: A member sought RAG-related diagrams within LlamaIndex documentation, referencing a specific [arXiv paper](https://arxiv.org/abs/2312.10997). Suggested resources included links to LlamaIndex documentation on [concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) and [agent flow](https://docs.llamaindex.ai/en/stable/understanding/agent/basic_agent/).

- **Retriever top_k setting issue**: A user’s retriever was not respecting the `top_k` setting of 10 nodes, only returning 2 nodes. The problem was resolved by setting `similarity_top_k` to 10 in the retriever configuration.

- **Prompt template with vllm**: Clarification was sought on using prompt templates with vllm in LlamaIndex. It was explained that `messages_to_prompt` and `completion_to_prompt` provide function hooks, and few-shot prompting should be implemented by updating the specific module’s prompt.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2312.10997">Retrieval-Augmented Generation for Large Language Models: A Survey</a>: Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented...</li><li><a href="https://www.youtube.com/watch?v=qwSAKg1YafM">Aqua Voice (W24) Demo</a>: Check it out at withaqua.com</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/llms/vllm/">Vllm - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">High-Level Concepts - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/agent/basic_agent/">Building a basic agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/duckdb_sql_query/?h=sql">SQL Query Engine with LlamaIndex + DuckDB - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1254874684395229255)** (9 messages🔥): 

- **Mojo changelog and git log search tip**: A member shared a useful command, *"TIL: `git log -S'<code text here> -p`"*, and linked the [Mojo changelog](https://github.com/modularml/mojo/blob/1b79ef249f52163b0bafbd10c1925bfc81ea1cb3/docs/changelog.md#v070-2024-01-25). They mentioned the removal date for autotune and how docs were rebuilt with no searchable history prior to three months ago.

- **Future Integration of Torch with Mojo**: Responding to a question about using Torch and Mojo simultaneously, a member explained that though it's not straightforward now, **eventual integration** will make it easy. They emphasized that Mojo aims to combine the Python and C++ capabilities of Torch, suggesting a complete rewrite for performance.

**Link mentioned**: <a href="https://github.com/modularml/mojo/blob/1b79ef249f52163b0bafbd10c1925bfc81ea1cb3/docs/changelog.md#v070-2024-01-25">mojo/docs/changelog.md at 1b79ef249f52163b0bafbd10c1925bfc81ea1cb3 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1805642326129492195>
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1255200900339863632)** (1 messages): 

- **MAX 24.4 hits MacOS with Llama3 and more**: The latest [MAX 24.4 release](https://www.modular.com/blog/whats-new-in-max-24-4-max-on-macos-fast-local-llama3-native-quantization-and-gguf-support) supports MacOS, local Generative AI models like [Llama3](https://llama.meta.com/llama3/), and introduces native quantization and GGUF support. This new feature set allows developers to use a single toolchain for building and deploying Generative AI pipelines with top-tier performance.
- **Running MAX Pipelines**: To explore MAX 24.4 features for GenAI applications, you need to have it [installed](https://docs.modular.com/max/install). Upon successful installation, running `_max -v_` should confirm you have the release version `24.4.0 (59977802)`.

**Link mentioned**: <a href="https://www.modular.com/blog/whats-new-in-max-24-4-max-on-macos-fast-local-llama3-native-quantization-and-gguf-support">Modular: What&#x27;s New in MAX 24.4? MAX on MacOS, Fast Local Llama3, Native Quantization and GGUF Support</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: What&#x27;s New in MAX 24.4? MAX on MacOS, Fast Local Llama3, Native Quantization and GGUF Support

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1254884479076663296)** (3 messages): 

- **Manual data labeling automation on the rise**: A member discussed their work on automating manual data labeling for PDFs with a fine-tuned model, mentioning [Haystack](https://haystack.deepset.ai/) as a promising tool but noted **accuracy is key**. They see potential in integrating this with Quickbooks for ERP systems to alleviate manual data entry, which many users currently conduct.

- **AI for ERP integration gains interest**: Another member showed interest in exploring ERP integration for their standalone tool aimed at labeling large quantities of data. They found the previous conversation about automating data entry processes particularly insightful.

- **ARC Test and AI intelligence debate gets nuanced**: A user commented on the ARC Test, noting it measures culturally common human patterns like closed area, symmetry, and object features. They humorously suggested a dog-focused version of the test with criteria relevant to dogs, like poop smell and bark pitch, and opined that IQ tests don't measure true intelligence, making them easily solvable by AI.

**Link mentioned**: <a href="https://haystack.deepset.ai/">Haystack | Haystack</a>:      Haystack, the composable open-source AI framework   

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1254900143677771796)** (2 messages): 

- **Mojo code without system calls should run on GPUs**: A member pointed out, *"code which doesn’t make system calls (aside from asking for memory), will work on GPUs."* This hint underscores a future capability of Mojo.

- **MAX Graph API to facilitate GPU programming via Mojo**: Brad Larson mentioned, *"one important way to program GPUs via Mojo will be through the MAX Graph API."* Users can currently construct computational graphs targeting CPUs which will be extended to GPUs when support launches in MAX.
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1254895496900776097)** (24 messages🔥): 

- **Mojo lacks `parallelize_tile` feature**: A member inquired about implementing a `parallelize_tile`, and it was clarified that **Mojo currently does not have this feature**. The suggestion was made to pad structs to prevent false sharing in the meantime.

- **Hand-rolled SIMD and Vectorization**: Members discussed **hand-rolled SIMD** and the difference between inlining loops manually versus compiler vectorization. It was noted that the Mojo compiler **disables LLVM's loop vectorizer**.

- **SIMD and SVE Challenges in Mojo**: Concerns were raised about **supporting SVE** (Scalable Vector Extension) due to its unique handling of loop drains and list alignments. One member pointed out that Mojo's current implementation might be artificially restricting SIMD benefits by not supporting these features fully.

- **Feature Request Encouraged**: Members encouraged submitting a **feature request or PR** to align lists to SIMD requirements like 128-bit for NEON or 512-bit for AVX-512. This was in context with potentially adding "hugepage backed if available" lists for better performance.
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1255073564093190215)** (2 messages): 

- **Custom AI models face challenges with Triton Serving setup**: One user inquired about serving a custom AI model written with **MAX graph API** using **Triton Inference Server** and how to structure the model files for compatibility. They referenced the [documentation](https://docs.modular.com/max/serve/get-started) and asked about converting their model into a standard format.
- **MAX graph serde feature in development**: In response, another member mentioned that a target-specific, compiled **MAX graph serde** for such inference use cases is planned for release, advising to "stay tuned" for updates.
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1255037321020047483)** (9 messages🔥): 

- **Mojo releases new nightly compiler versions**: There were announcements for nightly updates to Mojo compilers with versions `2024.6.2505` and `2024.6.2516`. Users can update using `modular update nightly/mojo`, and detailed changelogs are available [here](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) along with [raw diffs](https://github.com/modularml/mojo/compare/ddf9b1183c22fac2d1deb2ad95d7fc90ae051542...6961ce560d0457689f8667986b94a7ea02940cea).
- **List autoderferencing improves performance**: A new change now makes `List[T]` subscript return an autodereference instead of a copy, significantly boosting performance. Similar behavior is needed for `Dict` to achieve another 15%-20% performance improvement.
- **Compiler struggles with boolean expressions**: An issue with handling boolean expressions at compile time was identified, particularly with the `@parameter` decorator. Removing certain parts like `not` or switching to `var` mitigates the problem, possibly related to [this commit](https://github.com/modularml/mojo/commit/57ab0baf2352abee9e83e30967634c2be357afd5).
- **Better reference handling proposed**: A user highlighted that using `__get_ref(k)` for dictionaries can yield better performance. They suggested that switching `__getitem__` to return an auto-dereferencing reference could optimize the current practice.
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1254873923376513186)** (21 messages🔥): 

- **Debate over LingOly Benchmark's Effectiveness**: A user shared [the LingOly benchmark paper](https://arxiv.org/abs/2406.06196v2), sparking discussion on its scope and memorization concerns. One participant highlighted issues with the scoring, while another doubted the benchmark's credibility if the test set is public.

- **Mozilla Honors AI Innovators**: Mozilla's [announcement](https://blog.mozilla.org/en/mozilla/mozilla-announces-finalists-for-the-2nd-annual-rise25-awards) about the 2nd Annual Rise25 Awards celebrating AI innovators prompted congratulations within the community. The honorees were applauded for their ethical and inclusive work in AI.

- **Welcome to New Members**: New members Eitan, a researcher in generative models, and another individual with a keen interest in security exploits, introduced themselves. Eitan shared his background and current work at Lightricks, expressing excitement about joining the EleutherAI Discord.

- **Security Exploits and Model Vulnerabilities**: A user discovered an exploit in a local Llama3 model that allows it to provide instructions for prohibited activities. Initially thought to be a fluke, the exploit was later confirmed to be replicable, raising concerns about model security.

- **Community Celebrations and Greetings**: Multiple users congratulated each other on achievements and announcements, creating a lively and supportive environment. The discussions were punctuated with humor and camaraderie.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06196v2">LINGOLY: A Benchmark of Olympiad-Level Linguistic Reasoning Puzzles in Low-Resource and Extinct Languages</a>: In this paper, we present the LingOly benchmark, a novel benchmark for advanced reasoning abilities in large language models. Using challenging Linguistic Olympiad puzzles, we evaluate (i) capabilitie...</li><li><a href="https://blog.mozilla.org/en/mozilla/mozilla-announces-finalists-for-the-2nd-annual-rise25-awards/">Mozilla announces finalists for the 2nd annual Rise25 Awards | The Mozilla Blog</a>: 25 Honorees and AI Innovators to be celebrated at event in Dublin, Ireland&nbsp; Mozilla is proud to announce the 25 honorees for The 2nd Annual Rise25 Awa</li><li><a href="https://blog.mozilla.org/en/mozilla/mozilla-announces-finalists-for-t">Mozilla announces finalists for the 2nd annual Rise25 Awards | The Mozilla Blog</a>: 25 Honorees and AI Innovators to be celebrated at event in Dublin, Ireland&nbsp; Mozilla is proud to announce the 25 honorees for The 2nd Annual Rise25 Awa
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1254896464472309871)** (17 messages🔥): 

- **MoE favored by specific parameter additions**: A member notes that under "some specific regime," the phenomenon discussed favors Mixture of Experts (MoE) architecture when adding parameters, as it avoids increasing the depth dimension. They argue that "MoE was the optimal way to add parameters."
- **Backdoor vulnerabilities in FL and AI models**: Discussion reveals concerns about federated learning's susceptibility to adversarial backdoor attacks during training, as shown in [this paper](https://arxiv.org/abs/2206.10341). Concerns extend to open weights models, with the thought experiment pondering if large entities like Google or Meta could knowingly distribute models with backdoors.
- **Security risks with open weights**: Members debate whether open weights distribution is less secure than private hosting, since "open weights model developer program" could distribute and activate backdoors without notice. The conversation touches on detection methods and theoretical backdoor inception in widely used models like LLaMA 3.
- **Inefficiencies in homomorphic encryption**: While cryptographic approaches like homomorphic encryption are mentioned as potential solutions to secure federated learning, they are criticized for being "unusably inefficient" and remain primarily theoretical in practice. This inefficiency has led some to suggest avoiding federated learning altogether.
- **Inductive biases in neural networks**: A recent [paper](https://arxiv.org/abs/2403.02241) is lauded for exploring the inductive biases of neural network architectures independently from gradient descent. The paper highlights how alternative architectures can be biased towards complexity and recontextualizes previous understandings regarding these biases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16858">EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees</a>: Inference with modern Large Language Models (LLMs) is expensive and time-consuming, and speculative sampling has proven to be an effective solution. Most speculative sampling methods such as EAGLE use...</li><li><a href="https://arxiv.org/abs/2406.16838">From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models</a>: One of the most striking findings in modern research on large language models (LLMs) is that scaling up compute during training leads to better results. However, less attention has been given to the b...</li><li><a href="https://arxiv.org/abs/2403.02241">Neural Redshift: Random Networks are not Random Functions</a>: Our understanding of the generalization capabilities of neural networks (NNs) is still incomplete. Prevailing explanations are based on implicit biases of gradient descent (GD) but they cannot account...</li><li><a href="https://arxiv.org/abs/2206.10341">Neurotoxin: Durable Backdoors in Federated Learning</a>: Due to their decentralized nature, federated learning (FL) systems have an inherent vulnerability during their training to adversarial backdoor attacks. In this type of attack, the goal of the attacke...</li><li><a href="https://arxiv.org/abs/2401.17948">HyperZ$\cdot$Z$\cdot$W Operator Connects Slow-Fast Networks for Full Context Interaction</a>: The self-attention mechanism utilizes large implicit weight matrices, programmed through dot product-based activations with very few trainable parameters, to enable long sequence modeling. In this pap...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1254882986529394738)** (3 messages): 

- **Neural Redshift paper shared**: A member highlighted an interesting paper related to scaling laws titled *"Neural Redshift: Random Networks are not Random Functions"*[CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Teney_Neural_Redshift_Random_Networks_are_not_Random_Functions_CVPR_2024_paper.pdf). The paper provides insights into how neural network initializations are far more structured than previously assumed.
- **Initializations are key**: Another member emphasized the importance of initializations in AI, stating they are much more significant than researchers often consider. They comically linked this realization to achieving "enlightenment" and shared a [link to AI koans](http://www.catb.org/esr/jargon/html/koans.html), humorous Zen-like stories from the MIT AI Lab.

**Link mentioned**: <a href="http://www.catb.org/esr/jargon/html/koans.html">Some AI Koans</a>: no description found

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1255015122762465353)** (4 messages): 

- **SAEs recover linear features from superposition**: Loganriggs highlighted work by Lee Sharkey and others showing that sparse autoencoders (SAEs) can recover linear features from an overcomplete basis. The source article, [Interim Research Report: Taking Features Out of Superposition](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition), was crossposted from [AI Alignment Forum](https://alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition).
- **Interest in toy models for SAE testing**: Loganriggs expressed interest in other toy models and testing SAEs, inspired by another post from Apollo Research titled [SAE Feature Geometry Is Outside the Superposition Hypothesis](https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis). This post suggests that superposition-based interpretations of neural network activation spaces are limited and highlights the importance of feature geometry.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis">SAE feature geometry is outside the superposition hypothesis — LessWrong</a>: Written at Apollo Research • Summary: Superposition-based interpretations of neural network activation spaces are incomplete. The specific locations…</li><li><a href="https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition.">[Interim research report] Taking features out of superposition with sparse autoencoders — LessWrong</a>: We&#x27;re thankful for helpful comments from Trenton Bricken, Eric Winsor, Noa Nabeshima, and Sid Black.  …
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1255107515172261899)** (5 messages): 

- **Machine Translated ARC Challenge PR sparks debate**: A member sought feedback on a [PR for machine-translated ARC Challenge](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900), mentioning they needed to merge the changes before evaluation results to avoid maintaining a fork. One reviewer approved merging immediately, while another pointed out a miscommunication regarding the method's publication status, ultimately resolving the discussion as a communication error.

**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1900">add arc_challenge_mt by jonabur · Pull Request #1900 · EleutherAI/lm-evaluation-harness</a>: This PR adds tasks for machine-translated versions of arc challenge for 11 languages.  We will also be adding more languages in the future.

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1254900083950882878)** (7 messages): 

```html
<ul>
  <li><strong>Multi app joins OpenAI Family</strong>: <a href="https://multi.app/blog/multi-is-joining-openai">Multi's blog post</a> announced that the app will join OpenAI, exploring how to work with computers alongside AI. Active teams can use the app until July 24, 2024, after which all user data will be deleted.</li>
  <li><strong>Apple Dismisses AI Partnership with Meta</strong>: <a href="https://archive.is/uUv1L">Apple rejected Meta's proposal</a> to integrate the Llama AI chatbot into iPhones, opting instead for deals with OpenAI's ChatGPT and Alphabet's Gemini. Concerns over Meta's privacy practices contributed to Apple's decision.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://multi.app/blog/multi-is-joining-openai">Multi Blog &#x2013; Multi is joining OpenAI </a>: Recently, we&#x2019;ve been increasingly asking ourselves how we should work with computers. Not on or using computers, but truly with computers. With AI. We think it&#x2019;s one of the most importan...</li><li><a href="https://archive.is/uUv1L">Apple Spurned Idea of iPhone AI Partnership With Meta Months Ago - Bl&#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1255268298577154152)** (5 messages): 

- **Rabbithole compromise exposes critical vulnerabilities**: A post about a security disclosure from [rabbitude](https://rabbitu.de/articles/security-disclosure-1) revealed that several **hardcoded API keys** were found in the rabbit codebase. These exposed keys allow unauthorized access to read every response, brick devices, alter responses, and replace voices using services like **ElevenLabs**, **Azure**, **Yelp**, and **Google Maps**.
- **Potential misuse of ElevenLabs credits discussed**: Some members reacted humorously to the security disclosure, contemplating using the compromised **ElevenLabs** credits. One noted, "*Shit isn’t cheap,*" highlighting the potential financial implications of the security breach.

**Link mentioned**: <a href="https://rabbitu.de/articles/security-disclosure-1">rabbit data breach: all r1 responses ever given can be downloaded - rabbitude</a>: rabbit inc has known that we have had their elevenlabs (tts) api key for a month, but they have taken no action to rotate the api keys.

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1254960113890885813)** (27 messages🔥): 

- **Market realizes nVidia's not a monopoly**: With Apple's talk of using their own chips for LLMs server-side, the market is adjusting, realizing "nVidia is not _actually_ a monopoly." This is reflected in changes to the TAM (Total Addressable Market).

- **SemiAnalysis "GPU Rich" discussion snubs TSMC**: It's remarkable that the discussion almost pointedly excluded TSMC’s largest customer, highlighting a significant omission when considering fab capacity.

- **nVidia's 25% drop explained**: Members discussed a sudden "25% one tick drop in nvidia" with some pointing to quirks in Google's stock data due to after-hours trading and lack of liquidity. [Quora](https://www.quora.com/Why-does-Googles-stock-GOOG-frequently-tank-for-a-moment-in-after-hours-trading) and [Money StackExchange](https://money.stackexchange.com/questions/142738/nvda-stock-after-hours-spike-on-google-finance) links were shared for further context.

- **Imbue AI releases new toolkit**: Despite skepticism, Imbue AI released a toolkit for training 70B models optimized for reasoning and coding, including various benchmarks and infrastructure scripts. [Read more](https://imbue.com/research/70b-intro) about their resources released.

- **Hiring experience at Imbue AI**: Members reflected on past experiences interviewing with Imbue AI, which are now seemingly on "a better track," though there are still mixed feelings about their founders and their AGI ambitions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/imbue_ai/status/1805629542914211951">Tweet from Imbue (@imbue_ai)</a>: Early this year, we trained a 70B model optimized for reasoning and coding. This model roughly matches LLAMA 3 70B despite being trained on 7x less data.  Today, we’re releasing a toolkit to help othe...</li><li><a href="https://www.ft.com/content/7332b1f8-cf7c-4bfa-82f4-88d0deb23f98">Nvidia share slide erases more than $550bn in market value</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1254936550370705450)** (10 messages🔥): 

- **AI Lab Security Threats Highlighted**: In a [tweet](https://x.com/jordanschnyc/status/1805340489391997177), an interview with **Alexandr Wang** discusses the urgent need for enhanced AI lab security to prevent espionage risks. Wang emphasized that powerful AI systems could surpass nuclear deterrence, offering capabilities such as "superhuman hacking" and "autonomous drone swarms."

- **Alex is special**: Nathan Lambert expresses appreciation for Alexandr Wang, saying "alex is special". The sentiment is shared in the conversation, with one member confessing, "after watching I do like Alex."

- **Hat market joke**: A humorous exchange occurs with a member joking about being in the market for a hat like Alexandr Wang's, saying, "*if he bought one to rock and one to stock just let him know that I'm in the market*".

**Link mentioned**: <a href="https://x.com/jordanschnyc/status/1805340489391997177">Tweet from Jordan Schneider (@jordanschnyc)</a>: The US government should be terrified about the current state of AI lab security. From our interview with @alexandr_wang releasing on ChinaTalk tomorrow, after I asked him what the US government shoul...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1254911314560811098)** (36 messages🔥): 

- **Replete-Coder-Llama3-8B model makes waves**: A [new model](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) called Replete-Coder-Llama3-8B, fine-tuned by Rombodawg, boasts advanced coding capabilities across over 100 languages. It promises to be a general-purpose model with extensive non-coding data, uncensored and fully cleaned.

- **Vision model confusion with OI and Llama3**: Users like itsahill and bebo.gpt noted that Llama3 is not a vision model, requiring Moondream or GPT4o. Despite attempts with flags such as `--local --vision`, users faced challenges running vision features locally.

- **Success and challenges in OpenInterpreter configurations**: techfren assisted kenharris with the correct flags for executing code with `claude-3-5-sonnet-20240620`, leading to success. However, execution quirks, particularly around function support, raised questions about model compatibility and settings.

- **Vision capability struggles**: Several participants, including daniel_farinax, reported issues and slow processing times when attempting vision tasks locally with configurations like `--os --local --vision`. There were laments about the high cost of OpenAI's vision functionalities and CUDA memory errors when using local GPUs.

**Link mentioned**: <a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>: no description found

  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

m.0861: man
ai videos just give me the creeps yalls
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1254907798022524989)** (32 messages🔥): 

- **ChatOllama updates queried and usage showcased**: A user asked about updates for ChatOllama and shared a link to a [notebook](https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/) showing how to use an experimental wrapper around Ollama with the same API as OpenAI Functions. It highlighted initializing OllamaFunctions and binding functions with JSON Schema parameters.
- **Appending files to chatbot knowledge efficiently**: A user sought advice on appending files to a chatbot's knowledge without reprocessing each file. Another user suggested using a vector database's "add_documents" method to add new documents without recreating the entire index, mentioning the use of "save_local" and "load_local" methods with FAISS.
- **Concurrent requests with OpenAI APIs**: A user asked for help sending notifications to multiple users simultaneously using GPT-4 without individual requests, seeking a concrete asynchronous solution. They previously used batch requests with the completion endpoint but were now struggling with the ChatCompletion endpoint.
- **Streaming responses in Ollama**: A user sought to optimize Ollama's response speed using streaming. Advice was given to import the `ChatOllama` from `langchain_community`, use the `.stream("query")` method, and print tokens iteratively for faster output.
- **Exploring Zep for long-term memory**: A user asked for opinions on using Zep for long-term memory in AI applications. They shared a link to [Zep](https://www.getzep.com/) which integrates with LangChain for persistent conversation summaries and relevant fact retention.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/">OllamaFunctions | 🦜️🔗 LangChain</a>: This notebook shows how to use an experimental wrapper around Ollama that gives it the same API as OpenAI Functions.</li><li><a href="https://github.com/ollama/ollama-python/?tab=readme-ov-file#streaming-responses">GitHub - ollama/ollama-python: Ollama Python library</a>: Ollama Python library. Contribute to ollama/ollama-python development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langgraph/tree/main/examples">langgraph/examples at main · langchain-ai/langgraph</a>: Build resilient language agents as graphs. Contribute to langchain-ai/langgraph development by creating an account on GitHub.</li><li><a href="https://www.getzep.com/">Zep - Long-Term Memory for AI Assistants</a>: Recall, understand, and parse chat dialog to power personalized experiences.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1255007310343045160)** (3 messages): 

- **AI Personal Trainer Toolkit Unveiled**: A member shared their project *Valkyrie*, an AI Personal Trainer built with **NVIDIA and LangChain tools**. They highlight the use of **LangGraph** for execution flow, **LangSmith** for execution tracing, and **NVIDIA AI Foundation Endpoints** for Llama 3 70b LLM-based voice, among other tools. [GitHub - pannaf/valkyrie](https://github.com/pannaf/valkyrie)
- **Instagram Leads Scraper Demo**: Another user showcased a Python script for scraping business leads from Instagram, specifically those in Kentucky, USA. They provided a [Google Sheet](https://docs.google.com/spreadsheets/d/1IYiaqHm_PmX5FdhZTIolxQhE3pWJ5T9X/edit?gid=87752022#gid=87752022) with collected data such as names, bios, emails, websites, and follower counts.
- **Lambda Integration in Visual Agents**: A user added Lambda support to Visual Agents (powered by LangChain) and shared a [YouTube tutorial](https://youtu.be/3xZlvR3aPQI) explaining how to invoke a Lambda function within the flows using a Javascript object payload.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/3xZlvR3aPQI">How to Invoke a Lambda from your Flows</a>: In this video, I show how to wire up a lambda block where it is fed a javascript object payload and an input parameter for choosing which AWS lambda function...</li><li><a href="https://docs.google.com/spreadsheets/d/1IYiaqHm_PmX5FdhZTIolxQhE3pWJ5T9X/edit?gid=87752022#gid=87752022).">Ketuncky Business Data.xlsx</a>: Sheet1  url,username,description,website,email,phone,profile_name,profile_type,posts,followers,following,date-joined,location,verified,facebook,instagram,twitter,tiktok,youtube,linkedin https://www.in...</li><li><a href="https://github.com/pannaf/valkyrie">GitHub - pannaf/valkyrie: V is an AI Personal Trainer, built with NVIDIA and LangChain tools.</a>: V is an AI Personal Trainer, built with NVIDIA and LangChain tools. - pannaf/valkyrie
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1254886695481114694)** (1 messages): 

- **Choosing AI frameworks: Essential questions video**: A member shared a [YouTube video](https://youtu.be/uG0cs8AlnHw) emphasizing the key considerations developers should evaluate before integrating AI into their applications. The video explores models like **GPT-4o, Gemini, Claude, and Mistral**, as well as frameworks like **LangChain**.

**Link mentioned**: <a href="https://youtu.be/uG0cs8AlnHw">Do you even need an AI Framework or GPT-4o for your app?</a>: So, you want to integrate AI into your product, right? Whoa there, not so fast!With models like GPT-4o, Gemini, Claude, Mistral, and others and frameworks li...

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1254874214016487425)** (30 messages🔥): 

- **Claude-3.5-Sonnet rumors dismissed**: A member dismissed seeking insider info on Claude-3.5-Sonnet, noting that they don't know anyone at Anthropic and have no specifics beyond public speculations.

- **Rerank model parameter size remains secret**: When inquired about the size of Cohere rerank models, another member confirmed that this information is not public.

- **Expedition Aya invites global AI collaboration**: [Expedition Aya](https://sites.google.com/cohere.com/expedition-aya/home) is a 6-week initiative by Cohere for AI researchers worldwide to build multilingual AI models, with opportunities for exclusive resources, API credits, and prizes.

- **Clarification on Cohere preambles**: Discussions and shared links provided clarity on preambles used in Cohere's models, including the specifics of the [Command R default preamble](https://docs.cohere.com/docs/preambles) guiding model behavior.

- **Cohere Developer Office Hours held**: Cohere announced and held a Developer Office Hours session on stage to discuss Command R+ tool use and capabilities, encouraging attendees to join via a provided [Discord link](https://discord.gg/CsqyQFhEXT?event=1248301309233336350).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/CsqyQFhEXT?event=1248301309233336350">Join the Cohere Community Discord Server!</a>: Cohere community server. Come chat about Cohere API, LLMs, Generative AI, and everything in between. | 17249 members</li><li><a href="https://sites.google.com/cohere.com/expedition-aya/home">Expedition Aya</a>:   </li><li><a href="https://docs.cohere.com/docs/introduction-to-text-generation?_gl=1*1n88nt9*_gcl_au*MTU2NjA5OTA4LjE3MTcxMzg4NjA.*_ga*MTMwMjc2MTU2NC4xNzA5MjQwMjYw*_ga_CRGS116RZS*MTcxOTMzNTU2MS4xODIuMS4xNzE5MzM4OTc2LjM5LjAuMA..">Introduction to Text Generation</a>: no description found</li><li><a href="https://docs.cohere.com/docs/preambles?_gl=1*dwa5mc*_gcl_au*MTU2NjA5OTA4LjE3MTcxMzg4NjA.*_ga*MTMwMjc2MTU2NC4xNzA5MjQwMjYw*_ga_CRGS116RZS*MTcxOTMzNTU2MS4xODIuMS4xNzE5MzM4OTM3LjExLjAuMA..">Preambles</a>: no description found</li><li><a href="https://docs.cohere.com/docs/introduction-to-text-generation?_gl=1*1n88nt9*_gcl_au*MTU2NjA5OTA4LjE3M">Introduction to Text Generation</a>: no description found
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1254931447073079376)** (19 messages🔥): 

- **'LazyBuffer' object has no attribute 'srcs' debugged**: A user encountered a *`'LazyBuffer' object has no attribute 'srcs'`* error when using `.clip()`. Members suggested using `.contiguous()` instead of `realize`, and mentioned this is a bug in `lazy.py`, with George Hotz noting the need for a fix and additional tests.

- **Debugging CI issues on Mac**: A user inquired about difficulties in reproducing CI errors on a local Mac. Qazalin suggested using Docker with a [specific Dockerfile](https://github.com/Qazalin/containers/blob/main/tinygrad/Dockerfile#L1C1-L14C1) for a Linux environment, which has proven useful in resolving such issues.

- **Bounty for Qualcomm GPU driver**: A user referenced a [Twitter post](https://x.com/__tinygrad__/status/1805317200200581282?t=0bk72a1BFj_jqFAJgwqLjA&s=19) about a $700 bounty for a Qualcomm GPU driver. The post provides instructions on setting up an Android phone with Termux and Tinygrad to aid in development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1805317200200581282?t=0bk72a1BFj_jqFAJgwqLjA&s=19">Tweet from the tiny corp (@__tinygrad__)</a>: We have a $700 bounty for a Qualcomm GPU driver. See ops_amd.py for the Radeon driver, there&#39;s a reason Adreno is an anagram of Radeon.  If you have an Android phone with Qualcomm, you have a dev ...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/5134">&#39;LazyBuffer&#39; object has no attribute &#39;srcs&#39; · Issue #5134 · tinygrad/tinygrad</a>: Minimal Repro: from tinygrad import Tensor timesteps = 16 s = 0.008 def cosine_beta_schedule(timesteps, s=0.008): x = Tensor.arange(0, timesteps+2) alphas_cumprod = (((x / (timesteps+1)) + s) / (1 ...</li><li><a href="https://github.com/Qazalin/containers/blob/main/tinygrad/Dockerfile#L1C1-L14C1">containers/tinygrad/Dockerfile at main · Qazalin/containers</a>: Contribute to Qazalin/containers development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1255000825898733612)** (9 messages🔥): 

- **Llm3 Multimodal Release Concerns**: A member expressed concern that by the time the 72 billion parameter model is finished training, the **llm3 multimodal model** might already be released. They estimated the training to complete around mid-July, which is approximately four epochs taking five days each.
  
- **Lengthy Training Process**: Another member inquired about the dataset size and training duration, to which it was revealed that the dataset takes **5 days per epoch**. The plan includes training for four epochs, culminating after 20 days.

- **Adam-mini Release on arXiv**: A member shared a link to the [Adam-mini optimizer paper](https://arxiv.org/abs/2406.16793) on arXiv, highlighting its significant memory reduction capabilities. **Adam-mini** achieves performance comparable to AdamW while using 45% to 50% less memory by reducing individual learning rates.

**Link mentioned**: <a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the number of learning rates in...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1255240413720215695)** (1 messages): 

- **Creating Cosine LR Scheduler with Minimum LR on Hugging Face**: A user inquired about an easy way to create a cosine learning rate (LR) scheduler on Hugging Face with a minimum LR set to a value greater than 0. The question suggests an interest in customizing learning rate schedules in Hugging Face's framework, likely for better model training performance.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1255095810266366083)** (3 messages): 

- **Minhash optimization excites members**: A user reminisced about finding **minhash** slow when using simple Python. Another user shared that they improved the performance by 12 times and invited others to try it out and provide feedback.
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1255061763817340980)** (8 messages🔥): 

- **Tokenizers config mismatch in Phi-3**: Members discuss differences between tokenizer configurations for Phi-3-mini and Phi-3-medium. The mini config has `"add_bos_token": true`, while the medium has `"add_bos_token": false`; this discrepancy raises questions about its impact on Torchtune.
  
- **Runtime error in TransformerDecoder**: During setup, a traceback error occurs indicating size mismatches in the model parameters, particularly in `attn.q_proj.weight`, `attn.k_proj.weight`, and `attn.v_proj.weight`. These mismatches highlight potential issues in implementing Phi-3-Medium-4K-Instruct support in Torchtune.

- **Phi-3-Medium-4K-Instruct needs more support**: Errors and misconfigurations indicate Phi-3-Medium-4K-Instruct is not fully supported by Torchtune yet. One member humorously points out the ongoing issues by saying, "*seems that still something to do for torchtune to officially support Phi3-Medium-4K-instruct 😂*".

- **Suggestions for tokenizer adjustments**: Contributors suggest creating a `phi3_medium_tokenizer` to remedy the tokenizer configuration differences. "*Just copying and pasting the phi3_mini_tokenizer and setting `add_bos = False`*" is recommended to align with medium settings.
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1255121210790903868)** (3 messages): 

- **Speed Improvements for beowulfbr's Tool**: "Made it 12x faster than datasketch now." Significant efficiency improvement claims were highlighted.
  
- **Search for Conference Talk Zoom Link**: A user requested the Zoom link for "Conference Talk: Language models on the command-line w/ Simon Willison." There were concerns about the video only being embedded in the Maven UI.

- **Simon Willison Shared Talk Video and Notes**: Simon's talk on accessing Large Language Models from the command-line is now available on [YouTube](https://www.youtube.com/watch?v=QUXQNi6jQ30). It includes an [annotated presentation](https://simonwillison.net/tags/annotatedtalks/) with detailed notes and screenshots, focusing on the [LLM Python command-line utility](https://llm.datasette.io/).

**Link mentioned**: <a href="https://simonwillison.net/2024/Jun/17/cli-language-models/">Language models on the command-line</a>: I gave a talk about accessing Large Language Models from the command-line last week as part of the Mastering LLMs: A Conference For Developers &amp; Data Scientists six week long …

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1254892734003548211)** (2 messages): 

- **Automated Dataset Generation Hack Discussed**: A member found a fascinating post about a method to generate a high-quality dataset for LLM instruction finetuning. The hack is *"fully automated and runs locally without seed questions,"* as detailed in the [post](https://x.com/rasbt/status/1805217026161401984).

- **Exploration of Synthetic Aperture Encoding by Linus Lee**: Members discussed Linus Lee's work on building his own **Prism** for finetuning. They referred to [Linus' personal site](https://linus.zone/prism) and his detailed blog post on [Prism](https://thesephist.com/posts/prism/), highlighting that current foundation models are too opaque for humans and need better understandability for richer interfaces.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rasbt/status/1805217026161401984">Tweet from Sebastian Raschka (@rasbt)</a>: I read about a fascinating hack to generate a high-quality dataset for LLM instruction finetuning this weekend. It&#39;s a fully automated way that doesn&#39;t require any seed questions and even runs...</li><li><a href="https://thesephist.com/posts/prism/.">Prism: mapping interpretable concepts and features in a latent space of language | thesephist.com</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/)** (1 messages): 

raminparker: Very cool. Thx for the article!
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1254920051878920222)** (1 messages): 

- **Private Model Loading Issues in Gradio**: A user attempted to create a Gradio space using a private model fine-tuned with AutoTune. They received an error message indicating the need to provide an `hf_token` because the model is in a private repository.
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1254906057256468573)** (2 messages): 

- **llamafile v0.8.7 gets a boost**: [llamafile v0.8.7](https://discord.com/channels/1089876418936180786/1182689832057716778/1254823644320763987) has launched with *faster quant operations* and *bug fixes*. There's also a cryptic hint about potential compatibility with Android.

- **San Francisco's AI events spotlight**: Key IRL events include talks by members at the [World's Fair of AI](https://www.ai.engineer/worldsfair) and hosting duties at the [AI Quality Conference](https://www.aiqualityconference.com/) this week.

- **Investigate Firefox Nightly's AI courage**: Firefox Nightly is testing new optional AI services, details can be explored on the [Nightly blog](https://discord.com/channels/1089876418936180786/1254858795998384239).

- **Catch up with the latest ML research**: The latest [ML Paper Picks](https://discord.com/channels/1089876418936180786/1253145681338830888) are now available, curated by a community member.

- **Engage with Mozilla AI July events**: Upcoming events include talks and sessions like [Jan AI](https://discord.com/events/1089876418936180786/1251002752239407134) and [AutoFix by Sentry.io](https://discord.com/events/1089876418936180786/1245836053458190438), along with the [AI Foundry Podcast Roadshow](https://discord.com/events/1089876418936180786/1253834248574468249).
  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1254883641230758060)** (4 messages): 

- **Llamafile Guidance Needed for New Users**: A member suggested providing a recommended **llamafile and configuration** along with a step-by-step guide to help new users get started. They emphasized the challenge of navigating local LLMs and the importance of making this introduction smooth to avoid deterring novices.

- **Balancing User Accessibility with Advanced Features**: Another member agreed that **more guidance or limited capabilities** might benefit new users. They discussed the possibility of Firefox incorporating built-in local inference that, while slower, could ease users into private on-device inference without requiring complex setups.

- **Llamafile Version Update**: A member shared a link to the [Llamafile 0.8.7 Release](https://www.phoronix.com/news/Llamafile-0.8.7-Released).
  

---



### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1255037953831473203)** (1 messages): 

- **Honeybot.ai Beta Launched**: A beta version for [Honeybot.ai](https://honeybot.ai/) was just launched. The site is intended for adults only and contains **AI-generated adult content**, available for free with the terms of use and privacy policy outlined on the site.

**Link mentioned**: <a href="https://honeybot.ai/"> Honeybot </a>: no description found

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1255035853672288326)** (1 messages): 

- **Honeybot.ai Beta Launch**: A member announced the beta launch of [Honeybot.ai](https://Honeybot.ai), which is an AI-generated adult content platform exclusively for users aged 18 and above. They emphasized that **the platform is completely free** and invited feedback.

**Link mentioned**: <a href="https://Honeybot.ai"> Honeybot </a>: no description found

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1254915683230158891)** (1 messages): 

- **Criticism on project activity**: A user expressed concerns over the current state of the project, highlighting that *"the spam in all channels doesn't really give me the impression this is still an active project."*
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1254913124658839754)** (1 messages): 

- **Detect bots and fraud in the age of LLMs**: An event titled *"Detecting Bots and Fraud in the Time of LLMs"* will be hosted on June 27, 2023, at 10 a.m. PDT. The session will cover the mechanisms of bot and fraud detection, challenges posed by bots, the evolution of LLM usage, and current methodologies to identify and counteract LLM-based bots.
- **Unmesh Kurup speaks on advanced security systems**: The featured speaker is **Unmesh Kurup**, Director of ML at Intuition Machines/hCaptcha, who has a prolific background in AI/ML. Registration for the event is free, and it can be completed [here](https://lu.ma/y6hz8cod?utm_source=diskord).

**Link mentioned**: <a href="https://lu.ma/y6hz8cod?utm_source=diskord">A Million Turing Tests per Second: Detecting bots and fraud in the time of LLMs · Luma</a>: The Data Phoenix team invites you to our upcoming webinar, which will take place on June 27th at 10 a.m. PDT. Topic: A Million Turing Tests per Second:…

  

---



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
