---
id: eab4785c-c252-4c7f-9cf6-1c8e4f50c74e
title: Hybrid SSM/Transformers > Pure SSMs/Pure Transformers
date: '2024-06-13T20:52:25.343318Z'
original_slug: ainews-to-be-named-2494
description: >-
  **NVIDIA**'s new paper on Mamba models reveals that hybrid architectures
  mixing Mamba and Transformer blocks with less than *20%* attention outperform
  pure Transformers, improving accuracy while reducing inference costs.
  Meanwhile, **Stability AI** released **Stable Diffusion 3 Medium**, a *2B*
  parameter model optimized for consumer GPUs with improved image quality and
  new licensing terms, though it faces criticism for anatomical inaccuracies and
  heavy censorship.
tags:
  - nvidia
  - stability
  - lamini
  - hugging-face
  - luma-labs
  - sakana
  - argilla
  - mamba
  - mamba-2-hybrid
  - transformer
  - stable-diffusion-3
  - stable-diffusion-3-medium
  - llama-3
  - table-llava-7b
  - qwen-72b
  - gpt-4
  - mixture-of-experts
  - hybrid-models
  - attention-mechanisms
  - large-language-models
  - multimodal-models
  - text-to-video
  - image-captioning
  - model-benchmarking
  - model-fine-tuning
  - dataset-creation
  - model-licensing
  - model-censorship
  - reasoning
  - coding
  - video-understanding
companies:
  - nvidia
  - stability
  - lamini
  - hugging-face
  - luma-labs
  - sakana
  - argilla
models:
  - mamba
  - mamba-2-hybrid
  - transformer
  - stable-diffusion-3
  - stable-diffusion-3-medium
  - llama-3
  - table-llava-7b
  - qwen-72b
  - gpt-4
topics:
  - mixture-of-experts
  - hybrid-models
  - attention-mechanisms
  - large-language-models
  - multimodal-models
  - text-to-video
  - image-captioning
  - model-benchmarking
  - model-fine-tuning
  - dataset-creation
  - model-licensing
  - model-censorship
  - reasoning
  - coding
  - video-understanding
---


<!-- buttondown-editor-mode: plaintext -->**7% Transformers are all you need.**

> AI News for 6/12/2024-6/13/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**414** channels, and **3646** messages) for you. 
Estimated reading time saved (at 200wpm): **404 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Lots of fun [image-to-video](https://x.com/karpathy/status/1801305852735115357?utm_source=ainews&utm_medium=email) and [canvas-to-math](https://x.com/tldraw/status/1801264226314408029) demos flying around today, but not much technical detail, so we turn elsewhere, to [Bryan Catanzaro of NVIDIA](https://x.com/ctnzr/status/1801050835197026696?utm_source=ainews&utm_medium=email) calling attention to [their new paper](https://arxiv.org/pdf/2406.07887) studying Mamba models:

 ![image.png](https://assets.buttondown.email/images/77bce511-0f83-4f5a-88a1-9cdc8878c4f9.png?w=960&fit=max) 

As Eugene Cheah remarked in the Latent Space Discord, this is the third team (after [Jamba]( https://buttondown.email/ainews/archive/ainews-jamba-mixture-of-architectures-dethrones/) and [Zamba](https://x.com/QuentinAnthon15/status/1780280071304937978) that has independently found the result that mixing Mamba and Transformer blocks does better than either can alone. And the paper does conclude empirically that the optimal amount of Attention is <20%, being FAR from all you need. 

 ![image.png](https://assets.buttondown.email/images/4cf09552-b0ef-4051-8eac-70ce04b41940.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/26f043f7-4f99-41be-b017-9443dab35eee.png?w=960&fit=max) 

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

**LLM Capabilities and Evaluation**

- **Mixture-of-Agents Enhances LLM Performance**: [@bindureddy](https://twitter.com/bindureddy/status/1801010849160818701) noted that Mixture-of-Agents (MoA) uses multiple LLMs in a layered architecture to iteratively enhance generation quality, with the MoA setup using open-source LLMs scoring **65.1% on AlpacaEval 2.0 compared to GPT-4 Omni's 57.5%**.
- **LiveBench AI Benchmark**: [@bindureddy](https://twitter.com/bindureddy/status/1801010849160818701) and [@ylecun](https://twitter.com/ylecun/status/1800897325759701489) announced LiveBench AI, a new LLM benchmark with challenges that can't be memorized. It evaluates LLMs on reasoning, coding, writing, and data analysis, aiming to provide an independent, objective ranking.
- **Mamba-2-Hybrid Outperforms Transformer**: [@ctnzr](https://twitter.com/ctnzr/status/1801050835197026696) shared that an 8B-3.5T hybrid SSM model using 7% attention gets better accuracy than an 8B-3.5T Transformer on the same dataset, with **MMLU jumping from 50 to 53.6%** while having the same training efficiency and lower inference cost.
- **GPT-4 Outperforms at Temperature=1**: [@corbtt](https://twitter.com/corbtt/status/1801026166020833457) found that GPT-4 is "smarter" at temperature=1 than temperature=0, even on deterministic tasks, based on their evaluations.
- **Qwen 72B Leads Open-Source Models**: [@bindureddy](https://twitter.com/bindureddy/status/1801010849160818701) noted that Qwen 72B is the best performing open-source model on LiveBench AI.

**LLM Training and Fine-Tuning**

- **Memory Tuning for 95%+ Accuracy**: [@realSharonZhou](https://twitter.com/realSharonZhou/status/1801271891954696317) announced @LaminiAI Memory Tuning, which uses multiple LLMs as a Mixture-of-Experts to iteratively enhance a base LLM. A Fortune 500 customer case study showed **95% accuracy on a SQL agent task, up from 50% with instruction fine-tuning alone**.
- **Sakana's Evolutionary LLM Optimization**: [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1801080453426024534) highlighted Sakana AI Lab's work using evolutionary strategies to discover new loss functions for preference optimization, outperforming DPO.

**Multimodal and Video Models**

- **Luma Labs Dream Machine**: [@karpathy](https://twitter.com/karpathy/status/1801305852735115357) and others noted the impressive text-to-video capabilities of Luma Labs' new Dream Machine model, which can extend images into videos.
- **MMWorld Benchmark**: [@_akhaliq](https://twitter.com/_akhaliq/status/1801077422676205708) introduced MMWorld, a benchmark for evaluating multimodal language models on multi-discipline, multi-faceted video understanding tasks.
- **Table-LLaVa for Multimodal Tables**: [@omarsar0](https://twitter.com/omarsar0/status/1801271773796716646) shared the Table-LLaVa 7B model for multimodal table understanding, which is competitive with GPT-4V and outperforms existing MLLMs on multiple benchmarks.

**Open-Source Models and Datasets**

- **LLaMA-3 for Image Captioning**: [@_akhaliq](https://twitter.com/_akhaliq/status/1801076206604783888) and [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1801073353899261982) highlighted a paper that fine-tunes LLaVA-1.5 to recaption 1.3B images from the DataComp-1B dataset using LLaMA-3, showing benefits for training vision-language models.
- **Stable Diffusion 3 Release**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1800908975409656014) and others noted the release of Stable Diffusion 3 by Stability AI, which quickly became the #1 trending model on Hugging Face.
- **Hugging Face Acquires Argilla**: [@_philschmid](https://twitter.com/_philschmid/status/1801274502879273009) and [@osanseviero](https://twitter.com/osanseviero/status/1801260106702590375) announced that Argilla, a leading company in dataset creation and open-source contributions, is joining Hugging Face to enhance dataset creation and iteration.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**Stable Diffusion 3 Medium Release**

- **Resource-efficient model**: In /r/StableDiffusion, Stable Diffusion 3 Medium weights were released, a 2B parameter model that is [**resource-efficient and capable of running on consumer GPUs**](https://www.reddit.com/r/StableDiffusion/comments/1de2qne/announcing_the_open_release_of_stable_diffusion_3/). 
- **Improvements over previous models**: SD3 Medium [**overcomes common artifacts in hands and faces, understands complex prompts, and achieves high quality text rendering**](https://www.reddit.com/r/StableDiffusion/comments/1de2qne/announcing_the_open_release_of_stable_diffusion_3/).
- **New licensing terms**: In /r/OpenAI, Stability AI announced new licensing terms for SD3: [**free for non-commercial use, $20/month Creator License for limited commercial use, and custom pricing for full commercial use**](https://www.reddit.com/r/OpenAI/comments/1debbp5/stability_ai_unveils_new_advanced_image_generator/).
- **Mixed initial feedback**: First testers are reporting mixed experiences, with some facing issues replicating results and others giving [**positive feedback on prompt adherence, detail richness, and lighting/colors**](https://www.reddit.com/r/OpenAI/comments/1debbp5/stability_ai_unveils_new_advanced_image_generator/).

**Issues and Limitations of SD3 Medium**

- **Struggles with human anatomy**: In /r/StableDiffusion, users report that SD3 Medium [**struggles with human anatomy, especially when generating images of people lying down or in certain poses**](https://www.reddit.com/r/StableDiffusion/comments/1deav7h/sd3_has_sd_20_level_censorship/). This issue is [further discussed](https://www.reddit.com/r/StableDiffusion/comments/1dehg03/some_nuanced_thoughts_on_stable_diffusion_3/) with nuanced thoughts on the model's limitations.
- **Heavy censorship**: The model appears to have been [**heavily censored, resulting in poor performance when generating nude or suggestive content**](https://www.reddit.com/r/StableDiffusion/comments/1deav7h/sd3_has_sd_20_level_censorship/).
- **Difficulty with artistic styles**: SD3 Medium [**has difficulty adhering to artistic styles and concepts, often producing photorealistic images instead**](https://www.reddit.com/r/StableDiffusion/comments/1dekudj/sd3_artist_styles_and_concepts/).

**Comparisons with Other Models**

- **Varying strengths and weaknesses**: Comparisons between SD3 Medium, SDXL, and other models like Stable Cascade and PixArt Sigma show [**varying strengths and weaknesses across different types of images (photorealism, paintings, landscapes, comic art)**](https://www.reddit.com/r/StableDiffusion/comments/1deb7hb/comparison_of_sd3_and_stable_cascade_a_woman/). Additional [comparison sets](https://www.reddit.com/r/StableDiffusion/comments/1deccwa/image_gen_comparisons_4_sets_each_with_sd3_sdxl/) further highlight these differences.
- **Outperforms in specific areas**: SD3 Medium [**outperforms other models in certain areas, such as generating images of clouds or text**](https://www.reddit.com/r/StableDiffusion/comments/1deb8mu/sd3_cant_produce_a_cloud_in_shape_of_a_cat_while/), but [falls short in others like human anatomy](https://www.reddit.com/r/StableDiffusion/comments/1dedjtv/sd3_is_good_at_text/).

**Community Reactions and Speculation**

- **Disappointment with release**: Many users in /r/StableDiffusion express [**disappointment with the SD3 Medium release, citing issues with anatomy, censorship, and lack of artistic style**](https://www.reddit.com/r/StableDiffusion/comments/1deaahg/sd3_dead_on_arrival/). Some even [call it a "joke"](https://www.reddit.com/r/StableDiffusion/comments/1de9wfz/sd3_is_a_joke/).
- **Speculation on causes**: Some users [**speculate that the poor performance may be due to bugs in adopting the weights or the model architecture**](https://www.reddit.com/r/StableDiffusion/comments/1depcxv/are_we_sure_its_not_a_bug_in_adopting_the_weights/).
- **Reliance on fine-tuning**: Others suggest that [**the community will need to rely on fine-tuning and custom datasets to improve SD3's capabilities, as was done with previous models**](https://www.reddit.com/r/StableDiffusion/comments/1deebnz/for_those_disappointed_with_sd3/).

**Memes and Humor**

- **Poking fun at shortcomings**: Users share [**memes and humorous images poking fun at SD3's shortcomings, particularly its inability to generate anatomically correct humans**](https://www.reddit.com/r/StableDiffusion/comments/1de9xt6/sd3_api_vs_sd3_local_i_dont_get_what_kind_of/). Some even [sarcastically claim "huge success" with the model](https://www.reddit.com/r/StableDiffusion/comments/1deano8/huge_success_with_sd3/).

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Stable Diffusion 3 Faces Scrutiny but Offers Alternatives**:

   - **SD3 Faces Criticism for Model Quality**: Users conveyed dissatisfaction with SD3—highlighting anatomical inaccuracies and prompt issues—while medium models can be [downloaded on Huggingface](https://huggingface.co/stabilityai/stable-diffusion-3-medium).
   - **Preferred Interfaces & Tools Discussed**: **ComfyUI** emerged as the favored interface, with suggested samplers like **uni_pc** and **ddim_uniform** for optimal performance. Alternatives like **Juggernaut Reborn** and [Playground](https://playground.com) are highlighted for their specific capabilities.

2. **Boosting AI Performance and Infrastructure Insights**:

   - **LLM Performance Boosted by Higher Model Rank**: Shifting from rank 16 to 128 resolved **Qwen2-1.5b**'s gibberish output, aligning it with **llama-3** caliber outputs. 
   - **Perplexity AI's Efficient LLM Use**: Quick results are achieved by leveraging **NVIDIA A100 GPUs**, **AWS p4d instances**, and [TensorRT-LLM optimizations](https://www.perplexity.ai/hub/blog/introducing-pplx-api).

3. **Innovations in Fine-Tuning and Quantization**:

   - **Fine-Tuning LLMs with New Models**: The discussion covered the legal aspects of using GPT-generated data, referencing **OpenAI's business terms**. Experimentations with [ToolkenGPT](https://arxiv.org/abs/2305.11554) show creative approaches to synthetic data for fine-tuning.
   - **CUDA Quantization Project discussions**: Projects like the [BiLLM](https://github.com/Aaronhuang-778/BiLLM) showcase rapid quantization of large models, essential for efficient AI deployments.

4. **Model Management and Deployment Techniques**:

   - **Strategies for Handling Large Embeddings**: Queries about **170,000 embedding indexes** led to recommendations on using **Qdrant** or FAISS for faster retrieval. Specific fixes for erroneous queries were shared [here](https://docs.llamaindex.ai).
   - **Docker and GPU Configuration Troubleshooting**: Users dealing with Docker GPU detection on WSL found solutions by consulting the official [NVIDIA toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

5. **AI Community Trends and Updates**:

   - **OpenAI's Revenue Milestone and Focus Shift**: OpenAI's revenue doubled, reflecting sales direct from ChatGPT and other services, not primarily facilitated by Microsoft ([source](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023?utm_source=ti_app&rc=c48ukx)).
   - **Partnerships and Conferences Engage Community**: Aleph Alpha and Silo AI joined forces to advance European AI ([read more](https://aleph-alpha.com/aleph-alpha-and-silo-ai-enter-a-strategic-partnership-to-advance-open-source-ai-and-enterprise-grade-solutions-in-europe/)), and [Qwak's free virtual conference](https://tinyurl.com/j8z6s8ka) promises deep dives into AI mechanisms and networking opportunities.

---

# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**SD3's Rocky Release**: Users have expressed dissatisfaction with **Stable Diffusion 3 (SD3)**, citing issues such as anatomical inaccuracies and non-compliance with prompts compared to **SDXL** and **SD1.5**. Despite the critiques, the medium model of SD3 is now downloadable on [Huggingface](https://huggingface.co/stabilityai/stable-diffusion-3-medium), requiring form completion for access.

**Preferred Interfaces and Samplers**: **ComfyUI** is currently the go-to interface for running **SD3**, and users are advising against Euler samplers. The favored samplers for peak performace with SD3 are **uni_pc** and **ddim_uniform**.

**Exploring Alternatives**: Participants in the channel have highlighted alternative models and tools like **Juggernaut Reborn** and **Divinie Animemix** to achieve more realism or anime style, respectively. Other resources include [Playground](https://playground.com/) and [StableSwarm](https://github.com/Stability-AI/StableSwarmUI?tab=readme-ov-file#installing-on-windows) for managing and deploying models.

**Keep Discussions Relevant**: Moderators have had to direct conversations back on topic after detours into global politics and personal anecdotes sidetracked from the technical AI discussions.

**Big Models, Bigger Needs**: The 10GB model of SD3 was mentioned as a very sought-after option among the community, showing the desire for larger, more powerful models despite the mixed reception of the SD3 release.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Boosting Model Performance with Higher Rank**: Increasing the model rank from 16 to 128 resolved issues with **Qwen2-1.5b** producing gibberish during training, aligning the output quality with results from **llama-3** training.

- **scGPT's Limited Practical Applications**: Despite interesting prompting and tokenizer implementation, **scGPT**, a custom transformer written in PyTorch, is deemed impractical for use outside of an academic setting.

- **Embracing Unsloth for Efficient Inference**: Implementing **Unsloth** has significantly reduced memory usage during both training and inference activities, offering a more memory-efficient solution for artificial intelligence models.

- **Mixture of Agents (MoA) Disappoints**: The **MoA** approach by Together AI, meant to layer language model agents, has been criticized for being overly complex and seemingly more of a showpiece than a practical tool.

- **Advancing Docker Integration for LLMs**: AI engineers are recommending the creation of command-line interface (CLI) tools for facilitating workflows and better integrating notebooks with frameworks like **ZenML** for substantial outcomes in real-world applications.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**SD3 Revolutionizes Stable Diffusion**: **Stable Diffusion 3 (SD3)** has dropped with a plethora of enhancements - now sporting three formidable text encoders ([CLIP L/14](https://huggingface.co/openai/clip-vit-large-patch14), [OpenCLIP bigG/14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k), [T5-v1.1-XXl](https://huggingface.co/google/t5-v1_1-xxl)), a Multimodal Diffusion Transformer, and a 16 channel AutoEncoder. Details of SD3's implementation can be found on the [Hugging Face blog](https://huggingface.co/blog/sd3#summary-of-memory-optimizations).

**Navigating SD3 Challenges**: Users encountered difficulties with **SD3** on different platforms, with recommendations such as applying `pipe.enable_model_cpu_offload()` for faster inference and ensuring dependencies like `sentencepiece` are installed. GPU setup tips include using **RTX 4090**, employing fp16 precision, and making sure paths are correctly formulated.

**Hugging Face Extends Family With Argilla**: In an exciting turn of events, Hugging Face welcomes **Argilla** into its fold, a move celebrated by the community for the potential to advance open-source AI initiatives and new collaborations.

**Community and Support in Action**: From universities, such as the newly created [University of Glasgow organization](https://huggingface.co/UniversityofGlasgow) on Hugging Face, to individual contributions like Google Colab tutorials for **LLM**, members have been contributing resources and sourcing support for their various AI undertakings.

**Enriched Learning Through Shared Resources**: Members are actively exchanging knowledge, with highlighted assets including a [tutorial for LLM setup on Google Colab](https://github.com/casualcomputer/llm_google_colab), a proposed reading group discussion on the **MaPO** technique for text-to-image models, and an [Academic paper on NLP](https://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf) elucidating PCFGs.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Path to AI Expertise**: Aspiring AI engineers were directed towards resources by **Andrej Karpathy** and a [YouTube series by sentdex](https://www.youtube.com/watch?v=GZqYr8_Q7DE) on creating deep learning chatbots. The discussions revolved around the necessary knowledge and skillsets for AI engineering careers.

- **GPT-4.5 Turbo Speculations Ignite Debate**: A debated topic was a leaked mention of **GPT-4.5 Turbo**, speculated to have a 256k context window and a knowledge cutoff in June 2024. It stirred speculation on its potential continuous pretraining feature.

- **Demystifying ChatGPT's Storage Strategy**: It was suggested that better memory management within ChatGPT could involve techniques for collective memory summary and cleanup to resolve current limitations.

- **Teamwork Makes the Dream Work**: Key points about ChatGPT Team accounts were shared, emphasizing the double prompt limit and the financial commitment for multiple team seats when not billed annually.

- **Breaking Down Big Data**: There was advice on managing substantial text data, like **300MB** files, by chunking and trimming them down for practicality. Useful tools and guides were linked, including a [forum post](https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2) with practical tips for large documents and a [notebook on handling lengthy texts](https://cookbook.openai.com/examples/embedding_long_inputs) through embeddings.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Fine-Tuning LLMs: Adding New Knowledge**: AI enthusiasts discussed fine-tuning large language models (LLMs) like "Nous Hermes" to introduce new knowledge, despite costs. A legal debate ensued concerning the use of GPT-generated data, with users consulting [OpenAI's business terms](https://openai.com/policies/business-terms/); a separate mention was made of generating synthetic data referenced in the [ToolkenGPT paper](https://arxiv.org/abs/2305.11554).

- **Technical Glitches and Advice**: In the realm of LLMs, users reported **preprocessing errors** with models like llama-3-8b and mistral-7b. On the practical side, members traded tips on maintaining SSH connections via `nohup`, with recommendations found on this [SuperUser thread](https://superuser.com/questions/448445/run-bash-script-in-background-and-exit-terminal).

- **Innovative Model Frameworks on Spotlight**: Model frameworks gained attention, with **LangChain** and **LangGraph** sparking diverse opinions. The introduction of the [glaive-function-calling-v1](https://huggingface.co/glaiveai/glaive-function-calling-v1) prompted talk about function execution capabilities in models.

- **Deployments and Showcases in Hugging Face Spaces**: Several users announced their RAG-based applications, such as the [RizzCon-Answering-Machine](https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine), built with Gradio and hosted on Hugging Face Spaces, though some noted the need for speed improvements.

- **Credits and Resources Quest Continues**: Queries arose about missing credits and who to contact for platforms like OpenPipe. Users who haven't received credits shared their usernames (e.g., *anopska-552142*, *as-ankursingh3-1-817d86*), and a mention of a second round of credits expected on the 14th was made.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM Objective Discovery Without Human Experts**: A [paper on arXiv](https://arxiv.org/abs/2406.08414) details a method for discovering optimization algorithms for large language models (LLMs) driven by the models themselves, which could streamline the optimization of LLM preferences without needing expert human input. The approach employs iterative prompting of an LLM to enhance performance according to specified metrics.
  
- **MoA Surpasses GPT-4 Omni**: A Mixture-of-Agents (MoA) architecture, as highlighted in a [Hugging Face paper](https://huggingface.co/papers/2406.04692), shows that combining multiple LLMs elevates performance, surpassing GPT-4 Omni with a 65.1% score on AlpacaEval 2.0. For AI enthusiasts wanting to contribute or delve deeper, the MoA model's implementation is available on [GitHub](https://github.com/togethercomputer/moa).

- **Stable Diffusion 3: A Mixed Bag of Early Impressions**: While Stable Diffusion 3 garners both applause and criticism in its initial release, discussions around GPT-4's counterintuitive better performance with higher temperature settings fuel the debate on model configuration. Conversely, a community member circulates an [uncensored version of the OpenHermes-2.5 dataset](https://huggingface.co/datasets/Replete-AI/OpenHermes-2.5-Uncensored), and a paper on [eliminating MatMul operations](https://arxiv.org/abs/2406.02528) promises remarkable memory savings.

- **In Search of the Lost Paper**: Engagement is seen around the task of locating a forgotten paper on **interleaving pretraining with instructions**, suggesting active interest in cutting-edge research sharing within community channels.

- **RAG Dataset Development Continues**: The dataset schema for RAG is still a work in progress, with further optimization needed for Marker's document conversion tool, where setting min_length could boost processing speeds. Simultaneously, Pandoc and make4ht emerge as possible conversion solutions for varied document types.

- **World-sim Project Status Quo**: There's no change yet in the closed-source status of the World-sim project, despite discussions and potential for future reconsideration. Additionally, calls for making the world-sim AI bolder and adapting it for mobile platforms reflect the community's forward-looking thoughts.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Enthusiasm for Perplexity's Search Update**: Members showed great excitement for the recently introduced **search feature** in Perplexity AI, with immediate interest expressed for an **iOS version**.

- **Musk and OpenAI Legal Battle Closes**: Elon Musk has **withdrawn** his lawsuit against **OpenAI**, alleging a shift from mission-driven to profit-orientation, a day prior to the court hearing. The lawsuit included claims of prioritizing investor interests such as those from **Microsoft** ([CNBC](https://www.cnbc.com/2024/06/11/elon-musk-drops-suit-against-openai-and-sam-altman.html)).

- **Perplexity AI Speed with Large Language Models**: **Perplexity.ai** is achieving fast results despite using large language models by utilizing **NVIDIA A100 GPUs**, **AWS p4d instances**, and software optimizations like **TensorRT-LLM** ([Perplexity API Introductions](https://www.perplexity.ai/hub/blog/introducing-pplx-api)).

- **Custom GPT Woes on Perplexity**: Engineers are experiencing **connectivity issues** with **Custom GPTs**; problems seem confined to the web version of the platform as no issues are reported on desktop applications, suggesting potential **API** or **platform-specific complications**.

- **Email's Environmental Footprint**: An average email emits about **4 grams of CO2**; the carbon impact can be mitigated by preferential use of **file share links** over attachments ([Mailjet's Guide to Email Carbon Footprint](https://www.mailjet.com/blog/email-best-practices/email-carbon-footprint/)).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Compute Intensity Discussion Left Hanging**: A member inquired whether the **compute intensity** calculation should consider only floating-point operations on data from **Global Memory**. The topic remained open for discussion without a conclusive answer.

**Streamlined Triton 3.0 Setup**: Two practical installation methods for **Triton 3.0** surfaced; one guide details [installing from source](https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/), while another involves using `make triton` with a [specific version from the PyTorch repository](https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt).

**Optimizing Optimizers in PyTorch**: A robust conversation on creating a fast 8-bit optimizer using **pure PyTorch** and **torch.compile**, as well as making a drop-in replacement for 32-bit with comparable accuracy was had, drawing inspiration from the [bitsandbytes implementation](https://arxiv.org/pdf/2110.02861).

**Breakthroughs in Quantization and Training Dynamics**: The [BiLLM project](https://github.com/Aaronhuang-778/BiLLM) boasts rapid quantization of large language models, while **torchao** members debate the trade-offs in speed and accuracy across various numeric representations during matrix multiplication, from **INT8** to **FP8** and even **INT6**.

**Hardware Showdown and Quantization Innovations**: AMD's **MI300X** showcases higher throughput for LLM inference than NVIDIA's H100, and **Bitnet** sees progress with refactoring and nightly build strategies, but a lingering build issue remains due to an *unrelated mx format test*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Gemini 1.5 JSON Woes**: Engineers report that **Gemini 1.5 flash** struggles with **JSON mode**, causing intermittent issues with output. Users are invited to share insights or solutions to this challenge.

**Tess Takes the Stage**: The **Tess 2.5 72b q3** and **q4 quant models are now live** on Hugging Face, offering new tools for experimentation.

**AVX2 Instruction Essential**: Users facing direct AVX2 errors should verify their **CPU's support for AVX2 instructions** to ensure compatibility with application requirements.

**LM Studio Limitations and Solutions**: **LM Studio** cannot be run on headless web servers or support safetensor files, but it succesfully employs **GGUF format** and **Flash Attention** can be enabled via alternatives like llama.cpp.

**Hardware Market Fluctuations**: There's a spike in the price of **electronically scrapped P40 GPUs** with current prices over $200, as well as a humorous note on sanctions possibly affecting **Russian P40 stocks**. A community member shares specs for an efficient **home server build**: R3700X, 128GB RAM, RTX 4090, and multiple storage options.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLAMA3 70B Shows Diverse Talents**: **LLAMA3 70B** displays a wide-ranging output capability, producing 60% AI2 arc format, 20% wiki text, and 20% code when prompted from an empty document, suggesting a tuning for specific formats. In a separate query, there's guidance on finetuning **BERT** for longer texts with a sliding window technique, pointing to resources such as a [NeurIPS paper and its implementation](https://github.com/Sleepychord/CogLTX).

- **Samba Dances Over Phi3-mini**: **Microsoft's Samba model**, trained on 3.2 trillion tokens, notably outperforms Phi3-mini in benchmarks while maintaining linear complexity and achieving exceptional long-context retrieval capabilities. A different conversation delves into **Samba's passkey retrieval** for 256k sequences, discussing Mamba layers and SWA effectiveness.

- **Magpie Spreads Its Wings**: The newly introduced method **Magpie** prompts aligned Large Language Models (LLMs) to auto-generate high-quality instruction data, circumventing manual data creation. Along this innovative edge, another discussion highlights the controversial practice of tying embedding and unembedding layers, sharing insights from a [LessWrong post](https://www.lesswrong.com/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are).

- **Debating Normalization Standards**: Within the community, the metrics for evaluating models spurred debate, particularly whether to normalize **accuracy by tokens or by bytes** for models with identical tokenizers. A [related log from a test on Qwen1.5-7B-Chat](https://pastebin.ai/i6qnlbg8x3) was shared, discussing solutions for troubleshooting empty responses in `truthfulqa_gen` tasks.

- **Open Flamingo Spreads Its Wings**: A brief message pointed members to [LAION's blog post about Open Flamingo](https://laion.ai/blog/open-flamingo/), a likely reference to their multimodal model work.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**TiDB AI Experimentation on GitHub**: PingCap demonstrates a **RAG application** using their TiDB database with LlamaIndex's knowledge graph, all available as open-source code with a [demo](https://t.co/JKOa6ab1Uh) and the [source code](https://t.co/bUWs9lM1ea) on GitHub.

**Paris AI Infrastructure Meetup Beckons**: Engineers can join an **AI Infrastructure Meetup** at Station F in Paris featuring speakers from LlamaIndex, Gokoyeb, and Neon; details and sign-up are available [here](https://twitter.com/llama_index/status/1801288913312760205).

**Vector Database Solutions for Quick Queries**: For indexes containing 170,000 embeddings, use of **Qdrant** or FAISS Index is recommended; discussion includes fixing an `AssertionError` related to FAISS queries and direct node retrieval from a VectorStoreIndex with **Chroma**.

**Adjacent Node Retrieval from Qdrant**:
A user inquiring about fetching adjacent nodes for law texts in a Qdrant vector store is advised to leverage node relationships and the latest API features for directional node retrieval.

**Pushing LLM-Index Capabilities with PDF Embedding**: An AI Engineer discusses embedding PDFs and documents into Weaviate using LLM-Index, demonstrating interest in expanding the ingestion of complex data types into vector databases.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R Takes the Stage**: **Coral** has been rebranded as **Command-R**, yet both Command-R and the original Coral remain operational facilitating model-related tasks.
- **To Tune or Not to Tune**: In the pursuit of optimal model performance, debate flourished with some engineers emphasizing **prompt engineering** over parameter tuning, while others exchanged note-worthy configurations.
- **Navigating Cohere's Acceptable Use**: A collective effort was noted to decode the nuances of the **[Cohere Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)**, with a focus on delineating private versus commercial usage nuances in the context of personal projects.
- **Trials and Tribulations with Trial Keys**: The community exchanged frustrations regarding **trial keys** encountering permission issues and limitations, contrasting these experiences with the smoother sailing reported by production key users.
- **Hats Off to Fluent API Support**: A quick nod was given in the conversations to the preference for **Fluent API** and appreciation for its inclusion by Cohere, evidenced by a congratulatory tone for a recent project release featuring Cohere support.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Gender Imbalance Troubles in Stable Diffusion**: The community discussed that **Stable Diffusion** has problems generating images of women, clothed or otherwise, due to censorship, suggesting the use of custom checkpoints and img2img techniques with SD1.5 as workarounds. Here's the [discussion thread](https://huggingface.co/stabilityai/stable-diffusion-3-medium/discussions/67).

- **Dream Machine Debuts**: **Luma AI's Dream Machine**, a text-to-video model, has been released and is generating excitement for its potential, though users note its performance is inconsistent with complex prompts. Check out the model [here](https://lumalabs.ai/dream-machine).

- **AI Landscape Survey**: Comparisons across models such as **SD3 Large, SD3 Medium, Pixart Sigma, DALL E 3, and Midjourney** were discussed, alongside the reopening of the /r/StableDiffusion subreddit and Reddit's API changes. The community is keeping an eye on these models and these issues, and the [Reddit post](https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/) offers a comparison.

- **Model Instability Exposed**: A study revealed that models like GPT-4o breakdown dramatically when presented with the **Alice in Wonderland** scenario involving minor changes to input—highlighting a significant issue in reasoning capabilities. Details can be found in the [paper](https://arxiv.org/abs/2406.02061).

- **Recaptioning the Web**: Enhancements in AI-generated captions for noisy web images are on the horizon, with **DataComp-1B** aiming to improve model training by better aligning textual descriptions. For further insights, review the [overview](https://www.haqtu.me/Recap-Datacomp-1B/) and the [scientific paper](https://arxiv.org/abs/2406.08478).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Chasing the Best Speech-to-Text Solution**: Engineers discussed **speech-to-text** solutions, seeking datasets with MP3s and diarization, beyond tools like **AWS Transcribe**, **OpenAI Whisper**, and **Deepgram Nova-2**. The need for robust processing that could handle simple responses without tools and manage streaming responses without losing context was also highlighted.

- **LangChain Links Chains and States**: In **LangChain AI**, integration of user and thread IDs in state management was explicitedly discussed, and tips to leverage **LangGraph** for maintaining state succinctly across various interactions were shared. For message similarity checks within LangChain, both string and embedding distance metrics were suggested with practical use cases.

- **Simplifying LLMs for All**: A **GitHub** project called [tiny-ai-client](https://github.com/piEsposito/tiny-ai-client) was presented to streamline LLM interactions, and a [YouTube tutorial](https://youtu.be/NLOY9RLMI6k?si=-OdUtYSWTJwhvtzy) showed how to set up local executions of LLMs with Docker and Ollama. Meanwhile, another member shared a [GitHub tutorial](https://github.com/casualcomputer/llm_google_colab) to set up LLM on **Google Colab** utilizing the 15GB Tesla T4 GPU.

- **Code Examples and Conversations to Streamline Development**: Throughout the discussions, various **code examples** and issues were referenced to aid in troubleshooting and streamlining LLM development processes, with links like the [Chat Bot Feedback Template](https://python.langchain.com/v0.2/docs/templates/chat-bot-feedback/#usage) and methodologies for *evaluating off-the-shelf evaluators* and *maintaining Q&A chat history* in LangChain.

- **Community Knowledge Sharing**: Members actively shared their own works, methods, and problem-solving strategies, creating a community knowledge base that included how-tos and responses to non-trivial LangChain scenarios, affirming the collaborative ethos of the engineering community.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Windows Woes and Workarounds**: Engineers discussed Windows support for **Modular (Mojo 🔥)**, with a predicted release in the Fall and a [livestream update](https://m.youtube.com/watch?v=uookgZ7Ojg8) anticipated. Meanwhile, some have turned to **WSL** as a temporary solution for Mojo development.

- **Mojo Gets Truthy with Strings**: It was noted that **non-empty strings in Mojo** are considered truthy, potentially causing unexpected results in code logic. Meanwhile, **Mojo LSP** can be set up in **Neovim** with configurations available on [GitHub](https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#mojo).

- **Optimizing Matrix Math**: Benchmarks revealed that **Mojo** has superior performance to **Python** in small, fixed-size **matrix multiplications**, attributing the speed to the high overhead of Python's numpy for such tasks.

- **Loop Logic and Input Handling**: The peculiar behavior of `for` loops in Mojo led to a suggestion to use `while` loops for iterations requiring variable reassignment. Additionally, the current lack of `stdin` support in Mojo was confirmed.

- **Nightly Updates and Compiler Quips**: **Mojo compiler release `2024.6.1305`** was announced, sparking conversations about update procedures with advice to use `modular update nightly/max` and consider aliases for simplification. Discussions also addressed compiler limitations and the potential benefits of the **ExplicitlyCopyable trait** for avoiding implicit copies in the language.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI's Indirect Profit Path**: OpenAI's revenue **surges without Microsoft's aid**, nearly doubling in the last six months primarily due to direct sales of products like ChatGPT rather than relying on Microsoft's channels, contradicting industry expectations. [Read more](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023?utm_source=ti_app&rc=c48ukx).

- **AI Research Goes Full Circle**: **Sakana AI's DiscoPOP,** a state-of-the-art preference optimization algorithm, boasts its origins in AI-driven discovery, suggesting a new era where LLMs can autonomously improve AI research methods. Explore the findings in their [paper](https://arxiv.org/abs/2406.08414) and contribute via the [GitHub repo](https://github.com/SakanaAI/DiscoPOP).

- **Hardware Hype and Research Revelations**: Anticipation builds around Nvidia's potentially forthcoming Nemotron as teased in a [tweet](https://x.com/SebastianB929/status/1800991419437367655), while groundbreaking advancements are discussed with the release of a [paper](https://arxiv.org/abs/2402.16819) exploring speech modeling by researchers like Jupinder Parmar and Shrimai Prabhumoye.

- **SSMs Still in the Game**: The community holds its breath with a 50/50 split on the continuation of **Structured State Machines** (SSMs), despite a leaning interest towards hybrid SSM/transformer architectures as attention layers may not be needed at each step.

- **Benchmarks Blasted by New Architecture**: Introducing **Samba 3.8B**, an architecture merging Mamba and Sliding Window Attention, showcasing it can significantly outclass models like Phi3-mini in major benchmarks, offering infinite context length with linear complexity. Details of Samba's prowess are found in this [paper](https://arxiv.org/abs/2406.07522).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Haize Labs Takes on AI Guardrails**: [Haize Labs](https://x.com/haizelabs/status/1800936720990384174?s=46&t=90xQ8sGy63D2OtiaoGJuww) launched a manifesto on identifying and fixing AI failure modes, demonstrating breaches in leading AI safety systems by successfully jailbreaking their protection mechanisms.

- **tldraw Replicates iPad Calculator with Open Source Flair**: The team behind [tldraw](https://x.com/tldraw/status/1800515870709706879?s=46&t=90xQ8sGy63D2OtiaoGJuww) has reconstructed Apple's iPad calculator as an open-source project, showcasing their commitment to sharing innovative work.

- **Amazon's Conversational AI Missteps Analyzed**: An examination of Amazon's conversational AI progress, or lack thereof, pointed to a culture and operational process that prioritizes products over long-term AI development, according to insights from former employees in an article shared by [cakecrusher](https://www.mihaileric.com/posts/how-alexa-dropped-the-ball-conversational-ai/).

- **OpenAI's Surging Fiscal Performance**: OpenAI has achieved an annualized revenue run rate of nearly [\$3.4 billion](https://x.com/deedydas/status/1801003523292729789), igniting dialogue about the implications of such earnings, including sustainability and spending rates.

- **Argilla Merges with Hugging Face for Better Datasets**: [Argilla](https://argilla.io/blog/argilla-joins-hugggingface) has merged with Hugging Face, setting the stage for improved collaborations to drive forward improvements in AI dataset and content generation.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Empowers LLMs**: **Open Interpreter** is being discussed as a means to transform natural language into direct computer control, offering a bridge to future integrations with tailored **LLMs** and enhanced sensory models.

- **Vision Meets Code in Practical Applications**: The community shared experiences and troubleshooting tips on running code alongside vision models using **Open Interpreter**, particularly focusing on the `llama3-vision.py` profile, and strategies for managing server load during complex tasks.

- **Browser Control Scores a Goal**: A real-world application saw **Open Interpreter** successfully navigating a browser to check live sports scores, showcasing the simplicity of user prompts and the implications on server demand.

- **DIY Approach to Whisper STT**: While seeking a suitable **Whisper Speech-To-Text (STT)** library, a guild member ended up crafting a unique solution themselves, reflecting the community's problem-solving ethos.

- **Tweaking for Peak Performance**: Discussions on fine-tuning **Open Interpreter**, such as altering *core.py*, highlighted the ongoing efforts to address performance and server load challenges to meet the particular needs of users.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Apple's 3 Billion Parameter Breakthrough**: Apple has unveiled a **3 billion parameter** on-device language model at WWDC, achieving the same accuracy as uncompressed models through a strategy that mixes **2-bit and 4-bit configurations**, averaging **3.5 bits-per-weight**. The approach optimizes memory, power, and performance; more details are available in their [research article](https://machinelearning.apple.com/research/introducing-apple-foundation-models).

- **Dockerized AI Hits GPU Roadblock**: An engineer encountered issues when Docker Desktop failed to recognize a GPU on an Ubuntu virtual machine over Windows 11 despite commands like `docker run --gpus all --rm -it winglian/axolotl:main-latest`. Suggested diagnostic steps include checking GPU status with `nvidia-smi` and confirming the installation of the CUDA toolkit.

- **CUDA Confusions and WSL 2 Workarounds**: The conversation shifted towards whether the CUDA toolkit should be set up on Windows or Ubuntu, with a consensus forming around installation within WSL 2 for Ubuntu. A user has expressed intent to configure CUDA on Ubuntu WSL, armed with the official [NVIDIA toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Param Clamping in OpenRouter**: Alex Atallah specified that parameters exceeding support, like Temp > 1, are clamped at 1 for OpenRouter, and parameters like Min P aren't passed through the UI, despite UI presentation suggesting otherwise.

**Mistral 7B's Lag Time Mystery**: Users noticed increased response times for **Mistral 7B** variants, attributing it to context length changes and potential rerouting, supported by data from an [API watcher](https://orw.karleo.net/changes) and a [model uptime tracker](https://openrouter.ai/models/mistralai/mistral-7b-instruct%3Anitro/uptime).

**Blockchain Developer on the Market**: A senior full-stack & blockchain developer is on the lookout for new opportunities, showcasing experience in the field and eagerness to engage.

**Vision for Vision Models**: A request surfaced for the inclusion of more advanced vision models such as cogvlm2 in OpenRouter to enhance dataset captioning capabilities.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Bounty for RDNA3 Assembly in tinygrad**: George Hotz sparked interest with a [bounty for RDNA3 assembly support in tinygrad](https://github.com/tinygrad/tinygrad/pull/3637), inviting collaborators to work on this enhancement.
- **Call for Qualcomm Kernel Driver Development**: An opportunity has arisen for developing a "Qualcomm Kernel level GPU driver with HCQ graph support," targeting engineers with expertise in Qualcomm devices and Linux systems.
- **tinygrad's Mobile Capabilities Confirmed**: Confirmation was given that **tinygrad** is functional within the **Termux** app, showcasing its adaptability to mobile environments.
- **In Discussion: Mimicking Mixed Precision in tinygrad**: A discourse on implementing mixed precision through casting between bfloat16 and float32 during matrix multiplication revealed potential speed benefits, especially when aligned with tensor core data types.
- **Tensor Indexing and UOp Graph Execution Queries**: Efficient tensor indexing techniques are being explored, referencing boolean indexing and UOp graph execution with `MetalDevice` and `MetalCompiler`, with an emphasis on streamlined kernel execution using `compiledRunner`.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **A Dose of Reality in AI Hype**: A blog from dbreunig suggests that the AI industry is predominantly filled with grounded, practical work, comparing the current stage of LLM work to the situation of data science circa 2019. His [article](https://www.dbreunig.com/2024/06/12/sober-ai-is-the-norm.html) hit Hacker News' front page, signaling a high interest in the pragmatic approach to AI beyond the sensationalist outlook.

- **CPU Yesteryear, GPU Today**: The search for computing resources has shifted from RAM and Spark capacity to GPU cores and VRAM, illustrating the changing technical needs as AI development progresses.

- **Token Economics in LLM Deployment**: Databricks’ client data shows a 9:1 input-to-output ratio for Large Language Models (LLMs), highlighting that input token cost can be more critical than output, which has economic implications for those operating LLMs.

- **Spotlight on Practical AI Application**: The recognition of dbreunig's observations at a Databricks Summit by Hacker News underscores community interest in discussions about the evolution and realistic implementation of AI technologies.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **European AI Synergy**: A strategic partnership has been formed between Aleph Alpha and Silo AI to push the frontiers of open-source AI and tailor enterprise-grade solutions for Europe. This collaboration leverages Aleph Alpha’s advanced tech stack and Silo AI’s robust 300+ AI expert team, with an eye to accelerate AI deployment in European industrial sectors. [Read about the partnership](https://aleph-alpha.com/aleph-alpha-and-silo-ai-enter-a-strategic-partnership-to-advance-open-source-ai-and-enterprise-grade-solutions-in-europe/).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Participate in the Poll, Folks**: A user has requested the community's participation in a poll regarding how they serve their finetuned models, with an undercurrent of gratitude for their engagement.

- **Tokenizers Are Getting a Makeover**: An RFC for a comprehensive tokenizer overhaul has been proposed, promoting a more feature-rich, composable, and accessible framework for model tokenization, as found in a [pull request](https://github.com/pytorch/torchtune/pull/1082) on GitHub.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Free Virtual Conference for AI Buffs**: *Infer: Summer '24*, a free virtual conference happening on June 26, will bring together AI and ML professionals to discuss the latest in the field, including **recommender systems** and **AI application in sports**.
- **Industry Experts Assemble**: Esteemed professionals such as **Hudson Buzby**, Solutions Architect at Qwak, and **Russ Wilcox**, Data Scientist at ArtifexAI, will be sharing their insights at the conference, representing companies like **Lightricks, LSports, and Lili Banking**.
- **Live Expert Interactions**: Attendees at the conference will have the chance for real-time engagement with industry leaders, providing a platform to exchange practical knowledge and innovative solutions within ML and AI.
- **Hands-On Learning Opportunity**: Scheduled talks promise insights into the pragmatic aspects of AI systems, such as architecture and user engagement strategies, with a focus on building robust, predictive technologies.
- **Networking with AI Professionals**: Participants are encouraged to network and learn from top ML and AI professionals by [registering for free access](https://tinyurl.com/j8z6s8ka) to the event. The organizers emphasize the event as a key opportunity to broaden AI understanding and industry connections.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Don't Miss the Future of AI Development**: An event discussing how AI software development systems can *amplify developers* is on the horizon. Further info and RSVP can be found [here](https://discord.com/events/1089876418936180786/1242653066512175157).
  
- **Catch Up on Top ML Papers**: The newest **Machine Learning Paper Picks** has been curated for your reading pleasure, available [here](https://discord.com/channels/1089876418936180786/1250679657263534152/322567890986752).

- **Engage with CambAI Team's Latest Ventures**: Join the upcoming event with CambAI Team to stay ahead in the field. RSVP [here](https://discord.com/events/1089876418936180786/1250168740667195455) to be part of the conversation.

- **Claim Your AMA Perks**: Attendees of the AMA, remember to claim your **0din role** to get updates about swag like T-shirts by following the customized link provided in the announcement.

- **Contribute to Curated Conversations with New Tag**: The new `member-requested` tag is now live for contributions to a specially curated discussion [channel](https://discord.com/channels/1089876418936180786/1231977676458168381), reflecting community-driven content curation.

- **Funding and Support for Innovators**: The **Builders Program** is calling for members seeking support and funding for their AI projects, with more details available through the linked announcement.



---



## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **GitHub Codespaces Roll Call**: A survey was launched in #[tech-discussion](https://discord.com/channels/958905134119784489/960713746702020608/1250874031679475743) to determine the usage of GitHub Codespaces among teams, using ✅ for yes and ❌ for no as response options.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1250525139192516609)** (854 messages🔥🔥🔥): 

- **Members express disappointment with SD3 quality**: Users have criticized the quality of the newly released **SD3**, often comparing it unfavorably with **SDXL** and **1.5** versions. Concerns include anatomical inaccuracies, the model not obeying prompts, and reduced quality in photographic styles.
- **SD3 medium released on Huggingface**: The **SD3** medium model is now available for download on [Huggingface](https://huggingface.co/stabilityai/stable-diffusion-3-medium), although users must fill out a form for access. There have been some issues with git cloning, and a 10GB model option is popular.
- **ComfyUI favored for SD3 usage**: **ComfyUI** is currently the preferred interface for running **SD3**, with users suggesting the best samplers and schedulers for optimal results. Recommendations include avoiding Euler samplers and using **uni_pc** and **ddim_uniform** for better performance.
- **Alternative model suggestions and tools**: Members shared alternatives and tools like **Juggernaut reborn** for realistic styles and **Divinie animemix** for anime styles. Other recommended resources for running models include [Playground](https://playground.com/) and [StableSwarm](https://github.com/Stability-AI/StableSwarmUI?tab=readme-ov-file#installing-on-windows).
- **Discussion off-topic around global and political issues**: At times, the channel veered into off-topic discussions about politics, international relations, and personal interactions, distracting from the core focus on AI models and technical help. Mods reminded community members to maintain topical relevance and report inappropriate content.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">stabilityai/stable-diffusion-3-medium · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/muajaja-the-simpsons-gif-7251407">Muajaja Risa Malvada GIF - Muajaja The Simpsons - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://playground.com/">Free AI image generator: Art, Social Media, Marketing | Playground</a>: Playground (official site) is a free-to-use online AI image creator. Use it to create art, social media posts, presentations, posters, videos, logos and more.</li><li><a href="https://youtu.be/Di1KqPXxx2Y?si=CyYMHhaZCzVhNy4N">SD3 IS HERE!! ComfyUI Workflow.</a>: SD3 is finally here for ComfyUI!Topaz Labs: https://topazlabs.com/ref/2377/HOW TO SUPPORT MY CHANNEL-Support me by joining my Patreon: https://www.patreon.co...</li><li><a href="https://tenor.com/view/crycat-crying-cat-crying-cat-thumbs-up-thumbs-up-ok-gif-17048449662472934214">Crycat Crying Cat GIF - Crycat Crying Cat Crying Cat Thumbs Up - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1de65iz/how_to_run_sd3medium_locally_right_now/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides">Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info</li><li><a href="https://imgur.com/a/D0p0cUf">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://github.com/RocketGod-git/stable-diffusion-3-gui/">GitHub - RocketGod-git/stable-diffusion-3-gui: GUI for Stable Diffusion 3 written in Python</a>: GUI for Stable Diffusion 3 written in Python. Contribute to RocketGod-git/stable-diffusion-3-gui development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://opendata.blender.org/">Blender - Open Data</a>: Blender Open Data is a platform to collect, display and query the results of hardware and software performance tests - provided by the public.</li><li><a href="https://stability.ai/news/deepfloyd-if-text-to-image-model">Stability AI releases DeepFloyd IF, a powerful text-to-image model that can smartly integrate text into images &mdash; Stability AI</a>: DeepFloyd IF is a state-of-the-art text-to-image model released on a non-commercial, research-permissible license that allows research labs to examine and experiment with advanced text-to-image genera...</li><li><a href="https://stability.ai/stable-assistant-gallery">Stable Assistant Gallery &mdash; Stability AI</a>: Stable Assistant delivers image creation capabilities like never before. Explore our gallery and be amazed by what's possible!</li><li><a href="https://stability.ai/stable-assistant-gallery?utm_campaign=Stable%20Assistant&utm_content=186946249&utm_medium=social&utm_source=twitter&hss_channel=tw-1281048162602369024">Stable Assistant Gallery &mdash; Stability AI</a>: Stable Assistant delivers image creation capabilities like never before. Explore our gallery and be amazed by what's possible!</li><li><a href="https://imgsys.org/">imgsys.org | an image model arena by fal.ai</a>: A generative AI arena where you can test different prompts and pick the results you like the most. Check-out the model rankings and try it yourself!</li><li><a href="https://civitai.com/models/490622/mobius">Mobius - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: Mobius: Redefining State-of-the-Art in Debiased Diffusion Models Mobius, a diffusion model that pushes the boundaries of domain-agnostic debiasing ...
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1250526449300017302)** (446 messages🔥🔥🔥): 

- **Rank boosts give better training results**: "I can confirm that the issues with Qwen2-1.5b training and it outputting gibberish is to do with too low of a rank. I switched from rank 16 to rank 128, and my outputs matched the quality of my outputs from my llama-3 train."
- **scGPT might not be practical outside academia**: "That's a custom transformer implemented in torch 😅 Unusable outside academia as is...interesting what they did on the prompting and the tokenizer!"
- **Inference support and memory efficiency in Unsloth**: A user asked if Unsloth affects inference and received confirmation that Unsloth reduces memory usage significantly during both training and inference.
- **Mixture of Agents (MoA) does not impress**: Users discussed Together AI’s [MoA](https://www.together.ai/blog/together-moa) approach to layering LLM agents but found it "kind of just for show" and overly complex.
- **Dockerizing for seamless pipelines**: "The notebooks are a great 'jump in' but in real-world apps you want to integrate that with other stuff. Making CLI tools for consistent workflows and easier integrations with frameworks like ZenML."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=l8pRSuU81PU">Let&#39;s reproduce GPT-2 (124M)</a>: We reproduce the GPT-2 (124M) from scratch. This video covers the whole process: First we build the GPT-2 network, then we optimize its training to be really...</li><li><a href="https://www.together.ai/blog/together-moa">Together MoA — collective intelligence of open-source models pushing the frontier of LLM capabilities</a>: no description found</li><li><a href="https://huggingface.co/google/recurrentgemma-9b">google/recurrentgemma-9b · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE">Let&#39;s build the GPT Tokenizer</a>: The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizer...</li><li><a href="https://tenor.com/view/smh-shaking-my-head-sneaky-gif-21653713">Smh Shaking GIF - Smh Shaking My - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://machinelearning.apple.com/research/introducing-apple-foundation-models">Introducing Apple’s On-Device and Server Foundation Models</a>: At the 2024 Worldwide Developers Conference, we introduced Apple Intelligence, a personal intelligence system integrated deeply into…</li><li><a href="https://github.com/bowang-lab/scGPT/tree/main/tutorials/zero-shot">scGPT/tutorials/zero-shot at main · bowang-lab/scGPT</a>: Contribute to bowang-lab/scGPT development by creating an account on GitHub.</li><li><a href="https://github.com/bowang-lab/scGPT">GitHub - bowang-lab/scGPT</a>: Contribute to bowang-lab/scGPT development by creating an account on GitHub.</li><li><a href="https://github.com/bowang-lab/scGPT/tree/integrate-huggingface-model">GitHub - bowang-lab/scGPT at integrate-huggingface-model</a>: Contribute to bowang-lab/scGPT development by creating an account on GitHub.</li><li><a href="https://github.com/ollama/ollama/tree/main/examples">ollama/examples at main · ollama/ollama</a>: Get up and running with Llama 3, Mistral, Gemma, and other large language models. - ollama/ollama</li><li><a href="https://www.ncbi.nlm.nih.gov/guide/howto/dwn-genome/">Download the complete genome for an organism</a>: no description found</li><li><a href="https://arxiv.org/abs/2406.05955">Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters</a>: Exploiting activation sparsity is a promising approach to significantly accelerating the inference process of large language models (LLMs) without compromising performance. However, activation sparsit...</li><li><a href="https://arxiv.org/abs/2406.06282">PowerInfer-2: Fast Large Language Model Inference on a Smartphone</a>: This paper introduces PowerInfer-2, a framework designed for high-speed inference of Large Language Models (LLMs) on smartphones, particularly effective for models whose sizes exceed the device&#39;s ...</li><li><a href="https://github.com/sebdg/unsloth/tree/cli-trainer">GitHub - sebdg/unsloth at cli-trainer</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - GitHub - sebdg/unsloth at cli-trainer</li><li><a href="https://finalspark.com/neuroplatform/">Neuroplatform - FinalSpark</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/code_bagel_hermes-2.5">Replete-AI/code_bagel_hermes-2.5 · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1250525091499085875)** (190 messages🔥🔥): 

- **Trainer and TrainingArguments Confusion**: A user expressed confusion about the specifics of [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) and [TrainingArguments](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) in the Huggingface documentation. They noted that the descriptions do not fully explain how to use these classes.

- **Saving as gguf Issues**: A user faced a `ValueError` when attempting to save a model as gguf, but found success after specifying `f16` as the quantization method. They shared this solution along with the syntax: `model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")`.

- **Untrained Tokens Error**: Another user encountered a `ValueError` related to untrained tokens while training a model, suggesting that `embed_tokens` and `lm_head` must be included in the training process. This issue was linked to adding new tokens that require enabling training on certain model parts.

- **Dataset Formatting for Multilabel Classification**: One member sought advice on finetuning Llama 3 for multilabel classification and whether the same prompt format could be used. While responses acknowledged the need for consistent dataset templates, no specific solution for multilabel classification was provided.

- **citing Unsloth**: Users discussed how to cite Unsloth in a paper, with a suggestion to reference it as: "Daniel Han and Michael Han. 2024. Unsloth, Unsloth AI." followed by the [Unsloth GitHub page](https://github.com/unslothai).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/afrizalha/Kancil-V1-llama3-fp16">afrizalha/Kancil-V1-llama3-fp16 · Hugging Face</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/how-do-you-calculate-max-steps/40177">How do you calculate max steps</a>: I am looking to understand the math behind how max steps is calculated when left alone, I’ve tried to work backwards from making changes to epoch, batch, and micro-batch to see if I could figure out t...</li><li><a href="https://github.com/unslothai/unsloth/wiki#continued-pretraining--finetuning-the-lm_head-and-embed_tokens-matrices">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments">Trainer</a>: no description found
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1250557790208917638)** (8 messages🔥): 

- **Stable Diffusion 3 Launches with Enhanced Features**: *SD3 is a diffusion model that includes three text encoders* ([CLIP L/14](https://huggingface.co/openai/clip-vit-large-patch14), [OpenCLIP bigG/14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k), and [T5-v1.1-XXL](https://huggingface.co/google/t5-v1_1-xxl)), a Multimodal Diffusion Transformer, and a 16 channel AutoEncoder model. Check the complete details and code on the [Stable Diffusion 3 Medium space](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium).
- **Community Highlights #62**: Featuring new tools and projects like a [formula 1 prediction model](https://huggingface.co/posts/Draichi/560425192506443), [Simpletuner v0.9.7](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7), and [Tiny LLM client](https://github.com/piEsposito/tiny-ai-client). Check out the [1M+ Dalle 3 captioned dataset](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions) and [MARS v5](https://github.com/camb-ai/mars5-tts) for more cool stuff.
- **Argilla Joins Hugging Face**: *Today's a huge day for Argilla, Hugging Face, and the Open Source AI community: Argilla is [joining](https://huggingface.co/posts/dvilasuero/203008804842390) Hugging Face*! Emphasizing the community and data-centric, open source AI approach.
- **Community Welcomes Argilla**: Members congratulated Argilla on joining Hugging Face and expressed excitement for future projects and collaborations. *"Looking forward to my first distilabel x arguilla project (of course hosted on hf!)"*.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">stabilityai/stable-diffusion-3-medium · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/sd3">Diffusers welcomes Stable Diffusion 3</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1250525199183642816)** (185 messages🔥🔥): 

- **Troubleshooting SD3 Issues with Diffusers**: Users faced multiple errors while trying to run the Stable Diffusion 3 (SD3) model with the latest diffusers library update. One user shared that adding `pipe.enable_model_cpu_offload()` significantly improved inference time.

- **University of Glasgow Organization on HuggingFace**: A member announced the creation of the University of Glasgow organization on HuggingFace, inviting faculty, researchers, and students to join using their university email. [Link to the organization](https://huggingface.co/UniversityofGlasgow).

- **Error in LLM Session Management**: A user sought help for maintaining sessions with their local LLM using the Ollama interface with Llama3. Another member provided a Python script solution for continuous session management using the OpenAI API format.

- **Generalized LoRA (GLoRA) Implementation Help**: A user working on integrating GLoRA into the PEFT library requested assistance with an error in their forked implementation. They provided [a link to the GitHub repository](https://github.com/viliamvolosv/peft) and related [research paper](https://arxiv.org/abs/2306.07967).

- **Tutorial for LLM on Google Colab**: A member shared a [tutorial for setting up LLM on Google Colab](https://github.com/casualcomputer/llm_google_colab) to take advantage of the free GPU for both GPU-accelerated and CPU-only inference. They highlighted the tutorial as beneficial for others encountering similar issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nroggendorff/sd3">Stable Diffusion 3 - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://huggingface.co/spaces/TheStinger/Ilaria_RVC">Ilaria RVC - a Hugging Face Space by TheStinger</a>: no description found</li><li><a href="https://huggingface.co/UniversityofGlasgow">UniversityofGlasgow (University of Glasgow)</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/LanguageBind/Video-LLaVA">Video LLaVA - a Hugging Face Space by LanguageBind</a>: no description found</li><li><a href="https://arxiv.org/abs/2306.07967">One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning</a>: We present Generalized LoRA (GLoRA), an advanced approach for universal parameter-efficient fine-tuning tasks. Enhancing Low-Rank Adaptation (LoRA), GLoRA employs a generalized prompt module to optimi...</li><li><a href="https://huggingface.co/Helsinki-NLP/opus-mt-ko-en">Helsinki-NLP/opus-mt-ko-en · Hugging Face</a>: no description found</li><li><a href="https://github.com/viliamvolosv/peft">GitHub - viliamvolosv/peft: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.</a>: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - viliamvolosv/peft</li><li><a href="https://github.com/casualcomputer/llm_google_colab">GitHub - casualcomputer/llm_google_colab: A tutorial on how to set up a LLM on Google Colab for both GPU-accelerated and CPU-only session.</a>: A tutorial on how to set up a LLM on Google Colab for both GPU-accelerated and CPU-only session. - casualcomputer/llm_google_colab</li><li><a href="https://github.com/VedankPurohit/LiveRecall">GitHub - VedankPurohit/LiveRecall: Welcome to **LiveRecall**, the open-source alternative to Microsoft&#39;s Recall. LiveRecall captures snapshots of your screen and allows you to recall them using natural language queries, leveraging semantic search technology. For added security, all images are encrypted.</a>: Welcome to **LiveRecall**, the open-source alternative to Microsoft&amp;#39;s Recall. LiveRecall captures snapshots of your screen and allows you to recall them using natural language queries, leverag...</li><li><a href="https://github.com/continuedev/deploy-os-code-llm">GitHub - continuedev/deploy-os-code-llm: 🌉 How to deploy an open-source code LLM for your dev team</a>: 🌉 How to deploy an open-source code LLM for your dev team - continuedev/deploy-os-code-llm</li><li><a href="https://tenor.com/view/smol-illegally-smol-cat-cute-cat-boop-gif-3484507763170497045">Smol Illegally Smol Cat GIF - Smol Illegally smol cat Cute - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://colab.research.google.com/drive/1Rlcbd3SCibgJQmzAGZW7pyiK01pwumB4?usp=sharing>">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1250530466365177856)** (10 messages🔥): 

- **Distill's invaluable yet inactive resources**: Despite being inactive, [Distill](https://distill.pub/) offers highly illustrative articles on various ML and DL topics. Highlights include *Understanding Convolutions on Graphs* and *A Gentle Introduction to Graph Neural Networks*.
  
- **Enterprise LLMs security query sparks interest**: A member inquired if anyone is working on security for Enterprise level LLMs. This reflects a growing concern in the community regarding AI model security.
  
- **Academic dive into NLP and Graph Models**: Various academic resources were shared including [NLP notes on PCFGs](https://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf) and a pioneering approach to neural ODE models on [arXiv](https://arxiv.org/abs/2209.03003).
  
- **DreamMachine by Luma AI hailed as Pikalabs successor**: [DreamMachine by Luma AI](https://lumalabs.ai/dream-machine) was celebrated as a successor to Pikalabs, offering 30 free videos a month with img2vid and text2vid features. Prompting tips were discussed to optimize results, with a suggestion to keep prompts simple and let the system's model assume control.
  
- **Nodus Labs' ACM paper**: A member shared a [paper from ACM](https://dl.acm.org/doi/10.1145/3308558.3314123) which may be insightful for those interested in further technical details.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2209.03003">Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow</a>: We present rectified flow, a surprisingly simple approach to learning (neural) ordinary differential equation (ODE) models to transport between two empirically observed distributions π_0 and π_1, henc...</li><li><a href="https://distill.pub/">Distill — Latest articles about machine learning</a>: Articles about Machine Learning</li><li><a href="https://ciechanow.ski/archives/">Archives - Bartosz Ciechanowski</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1250643738313887877)** (11 messages🔥): 

- **Colab Tutorial tackles LLM setup**: A member shared a [tutorial on setting up a LLM on Google Colab](https://github.com/casualcomputer/llm_google_colab) to leverage the free 15GB Tesla T4 GPU for both GPU-accelerated and CPU-only inference sessions. They hope it will assist others struggling with existing solutions and troubleshooting problems.

- **Tiny-AI-Client simplifies LLM usage**: Another member introduced a [tiny, intuitive client for LLMs](https://github.com/piEsposito/tiny-ai-client) that supports vision and tool use, aiming to be an alternative to langchain for simpler use cases. They offered to help with bugs and invited others to try it out.

- **SimpleTuner release integrates SD3**: A new [release of SimpleTuner](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7) was announced, fully integrating Stable Diffusion 3's unet and lora training. The member shared their dedication to the project, working tirelessly to achieve this integration.

- **French Deep Learning Notebooks**: A GitHub repository containing [Deep Learning notebooks in French](https://github.com/SimonThomine/CoursDeepLearning) was shared, aimed at making Deep Learning more accessible. The course, inspired by resources from Andrej Karpathy and DeepLearning.ai, is a work in progress.

- **Conceptual Captions dataset**: A member shared a [massive dataset](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext) with 22 million high-quality captions for 11 million images from Google's CC12M, created using LLaVaNext.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Hatman/pixel-prompt">Pixel Prompt - a Hugging Face Space by Hatman</a>: no description found</li><li><a href="https://huggingface.co/posts/Draichi/560425192506443">@Draichi on Hugging Face: &quot;Hey Hugging Face Community 🤗

I&#39;m excited to share my latest project that…&quot;</a>: no description found</li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7">Release v0.9.7 - stable diffusion 3 · bghira/SimpleTuner</a>: Stable Diffusion 3 To use, set STABLE_DIFFUSION_3=true in your sdxl-env.sh and set your base model to stabilityai/stable-diffusion-3-medium-diffusers.  What&#39;s Changed  speed-up for training sample...</li><li><a href="https://github.com/casualcomputer/llm_google_colab">GitHub - casualcomputer/llm_google_colab: A tutorial on how to set up a LLM on Google Colab for both GPU-accelerated and CPU-only session.</a>: A tutorial on how to set up a LLM on Google Colab for both GPU-accelerated and CPU-only session. - casualcomputer/llm_google_colab</li><li><a href="https://github.com/SimonThomine/CoursDeepLearning">GitHub - SimonThomine/CoursDeepLearning: Un regroupement de notebooks de apprendre le deep learning à partir de 0</a>: Un regroupement de notebooks de apprendre le deep learning à partir de 0 - SimonThomine/CoursDeepLearning</li><li><a href="https://github.com/piEsposito/tiny-ai-client">GitHub - piEsposito/tiny-ai-client: Tiny client for LLMs with vision and tool calling. As simple as it gets.</a>: Tiny client for LLMs with vision and tool calling. As simple as it gets. - piEsposito/tiny-ai-client</li><li><a href="https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext">CaptionEmporium/conceptual-captions-cc12m-llavanext · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1250561970114134106)** (1 messages): 

- **Proposal to discuss MaPO**: A member suggested inviting one of the authors to present the paper MaPO, highlighting the potential interest within the community. The paper discusses a novel alignment technique for text-to-image diffusion models that avoids the limitations of divergence regularization and is more flexible in handling preference data [MaPO](https://mapo-t2i.github.io/).

**Link mentioned**: <a href="https://mapo-t2i.github.io/">MaPO Project Page</a>: SOCIAL MEDIA DESCRIPTION TAG TAG

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1250557622910980106)** (28 messages🔥): 

- **Help on Image Retrieval Models**: A newcomer seeks advice on the best models for image retrieval, mentioning success with Microsoft's Vision API and CLIP. A suggestion includes using **Mobilenets** for real-time apps, **OpenCLIP** for best scores, and **Faiss** for versatile search engine capabilities.
- **Recommendation of smol-vision**: A member shares a [GitHub link](https://github.com/merveenoyan/smol-vision) for **smol-vision**, which provides *recipes for shrinking, optimizing, and customizing cutting edge vision models*.
- **Channel Redirection**: Guidance was given to post in a more relevant channel, <#1019883044724822016>, for deeper insights and recommendations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/smash-gif-21365305">Smash GIF - Smash - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/merveenoyan/smol-vision">GitHub - merveenoyan/smol-vision: Recipes for shrinking, optimizing, customizing cutting edge vision models. 💜</a>: Recipes for shrinking, optimizing, customizing cutting edge vision models. 💜  - GitHub - merveenoyan/smol-vision: Recipes for shrinking, optimizing, customizing cutting edge vision models. 💜
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1250680480085311488)** (1 messages): 

- **Finetuning Llama3 for MultiLabel Classification**: A member inquired about the method for finetuning **Llama3** for **MultiLabel classification**, asking if the same prompt format used in a [specific Kaggle notebook](https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook/notebookhere) should be followed for each row, or if there exists another method similar to BERT.
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1250562702632419451)** (79 messages🔥🔥): 

- **Different results between schedulers with SD3**: A user noted they were getting different results between **Comfy** and **Diffusers** when testing different schedulers with **SD3**. They solicited input from others to see if anyone else had similar experiences.

- **Stable Diffusion 3 Medium model released**: **Stable Diffusion 3 Medium** with 2B parameters has been released and is available on the **Hugging Face Hub**. The model includes integrations with Diffusers and comes with **Dreambooth and LoRA training scripts** [blog post](https://huggingface.co/blog/sd3#summary-of-memory-optimizations).

- **Common issues while running Diffusers scripts**: Multiple users ran into issues running Diffusers scripts for **SD3** with errors related to tokenizer and environment setup. Installing `sentencepiece` and ensuring proper paths and dependencies were suggested fixes.

- **Training SD3 LoRA with single GPU**: Users discussed the possibility of training **SD3 LoRA** using a single high-end GPU like the **RTX 4090**. Recommendations included adjusting batch sizes, using fp16 precision, and validating paths are absolute and properly formatted.

- **Troubleshooting GPU configurations**: Users with issues running scripts on **Windows** and **Linux** shared solutions like ensuring correct NVIDIA driver installations and **accelerate** configurations. It was suggested to use **Hugging Face model hub** to manually download models and check dependencies.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/sd3#summary-of-memory-optimizations>">Diffusers welcomes Stable Diffusion 3</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main">stabilityai/stable-diffusion-3-medium-diffusers at main</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers#using-with-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sd3.md#lora--dreambooth">diffusers/examples/dreambooth/README_sd3.md at main · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1250531973177413662)** (223 messages🔥🔥): 

- **AI Pathways Explored with Potential Guidance**: A user inquired about the pathways to become an AI engineer and it was suggested to look into resources provided by **Karpathy**, a former cofounder of OpenAI and Director of AI at Tesla. Another user linked the [YouTube series by sentdex](https://www.youtube.com/watch?v=GZqYr8_Q7DE) for creating a chatbot with deep learning.
- **Speculation on GPT-4.5 Turbo Model**: There was a heated discussion about a temporarily visible page suggesting OpenAI's release of **GPT-4.5 Turbo** with a 256k context window and a knowledge cutoff of June 2024. Some users believed it could have been a test page or a mistake while others speculated about continuous pretraining capabilities.
- **Memory Limits and Management in ChatGPT**: Several users discussed issues related to memory limits in ChatGPT. One member recommended checking and collectively summarizing memory to free up space as a workaround.
- **Team Account Benefits and Conditions**: There were clarifications about the benefits and restrictions of ChatGPT Team accounts, including doubled prompt limits and billing policies. A user shared that the ChatGPT Team plan requires a minimum commitment of two seats and has a higher monthly cost if not billed annually.
- **New UI and Behavioral Changes in ChatGPT**: Users observed new UI changes and more expressive text responses in the latest ChatGPT updates, sparking interest and speculation about improvements in the model's behavior.


**Link mentioned**: <a href="https://arxiv.org/abs/1903.00161">DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs</a>: Reading comprehension has recently seen rapid progress, with systems matching humans on the most popular datasets for the task. However, a large body of work has highlighted the brittleness of these s...

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1250684685957464135)** (3 messages): 

- **Clarification on model names**: The model known as **GPT-4 Turbo** is referenced with the identifier `"gpt-4-turbo"` in the OpenAI API. Another member confirmed that the model based on the GPT-4 architecture is often referred to with identifiers like `"gpt-4"` or `"gpt-4-turbo"`.

- **Customizing GPT roles generates confusion**: A member expressed confusion over the customization of GPT roles with model names like `"gpt-4"` and `"gpt-4-turbo"`. 

- **Temporary messages in GPTs possible**: A user inquired whether GPTs can have temporary messages. The same user confirmed that it is possible, expressing excitement with *"it is! awesome"*.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1250796047815540807)** (3 messages): 

- **Chunk and Trim Large Text Data**: A member suggested that for managing "300MB of text data", you "chunk that up, and cut it down." This implies breaking down the data into smaller, manageable parts.
- **Practical Tips for Large Documents**: A link to a forum post offering [practical tips for dealing with large documents](https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2) was shared. This resource likely provides strategies to handle large text inputs within the constraints of token limits.
- **Handling Long Texts with Embeddings**: Another link to a [notebook on embedding long inputs](https://cookbook.openai.com/examples/embedding_long_inputs) demonstrates handling texts exceeding a model's maximum context length. The guide uses embeddings from `text-embedding-3-small` and refers to the [OpenAI Embeddings Guide](https://beta.openai.com/docs/guides/embeddings) for further learning.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2">Practical Tips for Dealing with Large Documents (&gt;2048 tokens)</a>: Dear Mr Plane,  Please correspond with the OpenAI team and they will reply with your specific parameters.  Kind Regards, Robinson</li><li><a href="https://cookbook.openai.com/examples/embedding_long_inputs">Embedding texts that are longer than the model&#x27;s maximum context length | OpenAI Cookbook</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1250796047815540807)** (3 messages): 

- **Chunk your data for processing**: One member advised splitting large text data files, particularly files around **300MB**, to manage them better. They noted that smaller chunks are easier to handle.

- **Practical tips for large documents**: A link to a [community forum post](https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2) was shared, providing guidance on managing documents exceeding 2048 tokens in length.

- **OpenAI's embedding models limit**: A [resource on embedding long inputs](https://cookbook.openai.com/examples/embedding_long_inputs) was shared, explaining that OpenAI's embedding models have maximum text length limits measured by tokens. The post includes a notebook that demonstrates handling over-length texts and links to the [OpenAI Embeddings Guide](https://beta.openai.com/docs/guides/embeddings).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2">Practical Tips for Dealing with Large Documents (&gt;2048 tokens)</a>: Dear Mr Plane,  Please correspond with the OpenAI team and they will reply with your specific parameters.  Kind Regards, Robinson</li><li><a href="https://cookbook.openai.com/examples/embedding_long_inputs">Embedding texts that are longer than the model&#x27;s maximum context length | OpenAI Cookbook</a>: no description found
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1250525063736721438)** (15 messages🔥): 

- **Finding example data for LLM finetuning**: A user inquired about sourcing examples for finetuning when lacking a large dataset. No specific solutions were discussed in the responses.

- **Recordings for June sessions**: It was clarified that there wouldn't be a single recording for June's sessions, but each event has its own recording.

- **Memorization experiment reveals interesting results**: A user shared results from an experiment on LLM memorization, showing significant improvement with 10x example sets but little increase with higher multiples. Detailed findings and a [GitHub repo with datasets and scripts](https://github.com/petergpt/Fine-Tuning-Memorisation-Experiement-GPT-35) were shared for further exploration.

- **Call for LLM experts to join the team**: An open position was posted looking for an expert in developing and fine-tuning LLMs to help create a model for Hugging Face's leaderboard, emphasizing the need for innovative evaluation metrics. Interested candidates were invited to submit resumes and cover letters for consideration.

- **Morale-boosting reminder with a new channel announcement**: In response to a humorous complaint about a member enjoying free time, a new channel, <#1250895186616123534>, was announced for showcasing and demoing projects. Users were encouraged to use the General voice channel for live demonstrations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/hamelhusain/status/1801309935512469938?s=46&t=6mnB9S1kF_kR1cspkH1ayQ">Tweet from Hamel Husain (@HamelHusain)</a>: Its so nice to have free time!</li><li><a href="https://x.com/hamelhusain/status/1801309935512469938?s=46&t=6mnB">Tweet from Hamel Husain (@HamelHusain)</a>: Its so nice to have free time!</li><li><a href="https://docs.google.com/spreadsheets/d/1Sjj5N7J7AFotEeVGL7cHqbHIy2ET8XG3xni2nG5yAGQ/edit?usp=sharing">Fine-Tuning-Memorisation-v2</a>: Results-all-v2  Examples,Run 1 or 2,Object,No,Question,Expected Answer,Response,Extracted Number,Correct,Temperature 1-50-Examples,One,Apple,Q1,What is the number for Apple?,65451,The number for Apple...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1250564026312691732)** (4 messages): 

- **Trouble with text-to-SQL template in finetuning llama3**: An issue was raised about finetuning llama3 for text-to-SQL using the `llama-3.yml` config and `sqlqa.subsample.jsonl` dataset. The preprocessing stage did not format the data correctly, resulting in `<|begin_of_text|>` tags instead of `[INST]` and `[SQL]` markers.
- **Extra credit confusion**: Several members discussed missing out on extra credits due to the deadline on June 11th. One member expressed the complexity of keeping up with many channels and appreciated the ease of starting an app with Modal.
- **Modal credits gratitude**: Multiple members thanked for receiving $1000 credits with one sharing their email for not receiving $500 extra credits despite running the starter app before the deadline.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1250812787740114974)** (1 messages): 

- **Seeking resources for LLM input & output validations**: A user asked for recommendations on resources to write input & output validations for LLMs. They also requested suggestions for favorite tools in this context.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1250853288677478430)** (2 messages): 

- **Trouble Uploading SQLite DB to Huggingface Spaces**: A user struggled with uploading a 3MB SQLite file to Huggingface Spaces, initially unable to update it due to file size restrictions. The solution they discovered was to **use git lfs** to manage the file upload.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1250585502814306334)** (2 messages): 

- **LangSmith Free Developer Plan Info**: Users on the free Developer plan have a "$0/seat cost and 5K free traces a month." *You won't be charged for anything until your credits are exhausted.*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/)** (1 messages): 

hamelh: humanloop, promptfoo are both popular
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1250756995661955165)** (4 messages): 

- **Victory in building a working system:** A member expressed relief and satisfaction after successfully building a working system, mentioning it took several hours to accomplish.
- **Library issues with chunk separation:** Another message raised an issue about the **fasthtml library** not getting the correct answers due to separation in chunks, hinting at possible performance impacts.
- **Seeking examples of Colbert embeddings:** A member inquired about good examples showing how **Colbert embeddings** are used for retrieval, suggesting a need for clarity or learning resources.
- **Deployed app to Hugging Face Spaces:** The [deployed app](https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine) includes a feedback mechanism and is slow but functional. The app took longer to deploy than build, covers 20-21 talks, and stores feedback in a SQLite DB.

**Link mentioned**: <a href="https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine">RizzConn Answering Machine - a Hugging Face Space by t0mkaka</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1250554849825788107)** (2 messages): 

- **Evaluating Category Selection Approaches**: A member discussed challenges in selecting from 500 categories when filtering documents. They mentioned testing this approach with GPT-3.5 and 4 without success and are considering revisiting it with GPT-4.0 for different product feeds to explore potential improvements.

- **ParadeDB Installation Highlight**: A message recommended installing ParadeDB extensions, emphasizing its ability to *"unify operational and analytical data"* and *"unlock insights faster and simplify the data stack with an all-in-one search and analytical database."* ParadeDB's **ACID-compliant** transaction control and **Postgres compatibility** were also mentioned as key benefits.

**Link mentioned**: <a href="https://www.paradedb.com/">ParadeDB - Postgres for Search and Analytics</a>: ParadeDB is a modern Elasticsearch alternative built on Postgres.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[saroufimxu_slaying_ooms](https://discord.com/channels/1238365980128706560/1242224552415596554/1250674841778917396)** (7 messages): 

- **Member Requests Training Profiling Lecture**: A member asked if experts could give a lecture on **training profiling**, covering tools, setup, and log interpretation. They emphasized the importance of maximizing GPU utilization and reducing training costs, noting a *"BIG gap in speed between frameworks."*

- **Poll for Lecture Demand**: A suggestion was made to gauge interest through a poll. Another member received superpowers to create the poll to measure demand.

- **Potential Speaker Shows Interest**: The potential speaker preliminarily agreed to give the lecture if there was enough interest. They also humorously agreed to create a private lesson if community interest was lacking.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/1250548667149058059)** (4 messages): 

- **Gemini models differ in approach**: One user sought clarification on whether the Gemini.Google.com model is a mixture of a RAG/search model, while the Gemini 1.5 Pro is solely a language model. They were trying to reconcile differing outputs to the same prompt across these models.

- **Detailed description of the Adolescent Well-being Framework**: The user shared a comprehensive breakdown of the Adolescent Wellbeing Framework (AWF) from gemini.google.com, emphasizing its five interconnected domains such as mental and physical health and community and connectedness. The framework, developed by the WHO, aims to assess and promote well-being in adolescents. 

- **Comparing frameworks for adolescent well-being**: Another response from aistudio.google.com with Gemini 1.5 Pro provided a broader view, highlighting several well-known models like the Five Cs of Positive Youth Development and the Whole School, Whole Community, Whole Child (WSCC) Model. This alternative response presented various dimensions such as physical health, mental and emotional health, and economic well-being, noting these are crucial for adolescent development.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1250881489286070342)** (2 messages): 

- **RAG-based app launched on Huggingface Spaces**: A member announced that they built a **RAG-based app** with **Gradio** and uploaded it to Huggingface Spaces. [RizzCon-Answering-Machine](https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine) is presented as "Refreshing," processing data for the first 20 talks.
  
- **Performance needs improvement for RAG-based app**: The same member noted that their RAG-based app is presently too slow and needs some profiling for better performance.

**Link mentioned**: <a href="https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine">RizzConn Answering Machine - a Hugging Face Space by t0mkaka</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1250581077529989181)** (9 messages🔥): 

- **Llama-3-8b preprocessing error plagues user**: A member reported encountering a warning while preprocessing llama-3-8b, stating: *"copying from a non-meta parameter in the checkpoint to a meta parameter in the current model, which is a no-op"*. This error seems to repeat for each layer, also appearing with the mistral-7b model.
- **VSCode disconnects hinder fine-tuning tasks**: A member experienced frequent disconnections when running fine-tunes on a DL rig via SSH in VSCode. Another member suggested using `nohup` or switching to manual execution on the physical machine, with additional advice to consider using SLURM for better handling.
- **Nohup to the rescue for background processes**: Members discussed using `nohup` to run commands in the background, allowing processes to continue even if the SSH session disconnects. A helpful link to a [SuperUser thread](https://superuser.com/questions/448445/run-bash-script-in-background-and-exit-terminal) was shared for detailed instructions.

**Link mentioned**: <a href="https://superuser.com/questions/448445/run-bash-script-in-background-and-exit-terminal">Run Bash script in background and exit terminal</a>: Is it possible to launch a command or Bash script exit terminal and NOT interrupt command?&#xA;&#xA;My solution was to run cron at a specific time of day, but I&#x27;m sure there is something easier.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1250885608520286389)** (1 messages): 

- **Seamlessly switch between FSDP and DeepSpeed**: An article titled "A Hugging Face Accelerate Story of Multiple Backends: FSDP and DeepSpeed" was shared. It discusses how to go back and forth from FSDP to DeepSpeed using Hugging Face [Accelerate](https://huggingface.co/docs/accelerate/en/index), highlighting the differences between these backends and a precision-related change that was upstreamed. [Read more here](https://huggingface.co/blog/deepspeed-to-fsdp-and-back).



**Link mentioned**: <a href="https://huggingface.co/blog/deepspeed-to-fsdp-and-back">From DeepSpeed to FSDP and Back Again with Hugging Face Accelerate</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1250568402360078466)** (3 messages): 

- **Axolotl struggles on Modal but hope for success on Jarvis**: A user shared that they had difficulty running Axolotl on Modal despite its supposed ease. They plan to try it on Jarvis next.

- **Helpful YouTube tutorial for Axolotl on JarvisLabs**: Another user provided a [YouTube tutorial](https://www.youtube.com/watch?v=Y9464wasHuE&ab_channel=JarvisLabsAI) which they found useful for running Axolotl on JarvisLabs. The video includes links to both JarvisLabs and the Axolotl GitHub repository.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=Y9464wasHuE&ab_channel=JarvisLabsAI">How to run axolotl on JarvisLabs | Tutorial</a>: Check out axolotl on JarvisLabs : jarvislabs.ai/templates/axolotlCheck out axolotl github : https://github.com/OpenAccess-AI-Collective/axolotl

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/1250761957129191466)** (1 messages): 

- **Seeking local model access without Ollama**: A user inquired about accessing a local model running on a server with an API call instead of using Ollama, specifically mentioning **TGI** or **vllm**. The user referenced possible blogs that discuss this.

- **Interest in DeepSeek V2 API support**: The user asked if there are plans to support the **DeepSeek V2 API**. This query indicates a keen interest in enhanced API functionalities.

- **Autocomplete in command line with FIM models**: The user expressed a desire to use **LLM CMD with FIM models** to get inline autocomplete suggestions similar to Copilot but within the command line. They suggest it could run once and switch to a **CMD FIM mode** or simply handle completions with a prefix.

- **Function calling support in LLM**: There was a question about whether **LLM will support function calling** when the model produces a function call token. This indicates the user's interest in advanced features that handle code execution.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1250569058261274706)** (4 messages): 

- **Requests for credits support flood in**: Multiple users are seeking assistance with their credits after filling out the form. Users provided their account IDs such as *anopska-552142*, *as-ankursingh3-1-817d86*, *standonopenstds-ff2f17*, and *apratim941208-cc11fd* but haven’t received their credits yet.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[emmanuel_finetuning_dead](https://discord.com/channels/1238365980128706560/1245129595749925086/)** (1 messages): 

gitmaxd: great question
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1250531245310349352)** (3 messages): 

- **Greetings in the Europe TZ channel**: A member greeted the channel with "Salutare" and later switched to English, inquiring if anyone was interested in **fine-tuning a model** with **Euro24 news** and related data such as player/match stats. Another member responded with a friendly "Hello from Mainz, Germany!"
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1250525685022457916)** (5 messages): 

- **Office Hours Appreciation**: "Thanks everyone for coming to the office hours today! Lots of great questions, really appreciate everyone for taking the time!" This acknowledgment highlights the community's engagement and participation in a productive Q&A session.

- **DJL Topic from AWS Blog Post**: A member raised questions during office hours about "DJL" from an AWS blog post titled "Efficient and Cost-Effective Multi-Tenant LoRA Serving with Amazon SageMaker." The blog post discusses the benefits and use cases of generative AI models, mentioning [BloombergGPT](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) and technologies like vLLM and an "optimized rolling batch scheduler."

**Link mentioned**: <a href="https://aws.amazon.com/blogs/machine-learning/efficient-and-cost-effective-multi-tenant-lora-serving-with-amazon-sagemaker/">Efficient and cost-effective multi-tenant LoRA serving with Amazon SageMaker | Amazon Web Services</a>: In this post, we explore a solution that addresses these challenges head-on using LoRA serving with Amazon SageMaker. By using the new performance optimizations of LoRA techniques in SageMaker large m...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1250760637823451156)** (3 messages): 

- **Query about openpipe contact and credits**: A member asked if anyone knew who to contact for openpipe as they hadn't received their credits yet.
- **Follow-up on email receipt inquiry**: Another member asked if the first member had received the email regarding their credits.
- **Update on second round of credits**: It was mentioned that a second round of credits would be granted on the 14th, which is tomorrow.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1250547864334241802)** (40 messages🔥): 

- **Fine-Tuning Can Add New Knowledge**: A lively discussion centered on fine-tuning models to add new knowledge, with multiple members emphasizing that this process can be costly but effective. Examples like "Nous Hermes" were cited to illustrate successful multi-skill incorporation through fine-tuning.

- **GPT-Generated Data and Legal Limitations**: Questions about the usage of GPT-generated data for training within enterprises sparked debate. The discussion referenced OpenAI's business terms, which restrict using output to develop competing AI models, providing [a link to the policy](https://openai.com/policies/business-terms/).

- **BLEU Score's Relevance for Grammar**: There was a query about the appropriateness of using BLEU scores to test grammar, with some members expressing doubt and criticism about its effectiveness.

- **Generating Synthetic Data for Fine-Tuning**: The community was curious about the feasibility and legality of using synthetic data for fine-tuning AI models for commercial use. A link to a [paper on ToolkenGPT](https://arxiv.org/abs/2305.11554) was shared, discussing innovative approaches to combining tool demonstrations with in-context learning.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2305.11554">ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings</a>: Augmenting large language models (LLMs) with external tools has emerged as a promising approach to solving complex problems. However, traditional methods, which finetune LLMs with tool demonstration d...</li><li><a href="https://cookbook.openai.com/">OpenAI Cookbook</a>: Open-source examples and guides for building with the OpenAI API. Browse a collection of snippets, advanced techniques and walkthroughs. Share your own examples and guides.</li><li><a href="https://cookbook.openai.com/examples/fine-tuned_qa/olympics-1-collect-data">Fine-Tuned Q&amp;A - collect data | OpenAI Cookbook</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[pawel-function-calling](https://discord.com/channels/1238365980128706560/1250550872312643594/1250563483372617730)** (92 messages🔥🔥): 

- **Fireworks AI logo gets love**: One member expressed their admiration for the **Fireworks AI logo**. Another member shared a curiosity about function calls and how they could be used for stream evaluations and hallucination management tools.
- **Delta in model weights sparks interest**: There was a discussion about the concept of "delta", described as the difference between instruct and base weights in model fine-tuning. One member encouraged others to try model merging, calling it "fun".
- **Axolotl special tokens configuration**: A member asked about adding special tokens in Axolotl for function calls, and others confirmed it can be done through configuration. They also discussed using the "template-free format" to avoid issues with chat prompt templates.
- **LangGraph and LangChain debated**: Various members shared their experiences with **LangChain** and **LangGraph**, with mixed feelings about their usability. Some praised LangChain's extensive ecosystem and support, while others preferred LangGraph's customization and independence from LangChain.
- **Glaive Function Calling model introduction**: Shared a link to [glaive-function-calling-v1](https://huggingface.co/glaiveai/glaive-function-calling-v1), a 2.7B parameter open-source chat model capable of multi-turn conversations and intelligent function execution, based on the Replit-code-v1-3b model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg">LangGraph (Python)</a>: This video series covers how to use code functionality of LangGraph, as well as common modifications one could want to make.</li><li><a href="https://x.com/gitmaxd/status/1800234864329068708?s=46&t=QitgwfFVpCSQgUY0DIcTdA">Tweet from Git Maxd (@GitMaxd)</a>: LangGraph Learning Order:  1. LangGraph YouTube Learning Series: https://www.youtube.com/watch?v=5h-JBkySK34&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg  2. LangGraph Self-correcting code assistants with ...</li><li><a href="https://huggingface.co/glaiveai/glaive-function-calling-v1">glaiveai/glaive-function-calling-v1 · Hugging Face</a>: no description found</li><li><a href="https://discord.gg/scEHnRaz">Join the Fireworks.ai Discord Server!</a>: Check out the Fireworks.ai community on Discord - hang out with 2148 other members and enjoy free voice and text chat.</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - Template-free prompt construction</a>: no description found</li><li><a href="https://huggingface.co/spaces/arcee-ai/mergekit-gui">mergekit-gui - a Hugging Face Space by arcee-ai</a>: no description found</li><li><a href="https://nbsanity.com/static/d06085f1dacae8c9de9402f2d7428de2/demo.html">Llama-3 Function Calling Demo</a>: no description found</li><li><a href="https://clip.cafe/commando-1985/until-the-next-time/">'This was the last time. Until a next time. No chance.' - Commando</a>: [Matrix declines an offer to resume his unit]  John Matrix: This was the last time.  Major General Franklin Kirby: Until a next time.  [pause]  John Matrix: No chance.
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1250631155364134994)** (5 messages): 

- **LLM-driven objective discovery pushes the boundaries of preference optimization**: A [paper on arXiv](https://arxiv.org/abs/2406.08414) explores offline preference optimization for LLMs using LLM-driven objective discovery to automatically find new optimization algorithms. This approach allows for the discovery of preference optimization algorithms without expert human intervention by iteratively prompting an LLM based on performance metrics.
  
- **Mixture-of-Agents (MoA) methodology harnesses multiple LLMs' collective expertise**: Another paper on [Hugging Face](https://huggingface.co/papers/2406.04692) proposes using a Mixture-of-Agents architecture, where multiple LLM agents in layered configurations collaborate to enhance performance on various benchmarks. The MoA model outperforms GPT-4 Omni, achieving a remarkable score of 65.1% on AlpacaEval 2.0 compared to GPT-4 Omni's 57.5%.

- **GitHub repository for MoA methodology**: The implementation of the Mixture-of-Agents model can be found on [GitHub](https://github.com/togethercomputer/moa). Users can contribute to and explore the development of this promising approach to leveraging multiple LLMs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.08414">Discovering Preference Optimization Algorithms with and for Large Language Models</a>: Offline preference optimization is a key method for enhancing and controlling the quality of Large Language Model (LLM) outputs. Typically, preference optimization is approached as an offline supervis...</li><li><a href="https://github.com/togethercomputer/moa">GitHub - togethercomputer/MoA</a>: Contribute to togethercomputer/MoA development by creating an account on GitHub.</li><li><a href="https://huggingface.co/papers/2406.04692">Paper page - Mixture-of-Agents Enhances Large Language Model Capabilities</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1250537134767472810)** (152 messages🔥🔥): 

- **Stable Diffusion 3 Early Reviews**: [Stable Diffusion 3](https://x.com/minimaxir/status/1800921802765717754) is receiving mixed feedback, with some users calling it impressive and others highlighting initial issues and bugs. One user mentioned that *"it uses an LLM as an encoder,"* suggesting a different approach from traditional models.

- **Debate over GPT-4 Temperature Settings**: A tweet revealed that GPT-4 performs better at temperature=1 than temperature=0, even on deterministic tasks, sparking surprise and debate. Users discussed how this doesn't apply to fine-tuned Llama 3 models, with varying results for different tasks.

- **Uncensored Model Release**: A user shared their [uncensored version of the OpenHermes-2.5 dataset](https://huggingface.co/datasets/Replete-AI/OpenHermes-2.5-Uncensored), removing 2,697 censored lines. There was a discussion on the impact of removing alignment constraints and its effect on model responses.

- **MatMul-Free LLMs**: A new paper on [eliminating MatMul operations](https://arxiv.org/abs/2406.02528) in large language models was shared, boasting significant memory savings and strong performance at billion-parameter scales. The paper claims that MatMul-free models can match the performance of state-of-the-art Transformers while reducing memory usage by up to 61%.

- **Discussion on AI Model Jailbreaking**: [Haize Labs](https://x.com/haizelabs/status/1800936720990384174) announced a method to automatically jailbreak AI models, uncovering safety violations in top AI systems by eliciting harmful content. This sparked discussions on the ethics and consequences of such actions in the AI community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.02528">Scalable MatMul-free Language Modeling</a>: Matrix multiplication (MatMul) typically dominates the overall computational cost of large language models (LLMs). This cost only grows as LLMs scale to larger embedding dimensions and context lengths...</li><li><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium">Stable Diffusion 3 Medium - a Hugging Face Space by stabilityai</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Cm7IlwoVT4w">Testing Neuro&#39;s Morality with Slay the Princess (Part 1)</a>: Vedal attempts to make Neuro-sama play Slay the Princess.►Twitch: http://www.twitch.tv/vedal987►Twitter: https://twitter.com/Vedal987Edited by Paradomix</li><li><a href="https://x.com/corbtt/status/1801026166020833457">Tweet from Kyle Corbitt (@corbtt)</a>: Crazy fact that everyone deploying LLMs should know—GPT-4 is &#34;smarter&#34; at temperature=1 than temperature=0, even on deterministic tasks.  I honestly didn&#39;t believe this myself until I trie...</li><li><a href="https://x.com/haizelabs/status/1800936720990384174">Tweet from Haize Labs (@haizelabs)</a>: Today is a bad, bad day to be a language model.  Today, we announce the Haize Labs manifesto.  @haizelabs haizes (automatically red-teams) AI systems to preemptively discover and eliminate any failure...</li><li><a href="https://x.com/jeremyphoward/status/1801037736968913128">Tweet from Jeremy Howard (@jeremyphoward)</a>: New paper just dropped, showing how to greatly increase math scores on LLMs by combining monte-carlo tree search (MCTS) with a language model.  Nice! But... what if instead, we simply tell the LLM to ...</li><li><a href="https://x.com/minimaxir/status/1800921802765717754">Tweet from Max Woolf (@minimaxir)</a>: Taking a look at people testing out Stable Diffusion 3 and tbh this goes hard.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1df0kau/sd3_has_been_liberated_internally_pure_text2img/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1df0kau/sd3_has_been_liberated_internally_pure_tex">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/corbtt/status/1801059164954775643">Tweet from Kyle Corbitt (@corbtt)</a>: @eugeneyan @aidan_mclau ok well now I&#39;m seeing wildly different results with my fine-tuned llama 3 models (that match my prior intuitions -- higher temp outperforms on creative tasks, lower temp o...</li><li><a href="https://huggingface.co/datasets/Replete-AI/OpenHermes-2.5-Uncensored">Replete-AI/OpenHermes-2.5-Uncensored · Datasets at Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2406.05955">Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters</a>: Exploiting activation sparsity is a promising approach to significantly accelerating the inference process of large language models (LLMs) without compromising performance. However, activation sparsit...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1250639839939002421)** (2 messages): 

- **Lost paper on interleaving pretraining and instructions**: A member asked for help remembering **a recent paper** that interleaved pretraining documents and extracted instructions during finetuning. Another member expressed interest in being notified if the paper is found.
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1250760508274118707)** (15 messages🔥): 

- **No final RAG dataset schema yet**: Queries about the final dataset schema for RAG reveal that it isn't yet settled. "Doesn't seem like it" and discussions on finalizing datagen pipeline confirm ongoing development.
  
- **Embedding text for better verification**: Suggestions were made to link or show embedded texts to avoid hallucinations. One member noted, *"you could do a ctrl+f on the embeddings."*

- **Marker used for PDF to markdown conversion**: It's confirmed that the raw text will be markdown as PDFs are being converted via [Marker](https://github.com/VikParuchuri/marker). This tool is praised for its accuracy but noted to be slow for inference time conversion.

- **Pandoc and make4ht for other document conversions**: Another method discussed for converting various document types includes using [Pandoc](https://www.pandoc.org) and make4ht for LaTeX files. These tools were suggested as alternatives for different formats.

- **Speed optimization recommendations**: Speed improvements for Marker were suggested, such as setting min_length to avoid unwanted OCR. A member claimed about 1 page per second per worker on an A10 GPU, highlighting the potential for parallel processing.

**Link mentioned**: <a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>: Convert PDF to markdown quickly with high accuracy - VikParuchuri/marker

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1250560067032449074)** (6 messages): 

- **Open-source status remains closed for now**: When asked if the project is open-source, it was confirmed, *"it's not currently open-source."* There are no immediate plans to change this status, although it’s *"talked about"* and might be reconsidered as development continues.
- **Request for world-sim AI enhancements**: A user suggested making the **world-sim AI edgier** and also proposed developing a **free mobile port** for the application. They expressed belief that a mobile version would be *"cool."*
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1250530296504258685)** (136 messages🔥🔥): 

- **Love for the New Search Feature**: A user expressed excitement about the new search feature, saying, *"Loving this new search,"* and immediately requested an iOS version, *"Need it in iOS now pls."*

- **Concerns on Perplexity's Website Functionality**: One member asked if others were experiencing view issues on Perplexity’s website where the chat jumps halfway up the page after each answer.

- **Enterprise Support Woes**: A user with an enterprise plan expressed frustration over support issues, mentioning that they were *"dumped in the ticket system and never got a response"* despite emailing two weeks ago. Suggestions included contacting Alex or checking out the [support page](https://www.perplexity.ai/settings/account).

- **Discussions on API Limitations and Solutions**: Members discussed the limitations of the Perplexity API, with one suggesting breaking down complex requests and another recommending using smaller models to handle tasks. An alternative API product in development was also mentioned by a member claiming their product might offer better scalability and features.

- **Frustrations with Rabbit Inc.**: In a lengthy discussion, a user shared their negative experience with Rabbit Inc., criticizing their support and privacy policies, and linking to a [Reddit post](https://www.reddit.com/r/rabbitinc/comments/1desu27/rabbit_inc_will_not_allow_you_to_participate_in/) that details the issue. Another member noted that Perplexity does not require such proof to participate in discussions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/rabbitinc/comments/1desu27/rabbit_inc_will_not_allow_you_to_participate_in/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.perplexity.ai/settings/account">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1250554338917744701)** (8 messages🔥): 

- **Elon Musk drops lawsuit against OpenAI**: Elon Musk has withdrawn his lawsuit against OpenAI and its executives just a day before the court hearing. The lawsuit claimed that OpenAI shifted from its mission to a profit-driven entity, prioritizing Microsoft, its largest investor, over humanitarian goals [source](https://www.pymnts.com/artificial-intelligence-2/2024/elon-musk-drops-lawsuit-against-openai-one-day-before-hearing/) [source](https://www.cnbc.com/2024/06/11/elon-musk-drops-suit-against-openai-and-sam-altman.html).

- **First wooden satellite set to launch in 2024**: Researchers from Kyoto University and Sumitomo Forestry unveiled LignoSat, the world's first wooden satellite, which is expected to launch in September 2024. This project aims to evaluate wood as a sustainable material for space and reduce environmental impacts [source](https://www.japantimes.co.jp/news/2024/05/29/japan/science-health/world-first-wooden-satellite/) [source](https://www.space.com/japan-september-launch-first-wooden-satellite). 

- **Email carbon footprint and environmental impact**: The average carbon footprint of an email is around 4 grams of CO2, with significant increase when emails contain large attachments. Using file share links instead of attachments can reduce CO2 emissions [source](https://www.mailjet.com/blog/email-best-practices/email-carbon-footprint/) [source](https://blog.rwth-aachen.de/itc/en/2023/12/18/co2-auswirkungen/). 

- **Origin of the idiom "rock bottom"**: The phrase "hit rock bottom" refers to reaching the lowest possible level and has been metaphorically used since the mid-19th century. It originally described the experience of miners hitting the solid rock layer beneath the soil [source](https://dictionary.langeek.co/en/word/212505?entry=hit+rock+bottom) [source](https://english.stackexchange.com/questions/597487/what-is-the-origin-of-the-phrase-hit-rock-bottom).

- **Perplexity.ai's fast results with LLMs**: Perplexity.ai leverages state-of-the-art hardware like AWS p4d instances with NVIDIA A100 GPUs and software optimizations such as NVIDIA's TensorRT-LLM to achieve fast results despite using large language models. Integration with AWS and Kubernetes facilitates elastic scaling, reducing downtime and network overhead [source](https://www.perplexity.ai/hub/blog/introducing-pplx-api).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/eC882F-jaMw">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/where-does-the-fpYkPFAuS.Gbido9PnphDQ">where does the idiom &quot;rock bottom&quot; come from?</a>: The idiom &quot;hit rock bottom&quot; originated from the literal meaning of reaching the solid bedrock or bottom layer of rock when digging or mining. It has been used...</li><li><a href="https://www.perplexity.ai/page/Worlds-First-Wooden-j0Q0FI3MS6OKEDTPzSN7Fw">World&#x27;s First Wooden Satellite</a>: In a groundbreaking development, researchers from Kyoto University and Sumitomo Forestry have unveiled LignoSat, the world&#x27;s first wooden satellite, set to...</li><li><a href="https://www.perplexity.ai/page/Musk-drops-lawsuit-LLG_ToBhQ2..DzJ3e1RKJQ">Musk Drops Lawsuit Against OpenAI</a>: Elon Musk, co-founder of OpenAI, has withdrawn his lawsuit against the company and its executives, Sam Altman and Greg Brockman, just one day before a...</li><li><a href="https://www.perplexity.ai/page/Emails-Carbon-Footprint-m0T5zvUQQkC.G2wIQV0GSg">Email&#x27;s Carbon Footprint</a>: Studies show that the average carbon footprint of an email is around 4 grams of CO2 equivalent, with the footprint increasing significantly if the email...</li><li><a href="https://www.perplexity.ai/search/how-does-perplexityai-JQsCcEwOQSyyMFTrGbw43g">how does perplexity.ai fetch fast results despite using LLMs which are slow</a>: Perplexity.ai achieves fast results despite using large language models (LLMs) through a combination of state-of-the-art software and hardware. The key...</li><li><a href="https://www.perplexity.ai/search/what-do-you-dt6P7CulQAWGTwBa2.zbzA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1250598120169934848)** (14 messages🔥): 

- **Connection Issues with Custom GPTs**: A member reported issues with Custom GPTs saying *"it couldn't connect"* when trying to perform function calls on [chat.openai.com](https://chat.openai.com/). Another member suggested checking if the PPLX API key is added in the Auth panel when creating the GPT.
- **Desktop App vs. Web Issues**: It was noted that Custom GPTs were not functioning on the web version of chat.openai.com, but they worked normally on the desktop app. This implies there might be a platform-specific issue affecting API calls.
- **Still No Resolution**: Despite confirming the correct use of the API key and trying recommended solutions, a member continued to face the error with no response when using Custom GPTs. Another member indicated that their setup under the same conditions worked fine, suggesting the problem might be isolated.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1250856013259870301)** (3 messages): 

- **Question on Compute Intensity Metrics**: A member asked whether, in practice, only **floating point operations** done on data accessed from **Global Memory** are considered for compute intensity (operations per byte accessed), as opposed to operations on data that exists solely in registers. They followed up with a self-referential note, humorously admitting they had typed the query a week prior but hadn't sent it.
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1250854733481246801)** (2 messages): 

- **Install Triton 3.0 with ease**: A user shared a hassle-free guide on [installing Triton 3.0 from source](https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/). This involves uninstalling the current Triton version, cloning the repository, installing dependencies, and running the setup script.

- **Alternative Triton installation from PyTorch**: Another member suggested a method involving [cloning the PyTorch repository](https://github.com/pytorch/pytorch) and using the command `make triton`. The specific version can be found in the [triton_version.txt file](https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/">Installing Triton 3.0.0</a>: As of June 13 2024, to get Triton 3.0 you have to install it from source, like so:</li><li><a href="https://github.com/pytorch/pytorch">GitHub - pytorch/pytorch: Tensors and Dynamic neural networks in Python with strong GPU acceleration</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt">pytorch/.ci/docker/triton_version.txt at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1250629580742594581)** (13 messages🔥): 

- **Curiosity about fast 8-bit optimizer with PyTorch**: A member asked if it’s possible to create a fast 8-bit optimizer using only **pure PyTorch** and **torch.compile**. They highlighted the importance of creating lookup tables dynamically and performing calculations in FP32 as seen in the [bitsandbytes implementation](https://arxiv.org/pdf/2110.02861).

- **Drop-in replacement concerns**: Another member discussed the potential challenges of making a "drop-in replacement" for 8-bit optimizers, given possible accuracy deviations from 32-bit calculations. However, they noted that in practice, they observed no accuracy drop when using 8-bit AdamW from bitsandbytes.

- **Pure PyTorch + Triton version of 8-bit optimizers**: A member working on such an implementation proposed a pure `PyTorch` + `Triton` version of the `bitsandbytes` 8-bit optimizers. The goal is to avoid using custom CUDA kernels.

- **Long-term roadmap for bitsandbytes**: Another contributor acknowledged the possibility of integrating bitsandbytes with `torch.compile` for better compatibility. They pointed out the crucial aspects of the implementation, like quantization maps and ensuring operations are performed in FP32.

**Link mentioned**: <a href="https://arxiv.org/abs/1511.04561">8-Bit Approximations for Parallelism in Deep Learning</a>: The creation of practical deep learning data-products often requires parallelization across processors and computers to make deep learning feasible on large data sets, but bottlenecks in communication...

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1250617944174493857)** (2 messages): 

- **1-bit LLM promises rapid quantization**: A link to the [BiLLM GitHub project](https://github.com/Aaronhuang-778/BiLLM) was shared, which claims to quantize 7B LLMs in 0.5 hours with a single GPU. One member noted, "*seems like no fused kernel so no speedup (yet)*."
- **Upward trend in benchmark accuracy observed**: A member observed that benchmark accuracy appears to trend upwards. They seemed intrigued by this pattern, but no further details were discussed.

**Link mentioned**: <a href="https://github.com/Aaronhuang-778/BiLLM">GitHub - Aaronhuang-778/BiLLM: (ICML 2024) BiLLM: Pushing the Limit of Post-Training Quantization for LLMs</a>: (ICML 2024) BiLLM: Pushing the Limit of Post-Training Quantization for LLMs - Aaronhuang-778/BiLLM

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

_shivasinghbagri: https://powerinfer.ai/v2/
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1250737710792966214)** (31 messages🔥): 

- **Struggles with Speed-Up on 4090**: Despite setting influential flags and using multiple approaches like `torch.matmul` and `torch.nn.functional.linear`, a member couldn't achieve a good speed-up on an RTX 4090 GPU during INT8 and FP16 mixed matrix multiplications. They experimented with various configurations and observed slower performance.
  
- **FP8 and INT8 Matrix Multiplication**: Discussions revealed that FP8 x FP8 matmuls resulted in slower operations compared to INT8, due to slow casting processes. INT8 quantization seems to provide better speed and accuracy and is supported even on older GPUs like Pascal.
  
- **INT6 Quantization Shows Promise**: INT6, using a group size of 64, showed minimal accuracy degradation in tests with Llama3-8B-instruct models. Members noted that INT6 and INT4 quantizations have promising performance but require careful handling to maintain accuracy.

- **Exploring Mixed Precision with FP16/FP8**: Mixed precision operation FP16 x FP8 was discussed, highlighting the potential benefits and existing support in Microsoft's BitBLAS. Implementation and integration challenges were mentioned, particularly with the necessity to compile TVM.

- **Microsoft BitBLAS Implementation**: BitBLAS supports mixed precision matrix multiplications such as FP16 x FP8E4M3, but it relies heavily on TVM's Python interface. This requirement complicates the integration with other libraries, making it less accessible for some projects.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/f9c9c06c4478fbfbbf6986f21410ecac37d3f63e/test/test_matmul_cuda.py#L316-L325">pytorch/test/test_matmul_cuda.py at f9c9c06c4478fbfbbf6986f21410ecac37d3f63e · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://gist.github.com/mobicham/f95a09eaf2db48632f8bf571693c0884">torch_compile_mixed_mm_test.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/pytorch/ao/blob/6f44d259fabe1195669654e22f7f97fc028f89af/torchao/quantization/subclass.py#L371-L383">ao/torchao/quantization/subclass.py at 6f44d259fabe1195669654e22f7f97fc028f89af · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1250525436593705101)** (54 messages🔥): 


- **Keeping norm kernel calls in check**: A member mentions that they pushed a change to control the number of additional norm kernel calls and suggests a similar strategy for the adam kernel could be beneficial.

- **Exploring C++20 possibilities**: Discussion about achieving similar functionality using `std::span<const float *, N>`. A member points out it might require changes at the call site or utilizing a constructor already present.

- **Race condition fixes for loss infinities**: The loss infinities issue was identified as a race condition. The fix combined with removing the atomic add provides full determinism and slight performance improvement when profiled on 8xH100s.

- **Random data loading improvements**: PR for introducing randomness in data loading to ensure better shuffling and processing. Observations are made about how document boundaries and data batching could impact training performance, with a histogram of document lengths shared to provide insights.

- **Discussing batch and document similarity**: Debate on the impact of semantic similarity in document batching during training. Some members suggest it aids in in-context learning, while there's skepticism about its effect on generalization.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/573">Dataloader - introducing randomness by gordicaleksa · Pull Request #573 · karpathy/llm.c</a>: On the way to fully random train data shuffling... This PR does the following:  Each process has a different unique random seed Each process train data loader independently chooses its starting sha...</li><li><a href="https://github.com/karpathy/llm.c/pull/522">Add master weights to resume state by gordicaleksa · Pull Request #522 · karpathy/llm.c</a>: We&#39;re currently not saving master weights as part of the state -&gt; we lose some precision because otherwise when we resume we&#39;ll have to reconstruct the master weights by upcasting from lowe...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1250752018293985373)** (1 messages): 

- **AMD's MI300X outperforms NVIDIA's H100**: A recent [TensorWave blog post](https://www.blog.tensorwave.com/amds-mi300x-outperforms-nvidias-h100-for-llm-inference/) shares exciting benchmarks showing AMD’s new MI300X accelerator achieving 33% higher throughput for LLM inference compared to NVIDIA’s H100 SXM. This initial success is based on MK1’s inference software running vLLM on Mixtral 8x7B, indicating that despite a less mature software ecosystem, AMD’s hardware is a formidable competitor.

**Link mentioned**: <a href="https://www.blog.tensorwave.com/amds-mi300x-outperforms-nvidias-h100-for-llm-inference/">AMD’s MI300X Outperforms NVIDIA’s H100 for LLM Inference</a>: Discover if AMD&#039;s MI300X accelerator can outperform NVIDIA&#039;s H100 in real-world AI workloads. Early results are in!

  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1250549480164425828)** (13 messages🔥): 

- **Unresolved MX Format Test Causes Build Failure**: The build process is failing due to an *unrelated mx format test*. Attempts to retry the test were unsuccessful, leaving the issue unresolved.
- **CoffeeVampir3 Shares Testing File**: When asked about a specific file for testing Bitnet, a member provided a link to [bitnet_trained_to_ao_test.py](https://github.com/CoffeeVampir3/ao-bitnet/blob/main/bitnet_staging/bitnet_trained_to_ao_test.py).
- **Release Branch Deadline and Nightly Builds**: It was mentioned that any work to be included in the 0.3 release branch must be merged by next Tuesday. Alternatively, there is the option to continue experimenting with nightly builds.
- **Refactoring Uint2 for Bitnet**: Refactoring of Uint2 is in progress and is expected to be completed by tomorrow. This work is unblocking further developments now that bitpacking for Bitnet is operational.
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1250538845229813781)** (71 messages🔥🔥): 

- **Discussing LM Studio on Web Servers**: A member asked, *"can LM Studio run on a web server?"* Another member clarified, *"If it's a remote headless server, it won't work."*
- **Error in Loading Qwen-2 Model**: A user encountered an error with the Qwen-2 model on LMChat. The solution was to use the ChatML preset and enable Flash Attention, as advised by another user.
- **Model Recommendations for Translation**: Members discussed the best LLM models for translation. Qwen2 and Aya 23 were suggested, though they were noted as not perfect.
- **Concerns About LM Studio Developers**: A user expressed concerns over the anonymity of LM Studio's developers. It was clarified that the lead dev and founder can be found on GitHub.
- **Enabling Flash Attention through CLI**: A user sought to enable Flash Attention from the command line interface. It was noted that the LMS CLI lacks this feature, but llama.cpp as an alternative was suggested.

**Link mentioned**: <a href="https://github.com/andrewyng/translation-agent">GitHub - andrewyng/translation-agent</a>: Contribute to andrewyng/translation-agent development by creating an account on GitHub.

  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1250572233215184896)** (8 messages🔥): 

- **Gemini 1.5 flash struggles in JSON mode**: "Guys, a bit OT here but... has anyone tried gemini 1.5 flash? json mode fails badly and randomly, producing empty strings." A member is experiencing issues with Gemini 1.5 flash in JSON mode and is seeking feedback from others.
  
- **Tess 2.5 72b quant models released on HF**: "Tess 2.5 72b q3 and q4 quant gguf are live on HF." These new quant models are now available on Hugging Face.

- **VRAM limitations affect model choices**: "But with 8GB of VRAM you're not going to have a lot of options. For small models, perhaps Llama is a good place to start." A member advises another on VRAM limitations and suggests smaller models like Llama.

- **Model recommendation for grammar and syntax issues**: "The last time I tried to use a model to review my content for grammar and syntaxes problems," a member is seeking recommendations for a model that can accurately review grammar and syntax, especially recognizing dialogue.

- **LM Studio supports GGUF, not safetensor files**: "hi! does lmstudio support multipart safetensor files ?" "Safetensor no, only GGUF." A member confirms that LM Studio only supports GGUF format and not safetensor files.
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1250724815799713893)** (2 messages): 

- **Direct AVX2 Error Solution Offered**: A member pointed out that the error message encountered is due to lacking AVX2 instructions. This directs users to check their CPU's AVX2 support.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1250545180012970096)** (9 messages🔥): 

- **P40 Prices Spike Up**: The **electronic scrap P40** used to be around *$75 on flea markets in China*, but now it costs over $200. This surge contrasts with a brand-new **4060Ti 16G**, which costs around $480.
- **Sanctions Affect Russian P40 Stock**: A member noted that **sanctions** might be reducing the stock of **P40s in Russia**. They humorously added it shows how *sanctions are working in a weird way*.
- **Home Server Build Insights**: A user inquired about **server specs for a home setup**. Another shared their configuration: *"linux workstation w/ R3700X, 128GB RAM, RTX 4090, 2x 1TB SSD, and some 3.5" SATA HDDs"*.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1250536384717127680)** (16 messages🔥): 

- **LLAMA3 70B generates diverse outputs**: A user reported that when prompting **LLAMA3 70B** from an empty document (`<|begin_of_text|>`), the model generates *“60% evals in AI2's arc format, 20% wiki text, and 20% code”*. This indicates that **LLAMA3** may have undergone final eval tuning for specific formats.
  
- **Help requested for debugging VLLM**: A member asked for help in debugging **VLLM**, to which another replied that they are not a pro but have explored the repository.

- **Finetuning BERT for long texts**: A user sought advice on finetuning a **BERT** model for 6000-word inputs using a sliding window approach. They were directed to specific resources, including a [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf) and its [implementation](https://github.com/Sleepychord/CogLTX).

- **Mixture of Millions of Memory Experts as RAG alternative**: A user mentioned a [research paper](https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf) on adopting a **Mixture of Millions of Memory Experts** for factual memorization and reducing hallucinations as an alternative to **RAG**. Another member felt this approach might be redundant, as it has likely been tried before.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://open-engineer.vercel.app/">Open Engineer</a>: A free educational platform offering AI learning through hands-on projects, practical tools, and foundational theory.</li><li><a href="https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf">Lamini-Memory-Tuning/research-paper.pdf at main · lamini-ai/Lamini-Memory-Tuning</a>: Banishing LLM Hallucinations Requires Rethinking Generalization - lamini-ai/Lamini-Memory-Tuning
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1250534867716935710)** (52 messages🔥): 

- **Samba-3.8B achieves perfect retrieval**: [Microsoft's Samba model](https://github.com/microsoft/Samba) trained on 3.2 trillion tokens significantly outperforms Phi3-mini in major benchmarks. It maintains linear complexity with respect to sequence length while achieving long-context retrieval abilities.
  
- **Challenges in passkey retrieval**: Members discussed Samba's ability to handle passkey retrieval with 256k sequence lengths. The conversation touched on the efficiency of Mamba layers and SWA in handling long-term and local information respectively, with rope potentially influencing results.

- **Search for a neural network paper**: One user tried to recall a paper on neural network behavior with OOD data, mentioning concepts like "mean reversion." Another user quickly found [the likely paper](https://arxiv.org/abs/2310.00873) and shared the link along with additional resources.

- **Self-synthesized high-quality instruction data**: A new method named [Magpie](http://arxiv.org/abs/2406.08464) was introduced, which synthesizes instruction data by prompting an aligned LLM with just the start-of-user-input template. This self-synthesis method aims to generate large-scale alignment data, bypassing the need for manual data creation.

- **Discussion about tying embedding and unembedding layers**: A member asked about the drawbacks of tying embedding and unembedding layers. Another member shared a [LessWrong post](https://www.lesswrong.com/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are) which explains that modern LLMs have moved away from this practice, although empirical data was lacking.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://arxiv.org/abs/2406.08478">What If We Recaption Billions of Web Images with LLaMA-3?</a>: Web-crawled image-text pairs are inherently noisy. Prior studies demonstrate that semantically aligning and enriching textual descriptions of these pairs can significantly enhance model training acros...</li><li><a href="http://arxiv.org/abs/2406.08464">Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing</a>: High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinde...</li><li><a href="https://arxiv.org/abs/2406.07887">An Empirical Study of Mamba-based Language Models</a>: Selective state-space models (SSMs) like Mamba overcome some of the shortcomings of Transformers, such as quadratic computational complexity with sequence length and large inference-time memory requir...</li><li><a href="http://arxiv.org/abs/2406.07548">Image and Video Tokenization with Binary Spherical Quantization</a>: We propose a new transformer-based image and video tokenizer with Binary Spherical Quantization (BSQ). BSQ projects the high-dimensional visual embedding to a lower-dimensional hypersphere and then ap...</li><li><a href="https://arxiv.org/abs/2106.00003">Parallelized Computation and Backpropagation Under Angle-Parametrized Orthogonal Matrices</a>: We present a methodology for parallel acceleration of learning in the presence of matrix orthogonality and unitarity constraints of interest in several branches of machine learning. We show how an app...</li><li><a href="https://arxiv.org/abs/2310.00873">Deep Neural Networks Tend To Extrapolate Predictably</a>: Conventional wisdom suggests that neural network predictions tend to be unpredictable and overconfident when faced with out-of-distribution (OOD) inputs. Our work reassesses this assumption for neural...</li><li><a href="https://arxiv.org/abs/2311.14648">Calibrated Language Models Must Hallucinate</a>: Recent language models generate false but plausible-sounding text with surprising frequency. Such &#34;hallucinations&#34; are an obstacle to the usability of language-based AI systems and can harm pe...</li><li><a href="https://x.com/sakanaailabs/status/1801069076003082502?s=46">Tweet from Sakana AI (@SakanaAILabs)</a>: Can LLMs invent better ways to train LLMs?  At Sakana AI, we’re pioneering AI-driven methods to automate AI research and discovery. We’re excited to release DiscoPOP: a new SOTA preference optimizatio...</li><li><a href="https://arxiv.org/abs/2406.08070v1">CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models</a>: Classifier-free guidance (CFG) is a fundamental tool in modern diffusion models for text-guided generation. Although effective, CFG has notable drawbacks. For instance, DDIM with CFG lacks invertibili...</li><li><a href="https://www.lesswrong.com/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are">LLM Basics: Embedding Spaces - Transformer Token Vectors Are Not Points in Space — LessWrong</a>: This post is written as an explanation of a misconception I had with transformer embedding when I was getting started. Thanks to Stephen Fowler for t…
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1250618347859607613)** (10 messages🔥): 

- **Debate on acc_norm by tokens vs bytes for same tokenizer**: A member questioned the appropriateness of using **acc_norm by tokens** instead of bytes when comparing models with the same tokenizer. Another member noted that "A lot of people do it" and recommended referencing the GPT-3 paper for more details.
  
- **Generate_until issues with Qwen1.5-7B-Chat**: One member reported that 670 out of 749 questions returned empty responses while running the task **truthfulqa_gen** on Qwen1.5-7B-Chat at dtype='float' and included a [pastebin log](https://pastebin.ai/i6qnlbg8x3) for reference. Another member suggested that the omission of `--apply_chat_template` could be the cause, citing prompt formatting issues and recommended using `--fewshot_as_multiturn` as a possible fix.

**Link mentioned**: <a href="https://pastebin.ai/i6qnlbg8x3">log_samples truthfulqa_gen - Pastebin.ai</a>: no description found

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

yash05880: icydk https://laion.ai/blog/open-flamingo/
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1250546854442045481)** (3 messages): 

- **PingCap showcases RAG app using TiDB and LlamaIndex**: Our friends at PingCap have crafted a high-quality **RAG application** centered on their TiDB database utilizing LlamaIndex’s knowledge graph feature. It's open source, and you can try it out [here](https://t.co/JKOa6ab1Uh) or check the code on [GitHub](https://t.co/bUWs9lM1ea).

- **Talk Alert: AI Infrastructure Meetup in Paris**: Join Pierre-Loic Douclet from LlamaIndex, along with speakers from **Gokoyeb** and **Neon**, at an **AI Infrastructure Meetup** at Station F in Paris on Wednesday, June 19. More details and registration [here](https://twitter.com/llama_index/status/1801288913312760205).

- **Mixture-of-Agents (MoA) enhances LLMs capabilities**: A study by **TogetherCompute** shows that a **Mixture-of-Agents (MoA)** setup using exclusively open-source LLMs can significantly enhance task capabilities. Learn more about the potential of MoA [here](https://t.co/awJyjj1F2W).

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/EPwlqsQ4kq">AI Infrastructure · Luma</a>: 📣 Calling all AI developers and infrastructure enthusiasts! We are excited to announce an AI Infrastructure Meetup on Wednesday, June 19! The Linux…</li><li><a href="https://t.co/JKOa6ab1Uh">TiDB.AI</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1250542964124422287)** (61 messages🔥🔥): 

- **Handling Large Embedding Indexes**: A member struggling with query times for an index containing 170,000 embeddings is advised to use vector databases like **Qdrant** or FAISS Index to speed up retrieval times. They shared detailed FAISS index and query code, seeking help with an **AssertionError** during querying.

- **Nodes Labeled as "Unknown"**: Users are discussing why some nodes in the property graph display as "Unknown." Suggestions include possible issues with the implicit extractor and the source/parent relationship, and a recommended fix using `pip install -U llama-index-graph-stores-neo4j`.

- **Extracting Nodes from VectorStores**: A user trying to extract nodes from a **VectorStoreIndex** built with **Chroma** encounters empty dictionaries and learns to use `vector_store._collection.get()` to retrieve nodes. They seek further help on performing this task directly from a VectorStoreIndex.

- **Filtering Node Retrieval Results**: Users discuss different methods to filter results from `index.as_retriever().retrieve()`, focusing on metadata filters, similarity postprocessors, and the use of `get_nodes()` for specific vector stores like Qdrant and Chroma.

- **Qdrant Database Node Retrieval**: A user working with a Qdrant database containing law text entries asks about retrieving adjacent nodes (previous and next articles) during a query. The suggestion involves using node relationships and the new API method in the latest Qdrant vector store.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/module_guides/deploying/agents/tools/#return-direct>).">Tools - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/f5263896121721de1051ce58338a1e0ea6950ca7/llama-index-core/llama_index/core/evaluation/context_relevancy.py">llama_index/llama-index-core/llama_index/core/evaluation/context_relevancy.py at f5263896121721de1051ce58338a1e0ea6950ca7 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/#custom-node-postprocessor>))">Node Postprocessor - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/node_postprocessors/#metadatareplacementpostprocessor>))">Node Postprocessor Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/#using-with-retrieved-nodes>))">Node Postprocessor - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1250796661295550504)** (1 messages): 

- **Embedding PDFs to Weaviate with LLM-Index**: A member asked for suggestions on embedding PDFs and documents to a Weaviate vector database using LLM-Index. The message indicates an ongoing project but does not include further details or responses.
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1250525374799024209)** (56 messages🔥🔥): 

- **Coral rebranded to Command-R**: A user inquired about **Coral's** status, and it was clarified that it's now called **Command-R**, but both the original Command-R and Coral are still functioning. 
- **Model usage parameters and prompt controls**: Discussions highlighted differing practices in model parameter adjustments, with some users favoring **prompt engineering over parameter tuning** for better control, while others shared specific configurations they found effective.
- **Clarification on acceptable use policy**: Users shared and questioned the **[Cohere Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)**, seeking clarity on private versus commercial use with personal projects being permissible for showcasing to employers.
- **Internal server errors with the API**: Users experiencing `ApiError: status_code: 500` errors discussed potential causes, including errors within specific prompts versus request issues, with others suggesting checking request details.
- **Trial key limitations and access issues**: Users reported issues with **trial keys** stating insufficient permissions and hitting usage limits, while others confirmed similar experiences but noted success with production keys.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/terms-of-use">Terms Of Use</a>: Access and use Cohere&#x27;s Natural Language Processing and Large Language Models with confidence, knowing that you are fully informed of our Terms of Use.</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">C4AI Acceptable Use Policy</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1250680242222272613)** (3 messages): 

- **Glarb-Glarb and Fluent API preferences**: One member humorously mentioned "Glarb-Glarb" and expressed their liking for the **Fluent API**.
- **Congratulations on release and thanks for Cohere support**: Another member congratulated someone on their release and expressed gratitude for adding **Cohere support**, ending with a salute emoji.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1250545818918977537)** (41 messages🔥): 

- **Stable Diffusion struggles to generate images of women**: A discussion highlighted that the StabilityAI's **Stable Diffusion** model struggles to generate naked or even clothed images of women due to heavy censorship. Users suggested waiting for custom checkpoints and using second-pass img2img with SD1.5 to achieve desired results. [Discussion link](https://huggingface.co/stabilityai/stable-diffusion-3-medium/discussions/67).

- **Celebration for Luma AI's new text-to-video model**: Users congratulated a member for Luma AI's release of **Dream Machine**, a text-to-video model. Despite some initial server issues, the model shows promise, though users noted its performance varies with complex prompts. [Try Dream Machine](https://lumalabs.ai/dream-machine).

- **Comparison of different AI models**: There's a shared link comparing **SD3 Large, SD3 Medium, Pixart Sigma, DALL E 3, and Midjourney**. This post discusses the reopening of the /r/StableDiffusion subreddit and ongoing issues with Reddit's API changes. [Comparison link](https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/).

- **Interest in video tokenization models**: A user inquired about the best open-source models for converting video into sequences of tokens, similar to **VAE**. Another user admitted they could only help with basic video editing tasks like splitting videos and removing watermarks. [Tweet link](https://fxtwitter.com/blizaine/status/1801126160904098247).

- **Luma Video's potential**: Users remarked that **Luma's Dream Machine** video generation has potential, although it is inconsistent. There are ongoing discussions about its capabilities and room for improvement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium/discussions/67">stabilityai/stable-diffusion-3-medium · The model does not create an image of naked women</a>: no description found</li><li><a href="https://fxtwitter.com/blizaine/status/1801126160904098247">Tweet from Blaine Brown  (@blizaine)</a>: Dream Machine from @LumaLabsAI really brings memes to life!   A thread 🧵</li><li><a href="https://fxtwitter.com/dome_271/status/1800922604511105246">Tweet from dome | Outlier (@dome_271)</a>: LET&#39;S GOOOOOOOOOOO  SO HAPPY TO SHARE OUR FIRST TEXT-TO-VIDEO MODEL!   https://lumalabs.ai/dream-machine  Quoting Luma AI (@LumaLabsAI)   Introducing Dream Machine - a next generation video model ...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://fxtwitter.com/blizaine/status/1801126279917547726">Tweet from Blaine Brown  (@blizaine)</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1250616787385581649)** (14 messages🔥): 

- **AIW Study Exposes Model Instability**: Discussing variations in AIW problems, the study revealed **dramatic breakdowns** in models like GPT-4o when faced with minor changes, such as altering numerical values. The analysis emphasizes that "GPT-4o (and all other models) is NOT robust to AIW variations," highlighting core deficiencies in reasoning capabilities ([source](https://arxiv.org/abs/2406.02061)).

- **DataComp-1B's Improved Captions**: A link shared from Haoqin Tu's personal site details work on enhancing textual descriptions for noisy web-crawled image-text pairs, improving model training through alignment ([source](https://www.haqtu.me/Recap-Datacomp-1B/)) ([paper](https://arxiv.org/abs/2406.08478)).

- **Masked Diffusion Models Simplified**: A shared paper presents a **simple and general framework** for masked diffusion models, showing superior performance over prior models at GPT-2 scale when evaluated by perplexity ([source](https://arxiv.org/abs/2406.04329)).

- **Recaptioning and Dataset Updates**: Multiple mentions indicate active work in recaptioning datasets such as **CC12M** and **GBC10M**, with links provided to complementary resources on Hugging Face ([GBC10M](https://huggingface.co/datasets/graph-based-captions/GBC10M), [CC12M LLaVaNext](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.08478">What If We Recaption Billions of Web Images with LLaMA-3?</a>: Web-crawled image-text pairs are inherently noisy. Prior studies demonstrate that semantically aligning and enriching textual descriptions of these pairs can significantly enhance model training acros...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>: Large Language Models (LLMs) are often described as being instances of foundation models - that is, models that transfer strongly across various tasks and conditions in few-show or zero-shot manner, w...</li><li><a href="https://www.haqtu.me/Recap-Datacomp-1B/">What If We Recaption Billions of Web Images with LLaMA-3?</a>: no description found</li><li><a href="https://arxiv.org/abs/2406.04329">Simplified and Generalized Masked Diffusion for Discrete Data</a>: Masked (or absorbing) diffusion is actively explored as an alternative to autoregressive models for generative modeling of discrete data. However, existing work in this area has been hindered by unnec...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/)** (1 messages): 

sidfeels: <@&825830190600683521>
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/)** (1 messages): 

.michu7: <@&825830190600683521>
  

---


### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/)** (1 messages): 

zuwop21: 50$ from steam
[steamcommunity.com/glft/918524](https://sc.link/HSvw7)
@everyone
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1250589784586453112)** (49 messages🔥): 

- **Seeking speech-to-text datasets and solutions**: A member asked for recommendations on speech-to-text datasets with MP3 files and original transcripts with diarization. They also mentioned having used AWS Transcribe, OpenAI Whisper, and Deepgram Nova-2 and are looking for the best available solution in this space.

- **Handling empty tool calls in chains**: A user is dealing with an issue in which a chain throws an error when the 'tool_calls' list is empty. They are seeking advice on managing scenarios where user input does not require tool usage, ensuring the chain can handle simple responses without errors.

- **Evaluating LLM message similarity**: Another user inquired about how LangChain handles the similarity measure between LLM messages and expected script messages. The response explained that LangChain can use string distance metrics or embedding distance metrics to evaluate this similarity, with practical code examples provided.

- **Design patterns for state management in LangGraph**: There was an extended discussion on managing state in LangGraph, with particular interest in integrating user ID and thread ID into the state. The conversation centered around whether state should include the entire conversation history or only recent interactions, offering detailed examples and best practices.

- **Handling human intervention and streaming responses in LangChain**: Members discussed methods for managing human intervention in LangGraph workflows and resuming conversations seamlessly. There were also questions about streaming responses while retaining chat history, with code examples provided to demonstrate maintaining context across multiple chat turns.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/templates/chat-bot-feedback/#usage>).">Chat Bot Feedback Template | 🦜️🔗 LangChain</a>: This template shows how to evaluate your chat bot without explicit user feedback. It defines a simple chat bot in chain.py and custom evaluator that scores bot response effectiveness based on the subs...</li><li><a href="https://docs.smith.langchain.com/how_to_guides/evaluation/use_langchain_off_the_shelf_evaluators#use-string-or-embedding-distance-metrics>).">Use LangChain off-the-shelf evaluators (Python only) | 🦜️🛠️ LangSmith</a>: Before diving into this content, it might be helpful to read the following:</li><li><a href="https://github.com/langchain-ai/langchain/issues/15934>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#basic-usage>)">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 Langchain</a>: Here we focus on how to move from legacy LangChain agents to LangGraph</li><li><a href="https://github.com/langchain-ai/langchain/issues/18598>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#agents>))">How to add chat history | 🦜️🔗 LangChain</a>: In many Q&amp;A applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of &quot;memory&quot; of past questions and answers, and some logi...</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#basic-usage>))">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 Langchain</a>: Here we focus on how to move from legacy LangChain agents to LangGraph</li><li><a href="https://github.com/langchain-ai/langchain/issues/18598>))">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1250670200982994994)** (2 messages): 

- **Tiny AI Client simplifies LLM use**: A member shared a project on GitHub called [tiny-ai-client](https://github.com/piEsposito/tiny-ai-client), describing it as a "tiny LLM client for simple use cases," supporting tools and vision for OAI, Anthropic, and Gemini models. They expressed hope that it would be helpful to others in the community.
- **Run LLM locally with Docker and Ollama**: Another member created a [YouTube video](https://youtu.be/NLOY9RLMI6k?si=-OdUtYSWTJwhvtzy) demonstrating how to run LLMs locally using Docker and Ollama. They invited feedback on their video, titled "Exécuter des LLMs en local avec Docker et Ollama."

**Link mentioned**: <a href="https://youtu.be/NLOY9RLMI6k?si=-OdUtYSWTJwhvtzy">Exécuter des LLMs en local avec Docker et Ollama</a>: no description found

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1250645308661039224)** (1 messages): 

- **New tutorial on setting up LLM on Google Colab**: A member shared a *comprehensive tutorial* on **GitHub** for setting up **LLM** on Google Colab, taking advantage of the free 15GB Tesla T4 Colab GPU. The tutorial includes instructions for both **GPU-accelerated** and **CPU-only** inference [here](https://github.com/casualcomputer/llm_google_colab).

**Link mentioned**: <a href="https://github.com/casualcomputer/llm_google_colab">GitHub - casualcomputer/llm_google_colab: A tutorial on how to set up a LLM on Google Colab for both GPU-accelerated and CPU-only session.</a>: A tutorial on how to set up a LLM on Google Colab for both GPU-accelerated and CPU-only session. - casualcomputer/llm_google_colab

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1250569249353633812)** (7 messages): 

- **Windows support hinted for Fall release**: A member inquired about when Windows support would be available. Another responded, guessing an end of summer or Fall release and referenced an upcoming [YouTube livestream](https://m.youtube.com/watch?v=uookgZ7Ojg8) for updates.
- **WSL as a workaround for Windows users**: A member reported successfully using WSL (Windows Subsystem for Linux) as a temporary workaround for Mojo development on Windows. However, the asker preferred a native Windows build environment to handle PE format.

**Link mentioned**: <a href="https://m.youtube.com/watch?v=uookgZ7Ojg8">Modular Community Livestream - New in MAX 24.4</a>: MAX 24.4 is now available! Join us on our upcoming livestream as we discuss what’s new in MAX Engine and Mojo🔥 - MAX on macOS, MAX Engine Quantization API, ...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1250553872603287633)** (31 messages🔥): 

- **Non-empty Strings in Mojo are Truthy**: A user discovered that any non-empty string in Mojo evaluates as truthy, leading to some unexpected behavior in their code. This was clarified with explanations that constants like `String.ASCII_LOWERCASE` are non-empty strings, thus always evaluating to True.
- **Mojo LSP Configuration in Neovim**: There was a query about setting up Mojo LSP in Neovim. It was answered with a [GitHub link](https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#mojo) confirming that it comes pre-installed with Neovim.
- **Matrix Multiplication Performance in Mojo vs. Python**: A user discussed performance benchmarks comparing Mojo and Python for small, fixed-sized matrix multiplications. They found that Mojo performed significantly faster due to Python's overhead in using libraries like numpy for small calculations.
- **Iterators and Loop Behavior in Mojo**: Some users discussed the reassignment behavior of loop variables in Mojo, particularly the use of `for` loops and modifying `i` inside the loop. It was suggested to use `while` loops if persistent modifications are needed.
- **Stdin and Stdout in Mojo**: There was a question about the lack of `stdin` support in Mojo, which was confirmed as not currently available.

**Link mentioned**: <a href="https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#mojo">nvim-lspconfig/doc/server_configurations.md at master · neovim/nvim-lspconfig</a>: Quickstart configs for Nvim LSP. Contribute to neovim/nvim-lspconfig development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1250595066184073226)** (13 messages🔥): 

- **Conditional non-conformance discussed**: A member suggested the need for conditional non-conformance in order to "get rid of implicit copies that can be made by dereferencing." Another member mentioned this might be handled by the ExplicitlyCopyable trait and CollectionElementNew.
- **Compiler limitation troubleshooting**: When a member asked if a compiler error was a bug, another clarified it was likely a compiler limitation. A suggested workaround was to use `fn (Optional[T]) capturing -> Bool` while assuming the argument would never be none.
- **New nightly Mojo compiler release announced**: The latest nightly Mojo compiler release, version `2024.6.1305`, was announced with changelog and raw diff links. A member humorously remarked on remembering to update with `modular update nightly/max` instead of `modular update nightly/mojo`.
- **Alias for updates suggested**: In response to the update command confusion, a member suggested setting up an alias to simplify the process. This was humorously acknowledged as a good example of "Actual Intelligence."

- **Conditional non-conformance discussed**: A member suggested the need for conditional non-conformance in order to "get rid of implicit copies that can be made by dereferencing." Another member mentioned this might be handled by the ExplicitlyCopyable trait and CollectionElementNew.
- **Compiler limitation troubleshooting**: When a member asked if a compiler error was a bug, another clarified it was likely a compiler limitation. A suggested workaround was to use `fn (Optional[T]) capturing -> Bool` while assuming the argument would never be none.
- **New nightly Mojo compiler release announced**: The latest nightly Mojo compiler release, version `2024.6.1305`, was announced with changelog and raw diff links. A member humorously remarked on remembering to update with `modular update nightly/max` instead of `modular update nightly/mojo`.
- **Alias for updates suggested**: In response to the update command confusion, a member suggested setting up an alias to simplify the process. This was humorously acknowledged as a good example of "Actual Intelligence."

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1250530820779937933)** (33 messages🔥): 

- **OpenAI's revenue skyrockets without Microsoft's help**: Revenue at OpenAI has nearly doubled in the last six months, and it primarily comes from direct sales of ChatGPT and other OpenAI products rather than from Microsoft's sales. [Source](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023?utm_source=ti_app&rc=c48ukx).
- **Sakana AI introduces DiscoPOP**: Sakana AI has launched DiscoPOP, a new state-of-the-art preference optimization algorithm discovered and written by an LLM, showcasing AI-driven methods to automate AI research. [Check the details](https://x.com/SakanaAILabs/status/1801069076003082502) and explore the [paper](https://arxiv.org/abs/2406.08414) and [GitHub repo](https://github.com/SakanaAI/DiscoPOP).
- **Mira Murati comments on OpenAI's models**: Mira Murati mentioned that the AI models available in OpenAI's labs are not significantly more advanced than the ones publicly available. [Source](https://x.com/tsarnick/status/1801022339162800336).
- **Rumors about Reka's acquisition**: Reka was rumored to be acquired by Snowflake for a billion dollars, but shortly after, Reka announced a long-term partnership with Shutterstock. This speculation highlights the dynamic nature of acquisitions in the AI industry.
- **Apple and OpenAI's surprising partnership**: Apple's partnership with OpenAI focuses more on promoting OpenAI’s brand and technology across its devices rather than generating substantial initial revenue. [Detailed article](https://www.bloomberg.com/news/articles/2024-06-12/apple-to-pay-openai-for-chatgpt-through-distribution-not-cash).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-06-12/apple-to-pay-openai-for-chatgpt-through-distribution-not-cash">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/amir/status/1800992276652630455">Tweet from Amir Efrati (@amir)</a>: News:   Revenue at OpenAI ~doubled~ in the last 6 months.   And while many ppl thought a lot of revenue was coming from Microsoft selling OpenAI tech and giving the startup a cut… au contraire. This r...</li><li><a href="https://x.com/SakanaAILabs/status/1801069076003082502">Tweet from Sakana AI (@SakanaAILabs)</a>: Can LLMs invent better ways to train LLMs?  At Sakana AI, we’re pioneering AI-driven methods to automate AI research and discovery. We’re excited to release DiscoPOP: a new SOTA preference optimizatio...</li><li><a href="https://x.com/tsarnick/status/1801022339162800336">Tweet from Tsarathustra (@tsarnick)</a>: Mira Murati says the AI models that OpenAI have in their labs are not much more advanced than those which are publicly available
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1250734937133285448)** (4 messages): 

- **Unusual Paper Submission Discussed**: A member shared a tweet from [Cosmin Negruseri](https://x.com/cosminnegruseri/status/1800683283069767691) and linked to an arXiv paper [2404.07221](https://arxiv.org/abs/2404.07221) noting an unusual aspect. Another member humorously responded, expressing surprise with "Hahahhahaha what."

**Link mentioned**: <a href="https://x.com/cosminnegruseri/status/1800683283069767691">Tweet from Cosmin Negruseri (@cosminnegruseri)</a>: haven&#39;t seen this in a paper before

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1250550474684366898)** (6 messages): 

- **Nvidia Nemotron hype grows**: A member shared a [tweet](https://x.com/SebastianB929/status/1800991419437367655) suggesting the potential release of Nvidia Nemotron. This tweet initiated excitement among the community about the new hardware.

- **Latest research paper on speech models**: A member posted a link to a [research paper on arXiv](https://arxiv.org/abs/2402.16819) authored by several researchers including Jupinder Parmar and Shrimai Prabhumoye. The paper explores advancements in speech modeling.

- **Insights from insider connection**: Another member commented, "their head of alignment is a paid sub and friend but not in the discord I think." This sparked curiosity about potential inside information related to the project discussed earlier.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SebastianB929/status/1800991419437367655">Tweet from SebastianBoo (@SebastianB929)</a>: Seems like Nvidia Nemotron  Quoting Xeophon (@TheXeophon)   june-chatbot👀</li><li><a href="https://arxiv.org/abs/2402.16819">Nemotron-4 15B Technical Report</a>: We introduce Nemotron-4 15B, a 15-billion-parameter large multilingual language model trained on 8 trillion text tokens. Nemotron-4 15B demonstrates strong performance when assessed on English, multil...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1250824109898862752)** (7 messages): 

- **Samba 3.8B smashes benchmarks**: A [tweet by Liliang Ren](https://x.com/liliang_ren/status/1801027052147216457) introduces Samba 3.8B, a combination of Mamba and Sliding Window Attention architecture that significantly outperforms Phi3-mini on major benchmarks like MMLU, GSM8K, and HumanEval. The architecture boasts an infinite context length with linear complexity ([paper here](https://arxiv.org/abs/2406.07522)).
- **SSMs still hold ground**: Members express relief and optimism that **SSMs** (Structured State Machines) are still being actively developed and discussed. One member humorously remarks that they are 50/50 on whether they will continue, noting the odds are decent.
- **Hybrids may be the future**: There is a consensus that trends are moving towards **hybrid SSM/transformer architectures**, citing the logic that attention isn't required for every layer.

**Link mentioned**: <a href="https://x.com/liliang_ren/status/1801027052147216457">Tweet from Liliang Ren (@liliang_ren)</a>: Introducing Samba 3.8B, a simple Mamba+Sliding Window Attention architecture that outperforms Phi3-mini on major benchmarks (e.g., MMLU, GSM8K and HumanEval) by a large margin.😮 And it has an infinit...

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1250543546579161169)** (43 messages🔥): 

- **Haize Labs unveils AI jailbreak detection**: A new startup, [Haize Labs](https://x.com/haizelabs/status/1800936720990384174?s=46&t=90xQ8sGy63D2OtiaoGJuww), has announced a manifesto focusing on preemptively discovering and eliminating failure modes in AI systems. They showcased their ability to jailbreak the safety guardrails of industry-leading AI models, exposing serious safety violations.
  
- **Open Repro of iPad Calculator by tldraw**: [tldraw](https://x.com/tldraw/status/1800515870709706879?s=46&t=90xQ8sGy63D2OtiaoGJuww) shared an open reproduction of Apple's iPad calculator, sparking interest and admiration. The tldraw team, though small, continually impresses with their innovative thought experiments and rapid demonstrations.

- **Article on Amazon's AI struggles**: An article shared by [cakecrusher](https://www.mihaileric.com/posts/how-alexa-dropped-the-ball-conversational-ai/) discusses how Amazon's culture and operations led to their lag in conversational AI development. Former employees mentioned issues like byzantine build/deploy systems and a product-first mentality hindering long-term development.

- **OpenAI's Revenue Soars**: OpenAI has reached an annualized revenue of [\$3.4 billion](https://x.com/deedydas/status/1801003523292729789), as shared by @deedydas. Discussions ensued about the profitability and potential burn rate associated with such impressive revenue figures.

- **Argilla joins Hugging Face**: [Argilla](https://argilla.io/blog/argilla-joins-hugggingface) announced their integration with Hugging Face, promising enhanced synergies for generating valuable datasets and content. The partnership aims to amplify both teams' efforts in building great products and fostering innovation in the AI space.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=sQar5NNGbw4">Scaling interpretability</a>: Science and engineering are inseparable. Our researchers reflect on the close relationship between scientific and engineering progress, and discuss the techn...</li><li><a href="https://www.mihaileric.com/posts/how-alexa-dropped-the-ball-conversational-ai/">no title found</a>: no description found</li><li><a href="https://x.com/tldraw/status/1800515870709706879?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from tldraw (@tldraw)</a>: i have an idea</li><li><a href="https://x.com/tldraw/status/1800515870709706879?s=46&t=">Tweet from tldraw (@tldraw)</a>: i have an idea</li><li><a href="https://x.com/haizelabs/status/1800936720990384174?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Haize Labs (@haizelabs)</a>: Today is a bad, bad day to be a language model.  Today, we announce the Haize Labs manifesto.  @haizelabs haizes (automatically red-teams) AI systems to preemptively discover and eliminate any failure...</li><li><a href="https://x.com/alvarobartt/status/1801278221901512839?s=46">Tweet from Alvaro Bartolome (@alvarobartt)</a>: As you may already know, @argilla_io  is now joining @huggingface 🎉  On a professional level, it&#39;s incredible to see two great teams sharing the same passion. I firmly believe in Argilla&#39;s mi...</li><li><a href="https://x.com/tldraw/status/1801212867061879175?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from tldraw (@tldraw)</a>: can your calculator do this?</li><li><a href="https://x.com/tldraw/status/1801264226314408029?s=46">Tweet from tldraw (@tldraw)</a>: Be maths</li><li><a href="https://x.com/realsharonzhou/status/1801271891954696317?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Sharon Zhou (@realSharonZhou)</a>: Excited to announce @LaminiAI Memory Tuning, a new research breakthrough! 🎉 ◽95%+ accuracy, cutting hallucinations by 10x ◽Turns any open LLM into a 1M-way adapter MoE (paper & Lamini-1 model weights...</li><li><a href="https://x.com/deedydas/status/1801003523292729789">Tweet from Deedy (@deedydas)</a>: OpenAI is at $3.4B annualized revenue. Wow.</li><li><a href="https://x.com/the_aiju/status/1800986743832736129?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Emily (@the_aiju)</a>: apparently you can nerdsnipe an LLM by asking it if there are primes whose digits sum to 9 (the correct answer is “no”) ^^ every single one i’ve tried it on goes crazy trying every single prime it can...</li><li><a href="https://www.anthropic.com/research/engineering-challenges-interpretability">The engineering challenges of scaling interpretability</a>: Anthropic is an AI safety and research company that&#x27;s working to build reliable, interpretable, and steerable AI systems.</li><li><a href="https://x.com/liliang_ren/status/1801027052147216457?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Liliang Ren (@liliang_ren)</a>: Introducing Samba 3.8B, a simple Mamba+Sliding Window Attention architecture that outperforms Phi3-mini on major benchmarks (e.g., MMLU, GSM8K and HumanEval) by a large margin.😮 And it has an infinit...</li><li><a href="https://github.com/microsoft/Samba">GitHub - microsoft/Samba: Official implementation of &quot;Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling&quot;</a>: Official implementation of &quot;Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling&quot; - microsoft/Samba
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1250550501959925921)** (40 messages🔥): 

- **Clarifying Open Interpreter's Capabilities**: A discussion clarified that **Open Interpreter** gives agentic power to Large Language Models (**LLMs**). One member explained, "*OI converts natural language into computer control*" and considered potential future integration with machine-specific **LLMs** and sensory input models.
- **Running Vision Models**: Members helped each other run simultaneous code and vision models with **Open Interpreter**, such as the `llama3-vision.py` profile. They discussed methods and issues like downloading models and getting them to perform tasks like screengrabs.
- **Browser Automation Example**: A user shared their success having **Open Interpreter** browse the web to get the current Mavericks score using a simple command. They highlighted the straightforward nature of the prompt and the potential server load causing delays.
- **Whisper STT Library Query**: A member inquired about a good/easy **Whisper Speech-To-Text (STT)** library only to later mention they created their own solution.
- **Performance and Customization**: There were discussions about performance issues, server load, and potential customizations, particularly in modifying *core.py* to achieve desired results.
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1250857028201480204)** (5 messages): 

- **Apple's model param size revealed in WWDC**: Curious about the model param size used in Apple's recent announcement? It's a **3 billion parameter** on-device language model. Details found [here](https://machinelearning.apple.com/research/introducing-apple-foundation-models).
- **Apple vs. model win clarified**: There was a query about whether the term "win" referred to Apple's victory or a model's victory. A participant clarified: "I believe it's Apple's model won against the selected model".
- **Model optimization technique explained**: Apple's on-device inference uses **low-bit palletization** and a mixed 2-bit and 4-bit configuration strategy. This approach averages **3.5 bits-per-weight**, achieving the same accuracy as uncompressed models while optimizing memory, power, and performance.

**Link mentioned**: <a href="https://machinelearning.apple.com/research/introducing-apple-foundation-models">Introducing Apple’s On-Device and Server Foundation Models</a>: At the 2024 Worldwide Developers Conference, we introduced Apple Intelligence, a personal intelligence system integrated deeply into…

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1250834851654013053)** (16 messages🔥): 

- **Docker Desktop struggles to find GPU**: A member tried running axolotl via Docker on a virtual machine (Ubuntu) on Windows 11 but received an error stating "No GPU found." They ran `docker run --gpus all --rm -it winglian/axolotl:main-latest` and `accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml` with no success.

- **Suggest using `nvidia-smi`**: Another member suggested running `nvidia-smi` to check GPU status. They also inquired if the CUDA toolkit was installed on the host system.

- **Where to install CUDA toolkit**: The troubleshooting discussion led to questions about whether to install the CUDA toolkit on Windows or Ubuntu.

- **Nvidia toolkit installation guide shared**: A link to the [NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) was shared to aid in setting up the toolkit.

- **WSL 2 and GPU configuration**: The user clarified they are using WSL 2 and will try configuring CUDA in Ubuntu WSL. They expressed gratitude and mentioned attempting the process again.

**Link mentioned**: <a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">Installing the NVIDIA Container Toolkit &mdash; NVIDIA Container Toolkit 1.15.0 documentation</a>: no description found

  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1250573286904168478)** (16 messages🔥): 

- **OpenRouter UI clamps unsupported parameters**: When user questioned about supporting Temp > 1 and Min P for Command R in OpenRouter, Alex Atallah clarified that while the UI supports it, parameters like temp will be clamped down to temp=1, and Min P will not be passed.

- **High response times for Mistral 7B models**: A user observed high response times for all Mistral 7B variants and linked it to changes in context length and possible rerouting of models. The discussion also pointed out a [context length adjustment](https://orw.karleo.net/changes) and continuous disruptions indicated by a [model uptime tracker](https://openrouter.ai/models/mistralai/mistral-7b-instruct%3Anitro/uptime).

- **Employment offer**: A user introduced themselves as a senior full stack & blockchain developer, stating they have enough experience and are seeking job opportunities.

- **Request for vision models**: Another user inquired about plans to add more vision models, like cogvlm2, for better dataset captioning capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mistr">OpenRouter</a>: LLM router and marketplace</li><li><a href="https://orw.karleo.net/changes">OpenRouter API Watcher</a>: Explore OpenRouter's model list and recorded changes. Updates every hour.</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct%3Anitro/uptime">Mistral: Mistral 7B Instruct (nitro) – Uptime and Availability</a>: Uptime statistics for Mistral: Mistral 7B Instruct (nitro) across providers - A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.  Note: this is...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1250708697714266204)** (3 messages): 

- **RDNA3 Assembly Bounty Hype**: George Hotz mentioned a [GitHub PR for RDNA3 assembly support](https://github.com/tinygrad/tinygrad/pull/3637), inviting contributions to the bounty. This update seeks to enhance **tinygrad** by adding RDNA3 assembly support.
- **Qualcomm Kernel Level Bounty Suggested**: Hotz also recommended a bounty for a "Qualcomm Kernel level GPU driver with HCQ graph support" as a solid starting point for those possessing a Qualcomm smartphone and low-level Linux knowledge. This suggests an opportunity to contribute to the GPU driver development.
- **Tinygrad works smoothly in Termux**: Hotz confirmed that **tinygrad** operates well within **Termux**. This implies that tinygrad is versatile and can be used in various environments including mobile.

**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/pull/3637">RDNA3 assembly support by geohot · Pull Request #3637 · tinygrad/tinygrad</a>: no description found

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1250589848344199250)** (10 messages🔥): 

- **Mixed precision in TinyGrad using bfloat16 and float32**: A member asked if mixed precision could be mimicked by casting during matmul operations. Another member confirmed it's possible and likely faster (*"especially if it fits tensor core dtype pattern"*).

- **Indexing a tensor with computed index X**: A member sought advice on efficient ways to access tensor indices computed from kernel operations and mentioned a potential inefficiency involving multiple kernels. They also referenced boolean indexing patterns discussed in PR#3707.

- **Using UOp graph in TinyGrad**: A member shared code for compiling and running a UOp graph using `MetalDevice` and `MetalCompiler`, but needed guidance on executing the compiled kernel. Another member suggested looking into `compiledRunner` for further information.
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1250871666096083005)** (6 messages): 

- **Sober AI reigns at Databricks Summit:** A blog post from [dbreunig](https://www.dbreunig.com/2024/06/12/sober-ai-is-the-norm.html) emphasizes that most AI work is practical and grounded, contrary to hype from companies like OpenAI and Google. He notes that best practices are just now settling, likening modern LLM work to data science in 2019.
- **GPU vs. Spark capacity fights:** Current AI development battles for GPU cores and VRAM swaps, contrasting with the previous struggles over Spark capacity and RAM swaps in data engineering. This highlights the evolving nature of technical constraints in the field.
- **High LLM input-to-output ratio:** At Databricks, clients have a 9:1 ratio of LLM input to output, suggesting that input token pricing is more significant than output token pricing. This ratio underscores the economic aspect of operating LLMs effectively.
- **Frontpage success:** dbreunig's insights from the Databricks Summit hit the front page of Hacker News, indicating a broad interest in the discussed themes of sober AI and practical application challenges in the AI domain today.

**Link mentioned**: <a href="https://www.dbreunig.com/2024/06/12/sober-ai-is-the-norm.html">Sober AI is the Norm</a>: Sober AI is the quiet default, despite all the hype you hear about human-replacements and AGI. Data scientists and engineers are quietly transforming business intelligence through practical applicatio...

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1250780693030637588)** (3 messages): 

- **Aleph Alpha and Silo AI partner up for European AI**: Aleph Alpha and Silo AI announced a strategic partnership to advance open-source AI and enterprise-grade solutions across Europe. Their collaboration aims to enhance AI deployment in industrial firms by leveraging Aleph Alpha’s tech stack and Silo AI’s expertise with a 300+ strong AI team. [Aleph Alpha and Silo AI Partnership](https://aleph-alpha.com/aleph-alpha-and-silo-ai-enter-a-strategic-partnership-to-advance-open-source-ai-and-enterprise-grade-solutions-in-europe/)

**Link mentioned**: <a href="https://aleph-alpha.com/aleph-alpha-and-silo-ai-enter-a-strategic-partnership-to-advance-open-source-ai-and-enterprise-grade-solutions-in-europe/">Aleph Alpha and Silo AI enter a strategic partnership to advance open source AI and enterprise-grade solutions in Europe - ALEPH ALPHA - AI for Enterprises and Governments</a>: To foster the adoption and fully leverage the potential of generative AI across European industrial firms, Europe’s largest AI lab Silo AI and European AI champion Aleph Alpha are announcing a long-te...

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1250813272375038033)** (1 messages): 

- **Survey on Torchtune user model serving**: A user prompted the community to participate in a poll regarding how they serve their finetuned models. They expressed appreciation for any help, stating, "*Thanks for your help!*".
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1250529900495110267)** (1 messages): 

- **Proposed tokenizer revamp receives enthusiastic suggestion**: A developer shared an RFC for a [wide-scale tokenizer revamp](https://github.com/pytorch/torchtune/pull/1082), detailing important reasons for the change. They emphasized benefits like easier addition of features, better composability, and reduced onboarding time for new or updated model tokenizers.

**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/1082.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1250792339916718161)** (1 messages): 

- **Join Infer: Summer '24 for AI insights**: Qwak is hosting a **free virtual conference** on June 26, aimed at AI and ML enthusiasts. The event features live expert interactions and practical insights from leaders in the field.
- **Insightful Talks and Practical Knowledge**: The highlights include discussions on building **recommender systems** and exploring **AI in sports**. You’ll learn techniques for implementing predictive solutions and robust systems focusing on architecture and user engagement.
- **Esteemed Speaker Lineup**: Experts from **Lightricks, LSports, and Lili Banking** will share their real-world experiences and knowledge. Notable speakers include **Hudson Buzby**, Solutions Architect at Qwak, and **Russ Wilcox**, Data Scientist and AI Consultant at ArtifexAI.
- **Register Now for Free Access**: Don’t miss this opportunity to expand your AI acumen and network with top industry professionals. [Register Here for Free](https://tinyurl.com/j8z6s8ka) and join the event on June 26, 2024.

**Link mentioned**: <a href="https://tinyurl.com/j8z6s8ka">Infer Summer ‘24 by Qwak | The Engineering Behind AI and ML</a>: Infer Summer ‘24 by Qwak brings AI leaders to share how the world’s leading companies use ML and AI in production. Join live on Jun 26, 2024, 11:00 AM EDT

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1250821788481359932)** (1 messages): 

- **Upcoming AI Software Development Systems Event**: An announcement about an upcoming event, **"AI software development systems that lead to a future where developers are amplified, not automated"**, has been made. Details and RSVP link are [here](https://discord.com/events/1089876418936180786/1242653066512175157).

- **Machine Learning Paper Picks #2 Released**: The latest edition of **Machine Learning Paper Picks** has been released. Check it out [here](https://discord.com/channels/1089876418936180786/1250679657263534152/1250679657263534152) for a curated selection of papers.

- **New CambAI Team Event**: A new event with the **CambAI Team** has been posted. More information and RSVP link can be found [here](https://discord.com/events/1089876418936180786/1250168740667195455).

- **AMA Attendees Role Assignment**: AMA attendees are reminded to pick up the **0din role** via the <id:customize> link. This role is necessary to be notified about T-shirt distributions.

- **Member-Requested Tags for Curated Channel**: A new `member-requested` tag has been added for users to contribute to the curated [channel](https://discord.com/channels/1089876418936180786/1231977676458168381). Special thanks were given to the member who initiated this.

- **Builders Program for Funding and Support**: Interested members are encouraged to check out the **Builders Program** for funding and support for their projects. [Details here](https://discord.com/channels/1089876418936180786/1089876419926032396/1247228938346958859).
  

---



### **YAIG (a16z Infra) ▷ #[tech-discussion](https://discord.com/channels/958905134119784489/960713746702020608/1250874031679475743)** (1 messages): 

- **Survey on GitHub Codespaces usage**: A member initiated a survey asking if teams use GitHub Codespaces, prompting a response of either ✅ for yes or ❌ for no. This message likely aims to gauge the adoption and utilization of the feature within teams.
  

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
