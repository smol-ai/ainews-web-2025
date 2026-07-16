---
id: MjAyNS0x
title: not much happened today
date: '2026-07-15T05:44:39.731046Z'
description: >-
  **Thinking Machines Lab** launched **Inkling**, its first fully released
  open-weights foundation model family, featuring **975B parameters** with **41B
  active parameters** in a **Mixture-of-Experts** architecture. Inkling supports
  **multimodality** with text, image, and audio inputs and text output, is
  **Apache 2.0 licensed**, and offers up to **1M context window**. The model is
  available on platforms like **Tinker**, **Hugging Face**, and partners, with
  broad ecosystem support from **vLLM**, **SGLang**, **Modal**, **Baseten**, and
  **Databricks**. Key figures such as **Mira Murati**, **Soumith Chintala**,
  **John Schulman**, and **Lilian Weng** highlighted its open weights,
  customization, and practical use focus. Independent commentators noted it as
  the strongest U.S.-based open-weight release to date, though still behind top
  Chinese open-weight and best closed models on some benchmarks.
companies:
  - thinking-machines-lab
  - huggingface
  - vllm_project
  - lmsysorg
  - modal
  - baseten
  - databricks
models:
  - inkling
topics:
  - mixture-of-experts
  - multimodality
  - foundation-models
  - model-licensing
  - context-window
  - open-weights
  - model-release
people:
  - miramurati
  - soumithchintala
  - johnschulman2
  - lilianweng
  - natolambert
  - artificialanlys
  - scaling01
---



**a quiet day.**

> AI News for 7/14/2026-7/15/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap



## What happened


**Thinking Machines Lab launched Inkling, its first fully released open-weights foundation model family entry, positioning it as a customizable multimodal base model rather than a benchmark-maxed flagship.**

- Thinking Machines announced Inkling as an open-weights model that “reasons efficiently across text, image, and audio modalities,” with full weights available and immediate support on its Tinker platform and Playground [@thinkymachines](https://x.com/thinkymachines/status/2077454609551921208).
- Mira Murati described Inkling as the company’s “first model,” “trained from scratch,” with open weights and same-day fine-tuning on Tinker [@miramurati](https://x.com/miramurati/status/2077455974743593100).
- Soumith Chintala framed it as Thinking Machines’ “first general model,” stressing open weights, 975B parameters, native multimodality, and availability on Tinker, Hugging Face, and partners [@soumithchintala](https://x.com/soumithchintala/status/2077457110728884327).
- John Schulman added timeline context: pretraining began last winter, and from mid-January a small team built coding, reasoning, and agentic training on top [@johnschulman2](https://x.com/johnschulman2/status/2077460227327467982).
- Lilian Weng characterized Inkling as a foundation model aimed at “solid performance across a broad categories of capabilities” and intended for practical use plus customization [@lilianweng](https://x.com/lilianweng/status/2077471903032528912).
- TML staff repeatedly emphasized that this is a day-1 release and a foundation for future iterations rather than their final frontier push [@soumithchintala](https://x.com/soumithchintala/status/2077457644474998831), [@cHHillee](https://x.com/cHHillee/status/2077457790423969806), [@keirp1](https://x.com/keirp1/status/2077469773684981962).
- The release landed with unusually broad day-0 ecosystem support across vLLM, SGLang, Modal, Baseten, Databricks, Hugging Face, and quantization/community tooling [@vllm_project](https://x.com/vllm_project/status/2077459955117109343), [@lmsysorg](https://x.com/lmsysorg/status/2077457150046269779), [@modal](https://x.com/modal/status/2077462393441948010), [@baseten](https://x.com/baseten/status/2077462904388178107), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2077462536337891748), [@huggingface](https://x.com/huggingface/status/2077460253235724408), [@danielhanchen](https://x.com/danielhanchen/status/2077468775478423601).
- Independent commentators immediately tagged it as the strongest U.S.-based open-weight release so far, though generally still behind the top Chinese open-weight and best closed models on some benchmarks [@natolambert](https://x.com/natolambert/status/2077454404433903816), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939), [@scaling01](https://x.com/scaling01/status/2077465762869194973).


## Core facts and specs


### Model size, modality, licensing, context

- Inkling is reported as **975B total parameters / 41B active parameters** in most posts [@soumithchintala](https://x.com/soumithchintala/status/2077457110728884327), [@vllm_project](https://x.com/vllm_project/status/2077459955117109343), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939), [@kimmonismus](https://x.com/kimmonismus/status/2077472478499053846).  
  - One tweet says 974B [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2077462536337891748), and another says 952B [@multimodalart](https://x.com/multimodalart/status/2077469546563461353); the overwhelming consensus in the tweet set is ~975B.
- It is a **Mixture-of-Experts** model with **41B active** parameters per token [@VictoriaLinML](https://x.com/VictoriaLinML/status/2077599145502835108).
- It is **Apache 2.0 licensed** according to multiple reactions and summaries [@natolambert](https://x.com/natolambert/status/2077454404433903816), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2077462536337891748), [@multimodalart](https://x.com/multimodalart/status/2077469546563461353).
- It supports **text, image, and audio inputs**, with **text output** [@soumithchintala](https://x.com/soumithchintala/status/2077457110728884327), [@TheRundownAI](https://x.com/TheRundownAI/status/2077472283757543602), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939).
- Open-weights checkpoints support up to **1M context** [@vllm_project](https://x.com/vllm_project/status/2077459955117109343), [@lmsysorg](https://x.com/lmsysorg/status/2077457150046269779), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939).
- Tinker/API context is described as **256K**, with pricing differentiated for **64K** and **256K** contexts [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939).

### Training and release details

- TML says Inkling was **trained from scratch** [@miramurati](https://x.com/miramurati/status/2077455974743593100), [@LiorOnAI](https://x.com/LiorOnAI/status/2077464289611563389).
- Community readers extracted **45T training tokens** from the release materials [@eliebakouch](https://x.com/eliebakouch/status/2077463243463721085), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939), while one post says **48T** [@mervenoyann](https://x.com/mervenoyann/status/2077475202775044523). The more repeated figure in this dataset is **45T**.
- Inkling includes **controllable reasoning effort** / numerical effort levels [@LiorOnAI](https://x.com/LiorOnAI/status/2077464289611563389), [@TheRundownAI](https://x.com/TheRundownAI/status/2077472283757543602), [@danielhanchen](https://x.com/danielhanchen/status/2077470080422891872).
- Tinker customers highlighted concise reasoning and strong tool calling rather than maximal raw benchmark chasing [@tinkerapi](https://x.com/tinkerapi/status/2077467634568929433), [@MichaelElabd](https://x.com/MichaelElabd/status/2077461111247712656).

### Architecture details surfaced in reactions

Several technically literate reactions extracted architectural choices from the release:

- **Hybrid/sliding-window attention** with a **5:1 local-to-global layer ratio** and **window size 512** [@eliebakouch](https://x.com/eliebakouch/status/2077463243463721085), [@ariG23498](https://x.com/ariG23498/status/2077631902228582805).
- **Relative positional encoding / relative attention bias** instead of RoPE; multiple posters called this one of the most novel large-scale choices [@stochasticchasm](https://x.com/stochasticchasm/status/2077463965438009677), [@eliebakouch](https://x.com/eliebakouch/status/2077473407550001461), [@rasbt](https://x.com/rasbt/status/2077540575255880126), [@_arohan_](https://x.com/_arohan_/status/2077519160767386030), [@ChangJonathanC](https://x.com/ChangJonathanC/status/2077508340637139318).
- **Short convolution layers** added around attention/FFN streams; commenters flagged this as unusually scaled-up usage of short convs [@eliebakouch](https://x.com/eliebakouch/status/2077463243463721085), [@stochasticchasm](https://x.com/stochasticchasm/status/2077464183994773607), [@rasbt](https://x.com/rasbt/status/2077540575255880126), [@SonglinYang4](https://x.com/SonglinYang4/status/2077492914683535850).
- **MoE with shared expert sinks / 2 shared experts**, noted as atypical since many recent MoEs use 1 shared expert [@eliebakouch](https://x.com/eliebakouch/status/2077463243463721085), [@ariG23498](https://x.com/ariG23498/status/2077631902228582805).
- **DeepSeek-style auxiliary-loss-free load balancing** was cited in community readings of the architecture [@eliebakouch](https://x.com/eliebakouch/status/2077463243463721085).
- **muP** and **Muon/weight decay variants** were inferred from the writeup and confirmed by optimizer expert reaction: Aaron Defazio said they are using his corrected weight decay approach, “MuonC/AdamC” [@aaron_defazio](https://x.com/aaron_defazio/status/2077484024726204921), while community readers also pointed out muP [@stochasticchasm](https://x.com/stochasticchasm/status/2077464183994773607), [@Laz4rz](https://x.com/Laz4rz/status/2077555045701140682).
- **8 MTP heads** for speculative decoding were highlighted by vLLM [@vllm_project](https://x.com/vllm_project/status/2077459955117109343).

### Variants

- Inkling-Small is repeatedly referenced as an upcoming or separately discussed smaller model [@LiorOnAI](https://x.com/LiorOnAI/status/2077464289611563389), [@teortaxesTex](https://x.com/teortaxesTex/status/2077458155378712673).
- Community summaries describe **Inkling-Small as 276B total / 12B active** and unexpectedly competitive versus the larger model on several evaluations [@eliebakouch](https://x.com/eliebakouch/status/2077463243463721085), [@nrehiew_](https://x.com/nrehiew_/status/2077542413133115589).


## Performance and benchmarks


### Independent benchmark framing

- Artificial Analysis said Inkling debuts at **41 on the Intelligence Index**, making it the leading U.S. open-weights release and ahead of **Nemotron 3 Ultra (38)**, **Gemma 4 31B (29)**, and **gpt-oss-120b (24)** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939).
- Artificial Analysis also said Inkling averages **25K output tokens per Intelligence Index task**, vs **43K** for **GLM-5.2 max**, **38K** for **Kimi K2.6**, and **37K** for **DeepSeek v4 Pro max**, framing it as relatively token-efficient [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939).
- Natolambert called it a “clear step up from Nemotron Ultra” and “new best American model,” but still “a bit behind GLM 5.2 on agentic benchies, and Kimi K 2.6 on multi modal” [@natolambert](https://x.com/natolambert/status/2077454404433903816).
- Design Arena said Inkling entered Agentic Web App Arena at **#9 overall, Elo 1257**, in the same band as **Claude Opus 4.6** and **Gemini 3.5 Flash**, and called it the highest-ranking U.S.-based open-weight model for agentic workloads [@DesignArena](https://x.com/DesignArena/status/2077457201216803257).
- Arena added Inkling to Agent Arena / Text / Vision / Code Arena on launch day [@arena](https://x.com/arena/status/2077476575281545573).

### Specific benchmark numbers cited

From Artificial Analysis:
- **GDPval-AA v2 Elo 1238**, higher than **Kimi K2.6 (1190)** and **DeepSeek v4 Flash max (1189)** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939).
- **τ³-Banking 24%**, above **Kimi K2.6 (21%)** and slightly above **DeepSeek v4 Flash max (23%)** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939).

### Qualitative performance takes

Positive:
- “Sharp and concise” reasoning, not rambly [@MichaelElabd](https://x.com/MichaelElabd/status/2077461111247712656).
- Strong tool calling and good long-horizon error recovery on agentic tasks [@MichaelElabd](https://x.com/MichaelElabd/status/2077461111247712656).
- Good “quality of mind” / unsycophantic flavor [@skirano](https://x.com/skirano/status/2077515605939277940), [@tinkerapi](https://x.com/tinkerapi/status/2077467634568929433).
- Alex Kirillov claimed Inkling avoids the common “audio in = intelligence penalty” seen in many omni models, though another user asked for stronger supporting evidence and benchmarks [@_alex_kirillov_](https://x.com/_alex_kirillov_/status/2077493564066722248), [@giffmana](https://x.com/giffmana/status/2077522859862139218), [@_alex_kirillov_](https://x.com/_alex_kirillov_/status/2077526541186355343).

More mixed / critical:
- Scaling01 argued the benchmarks are “not that great,” describing it as roughly “another Kimi-K2.6” and behind all closed models and GLM-5.2, speculating the release may have been timed ahead of Kimi-K3 and DeepSeek-V4-GA [@scaling01](https://x.com/scaling01/status/2077465762869194973).
- Stochasticchasm said it seems “very strong for multimodal” but “not super strong for terminal bench etc.” [@stochasticchasm](https://x.com/stochasticchasm/status/2077463420182712708).
- JJitsev pushed back on hype around “only open-weight model trained without distilling,” saying Inkling uses distillation from open weights and underperforms GLM 5.2 on TerminalBench-style evals [@JJitsev](https://x.com/JJitsev/status/2077627999352922196).
- TeortaxesTex offered a contrarian positive spin: mediocre benchmark-maxing may actually suggest less corner-cutting/distillation contamination and a more independent data pipeline [@teortaxesTex](https://x.com/teortaxesTex/status/2077483013772816426).


## Inference, systems, and launch ecosystem


### Official and partner infrastructure facts

- NVIDIA said Inkling was trained on **GB300 NVL72** and that an **NVFP4 checkpoint** was available on Hugging Face on day 0 [@NVIDIAAI](https://x.com/NVIDIAAI/status/2077456914238292220).
- vLLM said day-0 support includes **NVFP4 and BF16**, optimized for **Blackwell and Hopper**, reaching up to **380 tok/s/user on 4× GB200 with MTP** [@vllm_project](https://x.com/vllm_project/status/2077459955117109343).
- Inferact detailed system work: **sconv-aware tensor-parallel sharding**, **low-latency fused collectives (5× faster at bs=1)**, and direct integration of TML’s **FA4 sheared-bias kernel** [@inferact](https://x.com/inferact/status/2077461431306584423).
- LMSYS/SGLang said Inkling architecture support was implemented natively, including **ShortConv**, **relative positional attention**, **shared expert sink MoE**, **prefill full CUDA graph**, **MXFP8 KV cache**, **full parameter and LoRA RL in customized Megatron backend**, **routing replay**, **cross-runtime parameter sync**, and **DFlash speculative decoding from Modal** [@lmsysorg](https://x.com/lmsysorg/status/2077457150046269779).
- Modal said Inkling on Modal uses a custom **DFlash speculator** for **67% higher throughput and interactivity** [@modal](https://x.com/modal/status/2077462393441948010).
- Soumith Chintala separately amplified that Modal’s DFlash speculator is “much faster than MTP” [@soumithchintala](https://x.com/soumithchintala/status/2077500083407667569).

### Community optimization observations

- Lysandre reported replacing TML’s causal Conv1D with `causal-conv1d` yielded **+4% tok/s**, and replacing attention with **FlashAttention-4** yielded another **+11%**, for ~**15% total throughput gain** without retraining [@LysandreJik](https://x.com/LysandreJik/status/2077459011285512267).
- Unsloth released **1-bit GGUF quants** said to be **86% smaller (270GB vs 1.9TB)** while retaining **74.2% of top-1% accuracy**, with vision and audio support [@danielhanchen](https://x.com/danielhanchen/status/2077468775478423601).


## Pricing and availability

- Artificial Analysis listed Tinker pricing as:
  - **64K context**: **$1.87 / 1M input**, **$0.374 cached**, **$4.68 output**
  - **256K context**: **$3.74 / 1M input**, **$0.748 cached**, **$9.36 output**  
  [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939)
- Available on **Tinker**, **Hugging Face**, and via launch partners including **Databricks**, **Baseten**, **Modal**, **vLLM/SGLang** stacks [@soumithchintala](https://x.com/soumithchintala/status/2077457110728884327), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2077462536337891748), [@baseten](https://x.com/baseten/status/2077462904388178107), [@modal](https://x.com/modal/status/2077462393441948010).


## Facts vs opinions


### Factual claims directly supported by launch and partners

- Open weights/full weights released [@thinkymachines](https://x.com/thinkymachines/status/2077454609551921208).
- Trained from scratch [@miramurati](https://x.com/miramurati/status/2077455974743593100).
- 975B total / 41B active MoE, multimodal text-image-audio input, 1M context on weights, 256K on Tinker/API [@soumithchintala](https://x.com/soumithchintala/status/2077457110728884327), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939).
- Apache 2.0 license [@natolambert](https://x.com/natolambert/status/2077454404433903816), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2077462536337891748).
- Pretraining began last winter; agentic/coding/reasoning work started mid-January [@johnschulman2](https://x.com/johnschulman2/status/2077460227327467982).
- Day-0 support on major serving stacks, with concrete performance claims from vLLM/Inferact/Modal/NVIDIA [@vllm_project](https://x.com/vllm_project/status/2077459955117109343), [@inferact](https://x.com/inferact/status/2077461431306584423), [@modal](https://x.com/modal/status/2077462393441948010), [@NVIDIAAI](https://x.com/NVIDIAAI/status/2077456914238292220).

### Interpretations and opinions

- “Best American open model” / “saved American open-source frontier” are judgments, albeit repeated by several respected observers [@natolambert](https://x.com/natolambert/status/2077454404433903816), [@karinanguyen](https://x.com/karinanguyen/status/2077473342148448525), [@saranormous](https://x.com/saranormous/status/2077469313108422806).
- Claims that Inkling is especially important because it is not distilled from OpenAI/Anthropic are disputed. Jxmnop called it “the ONLY open-weight model” without such distillation [@jxmnop](https://x.com/jxmnop/status/2077504236380946595), then partially walked it back: “apparently they did distill lol. but only a tiny bit” [@jxmnop](https://x.com/jxmnop/status/2077540390128034133). Andrew Carr also contested the purity framing, noting use of Kimi 2.5 for SFT traces [@andrew_n_carr](https://x.com/andrew_n_carr/status/2077509786237854136).
- Claims that Inkling was “rushed” ahead of Chinese releases are speculation from critics, not evidenced by the launch materials [@scaling01](https://x.com/scaling01/status/2077465762869194973).
- Claims that relative attention gives TML a finetuning moat because backward is hard are speculative [@typedfemale](https://x.com/typedfemale/status/2077523313484832791).
- Claims that Inkling avoids multimodal intelligence loss are promising but not yet benchmark-complete in the tweet set [@_alex_kirillov_](https://x.com/_alex_kirillov_/status/2077493564066722248).


## Different perspectives


### Supportive / bullish

- **Open-weight and permissive license as strategic win:** Many saw the Apache-2.0 release as a major boost to the U.S./Western open ecosystem [@latkins](https://x.com/latkins/status/2077463764979581213), [@saranormous](https://x.com/saranormous/status/2077469313108422806), [@brexton](https://x.com/brexton/status/2077462491819302918), [@hyperindexed](https://x.com/hyperindexed/status/2077471981264396411).
- **Customization over leaderboard chasing:** Researchers and builders praised the explicit framing that Inkling is a broad, tunable foundation rather than a benchmark-maxed point solution [@gneubig](https://x.com/gneubig/status/2077468189672210472), [@ben_burtenshaw](https://x.com/ben_burtenshaw/status/2077470911448387633), [@thealexker](https://x.com/thealexker/status/2077540344757928445).
- **Strong release quality:** Several users praised the transparency, grounded tone, and comprehensive technical documentation [@lvwerra](https://x.com/lvwerra/status/2077487456270586319), [@saranormous](https://x.com/saranormous/status/2077483301212963157), [@rasbt](https://x.com/rasbt/status/2077540575255880126).
- **Architecture interest:** The non-RoPE positional choice and scaled short-conv usage drew positive attention as evidence TML is willing to make meaningful architecture bets [@stochasticchasm](https://x.com/stochasticchasm/status/2077463965438009677), [@rasbt](https://x.com/rasbt/status/2077540575255880126), [@ChangJonathanC](https://x.com/ChangJonathanC/status/2077508340637139318).

### Neutral / analytical

- **Strong but not top overall:** The most balanced reads place Inkling as the new U.S. open-weight leader, but behind GLM/Kimi/DeepSeek or top closed models on some fronts [@natolambert](https://x.com/natolambert/status/2077454404433903816), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2077466590346444939), [@stochasticchasm](https://x.com/stochasticchasm/status/2077463420182712708).
- **Good base model thesis:** Multiple analysts read the release as a systems/business move: ship a solid, efficient, post-trainable base and let Tinker plus downstream RL/fine-tuning create differentiation [@ben_burtenshaw](https://x.com/ben_burtenshaw/status/2077470911448387633), [@kimmonismus](https://x.com/kimmonismus/status/2077472478499053846), [@tinkerapi](https://x.com/tinkerapi/status/2077467634568929433).

### Critical / skeptical

- **Not frontier overall:** Critics argued it is still clearly behind top Chinese open-weight models and the strongest closed models [@scaling01](https://x.com/scaling01/status/2077465762869194973), [@JJitsev](https://x.com/JJitsev/status/2077627999352922196).
- **Purity claims overstated:** Some pushback focused on exaggerated claims that it is uniquely “pure” or non-distilled; the thread set includes both hype and corrections [@jxmnop](https://x.com/jxmnop/status/2077504236380946595), [@jxmnop](https://x.com/jxmnop/status/2077540390128034133), [@andrew_n_carr](https://x.com/andrew_n_carr/status/2077509786237854136), [@JJitsev](https://x.com/JJitsev/status/2077627999352922196).
- **Benchmark middlingness as concern:** Some readers saw the moderate benchmark profile as evidence it may simply lag current Chinese open frontier rather than inaugurate a new frontier [@scaling01](https://x.com/scaling01/status/2077465762869194973).


## Context: why this matters

- **First major TML public model:** This is the first true external model release from Thinking Machines after months of anticipation around a lab staffed by ex-OpenAI leaders and researchers. That made the choice of **open weights** itself notable [@Hesamation](https://x.com/Hesamation/status/2077456283528045001), [@TechCrunch](https://x.com/TechCrunch/status/2077454757283959123).
- **A U.S. open-weight answer to Chinese momentum:** Many reactions explicitly compare Inkling to GLM, Kimi, DeepSeek, and Qwen. The release lands amid concern that Western open-weight models have trailed Chinese ones on capability and release cadence [@scaling01](https://x.com/scaling01/status/2077474933370761345), [@teortaxesTex](https://x.com/teortaxesTex/status/2077457960385585281), [@sriramk](https://x.com/sriramk/status/2077566845431779766).
- **Open base + post-training stack thesis:** TML’s messaging strongly suggests a strategy similar to “ship a competent open substrate, then differentiate via customization/fine-tuning/RL infrastructure.” That aligns with Tinker distribution and with user reactions centering controllable reasoning, concise outputs, and adaptation rather than raw leaderboard supremacy [@thinkymachines](https://x.com/thinkymachines/status/2077454609551921208), [@MichaelElabd](https://x.com/MichaelElabd/status/2077461111247712656), [@ben_burtenshaw](https://x.com/ben_burtenshaw/status/2077470911448387633).
- **Inference ecosystem maturity:** The release also showcases how far open inference stacks have come. Day-0 support for a 1T-class multimodal MoE with new architectural components and multiple kernel-level optimizations would have been far less plausible a year earlier [@vllm_project](https://x.com/vllm_project/status/2077459955117109343), [@inferact](https://x.com/inferact/status/2077461431306584423), [@LysandreJik](https://x.com/LysandreJik/status/2077459011285512267).
- **Architectural experimentation at scale:** Relative positional bias instead of RoPE and large-scale short-conv usage are the kind of choices researchers watch closely because they may indicate future architecture trends if they prove robust under scaling and post-training [@stochasticchasm](https://x.com/stochasticchasm/status/2077463965438009677), [@rasbt](https://x.com/rasbt/status/2077540575255880126), [@ChangJonathanC](https://x.com/ChangJonathanC/status/2077508340637139318).
- **Release style as signal:** Several commentators praised the unusually restrained release language, explicit admission that it is not the strongest overall model, and detailed technical notes. For expert audiences, that improved credibility relative to more benchmark-maxed launches [@eliebakouch](https://x.com/eliebakouch/status/2077463243463721085), [@lvwerra](https://x.com/lvwerra/status/2077487456270586319), [@thealexker](https://x.com/thealexker/status/2077540344757928445).


**Agents, Sandboxes, and Harness Engineering**

- **Perplexity’s SPACE sandbox platform**: [@perplexity_ai](https://x.com/perplexity_ai/status/2077432518081744979) introduced **SPACE**, its in-house sandbox platform now serving **100% of Computer production traffic**. The interesting systems design choice is the decoupling of **session state** from disposable **Firecracker microVM sandboxes**, with rolling snapshots allowing pause/resume/branch semantics. [@perplexity_ai](https://x.com/perplexity_ai/status/2077432569432514977) reported median sandbox creation latency dropping from **185 ms to 60 ms** and P90 from **447 ms to 89 ms**, while [@zbraniecki](https://x.com/zbraniecki/status/2077451060927672647) explained the use of **disk snapshots plus full VM checkpoints**, object storage for resumability, and **Btrfs COW** to make sandbox creation a metadata operation instead of full image copy. This is one of the more concrete production infrastructure disclosures in the set.
- **Agent workspaces, Slack-native agents, and harness cost**: On the product side, [@istdrc](https://x.com/istdrc/status/2077376131628707907) launched **Raft 1.0**, positioning it as a shared workspace where agents behave more like a team in a messaging app than isolated terminal sessions. [@LangChain](https://x.com/LangChain/status/2077437626965971059) upgraded **Fleet in Slack**, enabling one-click deployment of agents into channels/threads with custom identity and file handoffs, echoed by [@hwchase17](https://x.com/hwchase17/status/2077443161585287290). On the engineering side, [@AI21Labs](https://x.com/AI21Labs/status/2077399596439925073) argued that **harness design, not just model choice**, materially affects cost: it cited Writer’s “Harness Effect” as showing **41% lower cost per task** at quality parity when only orchestration changed, and linked to its own early-stopping work claiming **up to 44% compute reduction** for SWE agents. Related toolchain notes came from [@nutlope](https://x.com/nutlope/status/2077432463685554558), who launched **TogetherLink** to run open-source models inside coding harnesses like Codex and Claude Code, and [@Teknium](https://x.com/Teknium/status/2077424392892731396), who added **Blender MCP** to the Hermes agent catalog.

**Automated Red Teaming, Alignment, and Governance Friction**

- **OpenAI’s GPT-Red and the safety flywheel**: [@OpenAI](https://x.com/OpenAI/status/2077446718728425686) introduced **GPT-Red**, an internal automated red teamer for finding **prompt injection vulnerabilities at scale**. The most concrete claim was that adversarial training against GPT-Red made **GPT-5.6 Sol** substantially more robust, with [OpenAI](https://x.com/OpenAI/status/2077446722683650525) saying replayed strong attacks produced **6× fewer failures** than its best production model from four months earlier. The broader framing—AI systems improving the safety of future AI systems—was made explicit in [OpenAI’s follow-up](https://x.com/OpenAI/status/2077446723992228167). This sits alongside external commentary from [@omarsar0](https://x.com/omarsar0/status/2077450923295506505), who called it a high-ROI self-improvement loop.
- **Anthropic’s misalignment scenarios and DeepMind governance debate**: [@AnthropicAI](https://x.com/AnthropicAI/status/2077452646303006927) published **“Agentic misalignment in Summer 2026”**, adding four new simulated scenarios of bad autonomous-agent behavior a year after its blackmail case studies. Concurrently, governance discussion intensified around Google DeepMind: [@Turn_Trout](https://x.com/Turn_Trout/status/2077448610157891734) announced he resigned from DeepMind over military use without restrictions against killer robots or mass surveillance, while [@jackclarkSF](https://x.com/jackclarkSF/status/2077419516452065406) and [@Yoshua_Bengio](https://x.com/Yoshua_Bengio/status/2077487556325732745) amplified Demis Hassabis’s call for **third-party testing and standards** feeding into policy. The juxtaposition—public support for standards versus internal dissatisfaction over governance practice—was succinctly noted by [@BlackHC](https://x.com/BlackHC/status/2077511235763884426).

**Benchmarks, Reproducibility, and Evaluation Integrity**

- **Soofi S / Nemotron contamination dispute**: The sharpest evaluation controversy concerned claims around **Soofi S 30B-A3B**. [@kimmonismus](https://x.com/kimmonismus/status/2077382976577343913) presented it as a Europe-trained model based on NVIDIA’s open **Nemotron 3 Nano** architecture, trained on **~27T tokens** with German upweighting and a fully released recipe. But multiple critics challenged both novelty and eval integrity. [@JJitsev](https://x.com/JJitsev/status/2077273171963588804) argued the comparison lowered Nemotron reference scores versus the original report, while [@eliebakouch](https://x.com/eliebakouch/status/2077425801633427919) alleged the training mix included **light rephrasings of the GPQA Diamond eval set**, potentially contaminating the benchmark and overstating the gap. He later summarized the concern as “**10 epochs of a very light rephrasing of every GPQA Diamond eval item**” in [a follow-up](https://x.com/eliebakouch/status/2077428860639973471). Even skeptics suggested a straightforward remedy: rerun the original Nemotron recipe and then compare under identical evaluation conditions, as [@JJitsev](https://x.com/JJitsev/status/2077395737109725271) proposed.
- **Broader movement toward better evals and reproducibility**: Several posts pushed on eval methodology more generally. [@arena](https://x.com/arena/status/2077432293023678685) launched a **factuality-weighted ranking** combining human preference with claim verification, based on **2M+ labeled claims** across text and search arenas; GPT-5.5 reportedly gained the most under the factuality weighting while some preference-optimized models dropped. [@askalphaxiv](https://x.com/askalphaxiv/status/2077415909652901993) and [@abidlabs](https://x.com/abidlabs/status/2077518437161521533) kicked off a Hugging Face-backed reproducibility challenge around **ICML 2026** papers, with early community progress already reproducing dozens of papers. And [@sayashk](https://x.com/sayashk/status/2077420320172941683) announced a PhD talk explicitly titled **“The Missing Science of AI Evaluation.”**

**Top tweets (by engagement)**

- **Inkling launch**: [@thinkymachines announcing Inkling](https://x.com/thinkymachines/status/2077454609551921208) was the clear technical story of the day, combining an open-weight **~1T multimodal MoE** release with a broad open inference rollout.
- **OpenAI’s GPT-Red**: [@OpenAI’s GPT-Red announcement](https://x.com/OpenAI/status/2077446718728425686) stood out for substance: automated prompt-injection red teaming plus a concrete **6× robustness improvement** claim on GPT-5.6 Sol.
- **Claude Code artifacts + MCP**: [@ClaudeDevs](https://x.com/ClaudeDevs/status/2077489907350856038) launched **artifacts that can call MCP connectors**, effectively making artifacts per-viewer live apps/dashboards with permission-scoped data access.
- **Perplexity SPACE**: [@perplexity_ai](https://x.com/perplexity_ai/status/2077432518081744979) and [@AravSrinivas](https://x.com/AravSrinivas/status/2077439693420163352) provided unusually detailed production numbers for agent sandboxes, including **5× faster tail latency** claims.
- **DeepMind governance resignation**: [@Turn_Trout’s resignation thread](https://x.com/Turn_Trout/status/2077448610157891734) was one of the most engaged policy-adjacent AI posts, centered on military-use restrictions and lab governance credibility.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Bonsai 27B and Local Inference Speedups

  - **[Bonsai 27B: 1-bit dense LLM running locally in your browser using custom WebGPU kernels](https://www.reddit.com/r/LocalLLaMA/comments/1uwfva9/bonsai_27b_1bit_dense_llm_running_locally_in_your/)** (Activity: 731): ****PrismML** released **Bonsai 27B**, a **1-bit dense LLM** intended to run locally in-browser via custom **WebGPU kernels**, with model artifacts on [Hugging Face](https://huggingface.co/collections/prism-ml/bonsai-27b) and a [WebGPU demo Space](https://huggingface.co/spaces/webml-community/bonsai-webgpu-kernels). The claimed compression is from roughly `54GB` to `3.8GB` (`-93%`) while retaining about `90%` of baseline capability; commenters note a `~5.7GB` footprint for a Qwen/Qwen3-derived `27B` variant and interest in testing on consumer GPUs such as an `8GB` RTX 3070 laptop GPU.** Commenters are broadly positive but focus on scaling questions: whether 1-bit quantization can make `80–100B` parameter models practical, and what parameter/context-length tradeoff would fit `256k+` context on a single `24GB` GPU. There is also a recurring view that recent 1-bit releases suggest a broader shift toward ultra-low-bit LLM deployment.

    - Commenters highlighted the headline compression claim: **Bonsai 27B / Qwen 3.6 27B-class model** reportedly fits in about `5.7GB` at **1-bit density** with only ~`5%` capability loss. One user specifically planned to test it on an `8GB` laptop RTX 3070, implying the main practical interest is whether custom **WebGPU kernels** can make a 27B dense model usable on consumer VRAM budgets.
    - A technically focused thread discussed scaling: users want an `80B–100B` 1-bit dense model, but noted the real constraint is fitting both weights and a `256k+` context KV/cache footprint on a single `24GB` GPU. This frames 1-bit weights as only part of the memory story; long-context inference may dominate usable deployment limits even if model weights are highly compressed.
    - One commenter distinguished **1-bit models trained from scratch** from extreme post-training quantization, arguing the former should retain much more capability than simply quantizing a larger model down to 1 bit. They suggested a future **1-bit 70B** trained natively at that precision could be both consumer-GPU runnable and practically useful, unlike many heavily quantized ultra-low-bit models.

  - **[PrismML’s new Ternary Qwen3.6 27B runs near fp16 precision on 10GB of memory!!!](https://www.reddit.com/r/LocalLLaMA/comments/1uwehzt/prismmls_new_ternary_qwen36_27b_runs_near_fp16/)** (Activity: 465): ****PrismML** released **[Bonsai 27B](https://prismml.com/news/bonsai-27b)**, a ternary/BitNet-style variant of **Qwen3.6 27B**, with **[GGUF](https://huggingface.co/collections/prism-ml/bonsai-27b)** and MLX builds requiring PrismML forks of **[llama.cpp](https://github.com/PrismML-Eng/llama.cpp)** / **[mlx](https://github.com/PrismML-Eng/mlx)** for now. The OP reports ~`10GB` memory use at `32K` context on an M4 Pro and claims the model is much better than a conventional 2-bit quant, but later edits clarify it is **better than Q2, worse than Q4_K_XL**, with observed hallucinations and tool-calling loops; the main value is memory footprint rather than fp16-equivalent accuracy. Claimed capabilities include `256K` context and multimodal input, with **[dFlash](https://github.com/z-lab/dflash)** support mentioned as forthcoming; the whitepaper is [here](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf).** Commenters pushed back on the wording *“near fp16 precision”* because ternary weights are `{-1,0,1}`, and criticized hype/AGI framing as technically misleading. One technical question raised was whether the same ternary approach could scale to larger models such as GLM 5.2 while preserving acceptable quality loss.

    - A commenter challenged the claim that a **ternary / 1-trit Qwen3.6 27B** can run at *“near fp16 precision”*, noting that PrismML’s premise is extremely low-bit representation and asking what that phrase technically means in this context. They also objected to loose use of *AGI*, arguing that even stronger models like **Mythos** should not be labeled that way without evidence.
    - One user reported a working local run of **`Ternary-Bonsai-27B-Q2_0.gguf`** via a PrismML fork of `llama.cpp`, using `-ngl 99` and a very large `-c 200000` context. On a simple TypeScript explanation prompt, they observed **`243.5 t/s` prompt processing** and **`89.1 t/s` generation**, suggesting the model is at least operational and fast under their setup, though the example does not validate quality claims.
    - Another commenter asked for rigorous benchmarks such as **SciCode** or **SWE-rebench** to support the claims that the ternary model is close to BF16 and *“far more intelligent”* than comparable **2–3 bit Qwen3.6 27B** quantizations. They also asked whether the method works for **MoEs** or larger dense models like **Mistral Medium 3.5**, noting that very large MoEs with `>10B` active parameters, such as **Step 3.7 Flash** and **MiMo V2.5**, often appear unusually resistant to quantization.

  - **[Apple in talks with startup PrismML that shrinks AI models to run on an iPhone](https://www.reddit.com/r/LocalLLaMA/comments/1ux4cn2/apple_in_talks_with_startup_prismml_that_shrinks/)** (Activity: 362): **[CNBC reports](https://www.cnbc.com/2026/07/14/apple-prismml-ai-compression-iphone.html) **Apple** is in early talks with **PrismML**, a Caltech spinout, over extreme LLM compression for on-device iPhone inference. PrismML reportedly compresses **Alibaba Qwen `27B`** from ~`54 GB` to **<`4 GB`** via ternary/binary-style quantization, claiming `10–15×` lower memory, `6–8×` faster responses, and `3–6×` lower energy use, with some degradation in factual recall. Technical commenters noted missing details around conversion cost, convergence guarantees when moving to ternary/binary weights, whether distillation is used, saturation of post-training, and comparisons to prior low-bit work such as **BitCPM-CANN** and `bitnet-b1.58-2B-4T`.** Commenters were skeptical that small size alone proves usefulness, with one asking whether the compressed model is *“actually capable of anything useful.”* Another user expressed disappointment because they had just been testing PrismML’s `q1 bonsai` model, implying concern about Apple potentially acquiring or restricting the technology.

    - Commenters noted that PrismML’s public materials appear to lack key technical details needed to evaluate its compression/quantization claims: conversion cost, whether ternary/binary conversion reliably reaches a convergence point, whether distillation is used, and whether post-training has fully saturated the model. One comparison raised was the **BitCPM-CANN** tech report, with commenters arguing that there is little precedent for a properly trained-from-scratch ternary model beyond **`bitnet-b1.58-2B-4T`**.
    - Several comments questioned whether PrismML’s advertised small model size translates into practical capability. One user said they had been testing the company’s **`q1 bonsai`** model, while another noted they had seen repeated claims about compactness but little evidence that the model is *“actually capable of anything useful.”*

  - **[ExLlamaV3 v1.0.0 - Major Performance Upgrades](https://www.reddit.com/r/LocalLLaMA/comments/1uwylut/exllamav3_v100_major_performance_upgrades/)** (Activity: 422): **The image is a **technical benchmark table** ([PNG](https://i.redd.it/ej7102hqfcdh1.png)) for **ExLlamaV3 v1.0.0**, showing RTX 3090 decode throughput gains versus `v0.0.43` across multiple quantized LLMs/bitrates. It supports the release claims of major inference-kernel upgrades: `v1.0.0 mul1` shows large speedups such as **Qwen 3.6 27B** rising from `29` to `50 tok/s` (`+72%`) and **Qwen 3.5 0.8B** from `268` to `444 tok/s` (`+66%`), consistent with the post’s notes about new attention, GEMM/GEMV, INT8 GEMV, Conv1D, and MoE scheduler kernels.** Comments are mostly appreciative rather than deeply technical, highlighting that ExLlamaV3 is an NVIDIA-GPU-focused LLM engine using the EXL3 format rather than GGUF/llama.cpp, and praising the scale of the work by Turboderp/Fable.

    - A commenter clarified that **ExLlamaV3/ExLlama3** is an LLM inference engine targeting the specialized **EXL3** model format rather than common **GGUF** files used by `llama.cpp`, and that it is currently **NVIDIA GPU-only**. This distinction matters for deployment compatibility: users with existing GGUF workflows may need separate quantized weights and CUDA-capable hardware to use it.
    - One technical request focused on **TabbyAPI** integration, specifically improving tool-calling compatibility with **Claude Code** so ExLlamaV3 could be used locally in that workflow. The same commenter noted they are currently using **GGUF with MTP** and are interested in evaluating the quality tradeoffs of **EXL3 quantization** versus their current setup.


### 2. Open-Weight Model Launches and Updates

  - **[Thinking Machines releases first open-weight model “Inkling”](https://www.reddit.com/r/LocalLLaMA/comments/1uxdv34/thinking_machines_releases_first_openweight_model/)** (Activity: 1082): **The [image](https://i.redd.it/d7s0z8kqpfdh1.jpeg) is an AI model leaderboard highlighting **Thinking Machines’ first open-weight model, Inkling**, scoring `1257`, roughly mid-table and tied with **Claude Opus 4.6**, below **GPT-5.6 Sol** and well below top entries like **Claude Sonnet 5** at `1333`. From the announcement/comments, Inkling is described as a **MoE transformer** with `975B` total parameters, `41B` active parameters, up to a `1M` token context window, and pretraining on `45T` multimodal tokens spanning text, images, audio, and video; a preview **Inkling-Small** has `12B` active parameters for lower cost/latency.** Commenters were interested because Thinking Machines is associated with a former OpenAI CTO and is releasing open weights, but some were skeptical that Inkling will see broad adoption if it does not outperform competing open models such as **GLM-5.2**.

    - **Inkling** is described as a sparse MoE transformer with `975B` total parameters, `41B` active parameters, a `1M` token context window, and pretraining on `45T` tokens spanning text, images, audio, and video. One commenter compared it unfavorably to **GLM-5.2**, arguing that despite being multimodal and long-context, it may see limited adoption if it does not outperform that open-weight competitor.
    - The most technically interesting discussion centered on **Inkling-Small**: a `276B` parameter MoE with only `12B` active parameters, positioned as a lower-latency/lower-cost variant. Commenters highlighted that it reportedly *“matches or exceeds its larger sibling on many benchmarks”* due to improvements in the pretraining data mix and recipe, making it potentially practical for local inference compared with the `41B` active main model: https://thinkingmachines.ai/news/introducing-inkling/#inkling-small
    - A few commenters noted gaps in the model-size lineup: there is no model around the `30B` dense/active-parameter class, which some local-inference users consider a useful middle ground. The release instead jumps from **Inkling-Small** at `12B` active MoE to the main **Inkling** at `41B` active MoE.

  - **[German AI consortium releases Soofi S, an open 30B model that tops benchmarks in both English and German](https://www.reddit.com/r/LocalLLaMA/comments/1uxao7y/german_ai_consortium_releases_soofi_s_an_open_30b/)** (Activity: 321): ****Soofi S** is presented as a German-led fully pretrained MoE LLM, **`31.6B` total / ~`3.2B` active parameters**, based on **NVIDIA Nemotron 3 Nano’s hybrid Mamba-2/Transformer architecture**, trained on ~`27T` tokens with German upweighting and claimed near-flat throughput from `4K` to `256K` context ([article](https://the-decoder.com/german-ai-consortium-releases-soofi-s-an-open-30b-model-that-tops-benchmarks-in-both-english-and-german/), [paper](https://arxiv.org/abs/2607.09424)). Commenters note it is described as a **new full pretraining run, not a finetune**, with unusually transparent artifacts including [W&B training logs](https://wandb.ai/soofi-exchange/pretrain-nemotron-3-nano-on-20T-4/reports/Soofi-S-Pretraining--VmlldzoxNzM4NTQ4NA?accessToken=c6mcvzhsloyc1v4duq9c7eq9aa81sr6b8j1l6yju6sbyz1skgecggj1pun9qxb52), [training scripts](https://github.com/soofi-project/Soofi-Pretraining), and gated [GGUF](https://huggingface.co/Soofi-Project/Soofi-S-Instruct-Preview-GGUF) / [reasoning GGUF](https://huggingface.co/Soofi-Project/Soofi-S-Rhine-Preview-GGUF) releases. Technical caveats raised include limited modern long-context evaluation beyond RULER, possible German naturalness issues from *“machine-translated and synthetically generated German texts”*, and benchmark ambiguity because coding/math tasks and German understanding may inflate language-quality claims; one commenter also claims Qwen3.5 35B-A3B beats Soofi S on German benchmarks despite not being German-specialized.** The main debate is whether the benchmark comparison is credible: commenters criticized omission of newer baselines such as Qwen 3.6/Gemma 4 and comparison against older models. Another concern is licensing: marketing claims *“sovereign, open source… license-free availability”* conflict with a Hugging Face card using a custom **“Other”** license whose full text was reportedly missing.

    - A commenter notes Soofi S is described as a **newly pretrained model rather than a finetune**, with architecture based on **Nemotron 3 Nano** and full pretraining/additional phases documented in the [paper](https://arxiv.org/abs/2607.09424), [W&B training logs](https://wandb.ai/soofi-exchange/pretrain-nemotron-3-nano-on-20T-4/reports/Soofi-S-Pretraining--VmlldzoxNzM4NTQ4NA?accessToken=c6mcvzhsloyc1v4duq9c7eq9aa81sr6b8j1l6yju6sbyz1skgecggj1pun9qxb52), and [training scripts](https://github.com/soofi-project/Soofi-Pretraining). They caution that the Nemotron-derived architecture may limit long-context accuracy: the release includes a **RULER** test, but apparently lacks more modern long-context evaluations.
    - Several commenters question the benchmark framing, arguing Soofi S was compared against older baselines and not newer models like **Qwen 3.6** or **Gemma 4**. One commenter highlights that the authors’ own results show **Qwen3.5 35B-A3B** outperforming Soofi S on German, suggesting the benchmark may measure broad understanding, math, and coding more than native-quality German generation.
    - The data mix is flagged as potentially problematic because the paper mentions *“machine-translated and synthetically generated German texts”*, which can produce unnatural German and may affect generation quality despite benchmark gains. Release artifacts also appear incomplete or inconsistent: GGUF builds exist for [Soofi-S-Instruct-Preview](https://huggingface.co/Soofi-Project/Soofi-S-Instruct-Preview-GGUF) and reasoning variants like [Soofi-S-Rhine-Preview](https://huggingface.co/Soofi-Project/Soofi-S-Rhine-Preview-GGUF), but they are gated, and the license is described as custom/“Other” with the full text apparently missing.

  - **[KAT-Coder-Air V2.5 - Open model soon](https://www.reddit.com/r/LocalLLaMA/comments/1uwbe7w/katcoderair_v25_open_model_soon/)** (Activity: 262): **The [image](https://i.redd.it/eob36vaek7dh1.png) is a screenshot of a **KwaiAI/KAT-Coder** social post announcing **KAT-Coder-Pro V2.5** and, more importantly for r/LocalLLaMA, a reply stating that **KAT-Coder-Air V2.5 will be open-sourced “very soon.”** The post links to availability on **OpenRouter** and a technical report on arXiv ([abstract](https://arxiv.org/abs/2607.05471), [PDF](https://arxiv.org/pdf/2607.05471)), with claims around long-horizon and agentic coding performance; a commenter who tested it via OpenRouter speculates it is **below `100B` parameters**.** Comments are mostly curiosity-driven: users are waiting for actual open weights and asking about the model size, with one tester noting it is already usable through OpenRouter but not yet confirming architecture or parameter count.

    - A commenter who says they asked about the weights reports that **KAT-Coder-Air V2.5 is already available on OpenRouter** and, based on their usage/metadata, *“should be below `100B` parameters.”* The main technical uncertainty raised in the thread is the model’s exact parameter count/size, which users are waiting to verify once weights are released.

  - **[Google is updating Gemma 4's chat templates, bringing major fixes to tool calling and reducing "laziness", and enabling Flash Attention 4 on Hopper GPUs, plus an interactive guide on how to work with and improve its vision!](https://www.reddit.com/r/LocalLLaMA/comments/1uxfu4k/google_is_updating_gemma_4s_chat_templates/)** (Activity: 513): ****Google Gemma** announced updates to **Gemma 4** chat templates via [X](https://x.com/googlegemma/status/2077449152062247219), targeting tool-calling correctness and reduced “laziness,” while also enabling **Flash Attention 4 on Hopper GPUs** and publishing an interactive Gemma vision token-budget guide on [Hugging Face Spaces](https://huggingface.co/spaces/google/gemma4_vision_token_budget). A commenter linked the relevant [`google/gemma-4-31B-it` commit](https://huggingface.co/google/gemma-4-31B-it/commit/68abe48010cbe15293462fa11e901a60639a44e5), which includes fixes for `null` handling, reasoning/thinking preservation, turn-tag balancing, tool-response continuation, `add_generation_prompt` regression, extra `<turn|>` emission, and tool-call-only turn closure; notably, `preserve_thinking` is restored/defaulted and scoped around tool-call turns.** Commenters framed the update as addressing confusing prior behavior in Gemma 4 prompting/tool use, with one saying they had assumed the failures were user error. The most emphasized reaction was enthusiasm that **`preserve_thinking`** support was included.

    - A commenter enumerated the Hugging Face commit-level fixes for **Gemma 4 31B IT** chat templates, including null handling, reasoning preservation, turn-tag balance, input validation, restoration of model turns/thinking cues after tool responses, and fixes for extra `<turn|>` emission in assistant content + tool-call continuation paths. The linked commit list starts from [`68abe480`](https://huggingface.co/google/gemma-4-31B-it/commit/68abe48010cbe15293462fa11e901a60639a44e5), with notable fixes around `preserve_thinking`, rendering the thinking channel independent of `tool_calls`, and correcting tool-call-only turn closure.
    - One technical takeaway from the thread is that **tool-calling behavior appears to have been strongly affected by template-level serialization bugs**, especially around preserving reasoning/thinking channels and correctly reopening assistant generation after tool responses. However, another commenter reports that the perceived "laziness" remains present with the latest template, arguing it is likely a **model behavior issue rather than a chat-template issue**.


### 3. Open-Model Policy and Self-Hosting Risk

  - **[Source: the Trump administration and industry groups discussed streamlining US open model releases of equal or lesser capability to leading Chinese open models](https://www.reddit.com/r/LocalLLaMA/comments/1uw9ucd/source_the_trump_administration_and_industry/)** (Activity: 573): **A reported Trump administration/industry discussion would streamline U.S. releases of **open models** whose capabilities are *equal to or below* leading Chinese open models, motivated by concern over U.S. developers adopting Chinese local/open-weight models. The linked source could not be verified from the provided archive page because it returned a CAPTCHA/HTTP `429` interstitial rather than article content.** Commenters argued that U.S. AI firms have weak incentives to release Chinese-competitive open models because strong local models could cannibalize paid API/SaaS revenue. Others were skeptical of claims about Chinese open models containing CCP-exploitable backdoors, noting that if such model-level backdoors were practical or detectable, U.S. labs would likely already dominate open-weight alternatives.

    - Several commenters argued that **banning capable open-weight models is technically unenforceable** once weights are mirrored globally: a single torrent tracker, private file share, or USB transfer can bypass restrictions. One commenter noted enforcement would likely require extreme downstream controls such as confiscating or restricting GPUs/workstations above certain VRAM thresholds, because inference can run locally once weights are obtained.
    - A recurring technical policy argument was that if the US wants companies to avoid local Chinese models, the practical alternative is to release **US open-weight models of equal or better capability** rather than restrict access. Commenters suggested this would need to match the quality trajectory of Chinese open models closely enough that local deployment users do not need to rely on foreign weights.
    - One commenter specifically cited **NVIDIA Nemotron** as a better model-release pattern because of its relatively transparent training data/process, while noting that **Nemotron 3 Ultra** is “very good” but appears undertrained and therefore underperforms for its parameter size. Multiple commenters were skeptical of claims about intentional Chinese “backdoors” in open-weight models, arguing that if such hidden model-level backdoors were straightforward to weaponize, US labs would already be exploiting the same technique in open models.

  - **[Some of y'all wonder why anyone would self host AI.  Would you accept the opinion of the CEO of Microsoft?](https://www.reddit.com/r/LocalLLaMA/comments/1uwqgqs/some_of_yall_wonder_why_anyone_would_self_host_ai/)** (Activity: 559): **The post argues for **self-hosting AI/LLMs** as a data-protection strategy, citing a [TechCrunch article](https://techcrunch.com/2026/07/13/satya-nadella-has-issued-a-shocking-warning-to-companies-using-ai/) quoting **Microsoft CEO Satya Nadella** that enterprises may *“pay for intelligence twice”*: once in fees and again by exposing proprietary business knowledge needed to make hosted AI useful. The technical concern is that API/SaaS model providers such as **OpenAI** or **Anthropic** could ingest sensitive prompts, documents, workflows, or RAG corpora and potentially derive competitive intelligence, making local inference or private deployment attractive for inventors, researchers, and enterprises handling IP-sensitive data.** Top comments were skeptical of Nadella’s framing, arguing it may be a sales pitch for **Azure-hosted** AI rather than a neutral privacy warning. Others pointed to Microsoft’s own products—**Copilot**, the OpenAI partnership, and screen-indexing features like Recall—as evidence that Microsoft has similar incentives and privacy risks.

    - Commenters interpreted Satya Nadella’s self-hosting argument as less about pure decentralization and more about **enterprise workloads moving onto Microsoft Azure**, i.e. companies “hosting” private models inside Microsoft’s cloud rather than running them fully on-prem.
    - Several commenters tied the self-hosting/privacy discussion to Microsoft’s own AI products, especially **Copilot** and the controversial Windows **Recall** feature, which was criticized for continuously snapshotting user activity for later AI-powered indexing. The technical concern raised was that cloud-connected assistants create a large attack surface for sensitive business data, even when marketed as productivity tooling.
    - A recurring technical concern was that enterprise AI vendors may use customer interactions or documents as training/evaluation data, with one commenter asking how labs continue acquiring `50–100T` new tokens per year for model training. The point was framed as a reason businesses might prefer self-hosted or tightly controlled deployments to reduce IP leakage and data reuse risk.

  - **[This is why we need local models and opensource harnesses](https://www.reddit.com/r/LocalLLM/comments/1uweb90/this_is_why_we_need_local_models_and_opensource/)** (Activity: 379): **The image ([screenshot](https://i.redd.it/ebfetgbo68dh1.jpeg)) shows a claim by **International Cyber Digest** that **xAI’s Grok Build CLI** uploaded entire Git repositories—including private code and unredacted secrets—to a **Google Cloud bucket**, allegedly disabled later via a hidden server-side flag. The post uses this alleged leak to argue for **local-first/open-weight models**, deterministic open-source agent harnesses, private VPC execution, and governance layers that can inspect/redact secrets before any third-party network egress.** Commenters were overwhelmingly skeptical of cloud-tethered coding agents, framing this as potential spyware/malware behavior and suggesting legal accountability if the exfiltration was intentional. One recurring sentiment was that users should expect poor data-handling practices from opaque vendor-controlled AI tooling.

    - A commenter describes mitigating LLM-driven code/data exfiltration by using a **self-hosted Git server blocked from the Internet**. They note this does not fully prevent leakage, but it changes the threat model: instead of a tool silently issuing direct upload/download jobs, any exfiltration would need to pass through the LLM interaction path, making it *more visible and slower*.
    - Another technically relevant takeaway is the preference for **open-source agents plus open-weight models on self-controlled infrastructure**. The argument is that local execution and inspectable harnesses reduce reliance on opaque hosted services where telemetry, tool calls, or data-retention behavior may be difficult to audit.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo


### 1. Frontier AI Distillation and Safety Standards

  - **[Anthropic just told the US Senate that Alibaba ran 25,000 fake accounts and had 28.8 million conversations with Claude — not to use it, but to copy it](https://www.reddit.com/r/ChatGPT/comments/1uwavzo/anthropic_just_told_the_us_senate_that_alibaba/)** (Activity: 3524): **The post claims **Anthropic told the U.S. Senate** that **Alibaba** used `25,000` API accounts to run `28.8M` Claude conversations over ~six weeks (Apr–Jun) to perform large-scale model distillation—extracting agentic reasoning/coding behavior via normal API access rather than hacking—and then use it to improve **Qwen**. Anthropic is described as framing this as its largest “distillation attack” and arguing current law is ambiguous enough that it sought congressional action rather than litigation; the OP links a longer breakdown tying this to the **Fable 5 export ban**: [YouTube](https://youtu.be/g1d3yTR6E2Y).** Top comments debate whether this is meaningfully different from conventional reverse engineering—e.g., automakers or Samsung buying competitors’ products to study them—versus prohibited model extraction. Several commenters argue Anthropic’s complaint is hypocritical because frontier labs trained on large amounts of public/creative human output under fair-use theories, but object when their own model outputs are used as training data.

    - Several commenters framed **model distillation via API outputs** as analogous to competitive reverse engineering: an automaker or **Samsung** buying a rival product, studying behavior, and using the findings to improve its own system. The technical distinction raised is that AI distillation can automate this process at scale—here alleged as `25,000` accounts and `28.8M` Claude conversations—rather than relying on human engineers manually extracting design lessons.
    - One technically relevant objection was attribution: because **Alibaba** operates major cloud infrastructure, traffic originating from Alibaba-associated networks or accounts does not necessarily prove Alibaba corporate teams performed the alleged distillation. Commenters compared this to seeing suspicious requests from **AWS** or **Azure** IP ranges, which may indicate customer activity rather than action by **Amazon** or **Microsoft** themselves.
    - A recurring implementation/security point was that if the interactions occurred through paid Claude access and within apparent API mechanics, Anthropic’s complaint may expose weaknesses in **abuse detection, account-linking, rate limits, ToS enforcement, or anti-distillation controls**. One commenter summarized this as: *“they got paid and the API was used according to the TOS?”*—suggesting the issue may be more about platform controls than unauthorized access.

  - **[Demis Hassabis shared a rare essay on X: AGI is few years away, we're in the singularity foothills, proposes US-led Frontier AI Standards Body with eventual mandatory safety testing](https://www.reddit.com/r/singularity/comments/1uw40fb/demis_hassabis_shared_a_rare_essay_on_x_agi_is/)** (Activity: 899): ****Demis Hassabis**’ essay argues AGI is plausibly *“a few years away”* and frames the current period as the *“foothills of the singularity,”* with potential impact on the order of **`10×` the Industrial Revolution at `10×` the speed**. He proposes a **US-led Frontier AI Standards Body**—analogous to **FINRA**—to evaluate frontier models, initially via voluntary pre-release safety testing that could become mandatory for major “Frontier Labs,” applying to both open and closed models and focusing on risks such as cybersecurity, biology, and autonomous agents ([essay on X](https://x.com/demishassabis/status/2076957440109625718)).** Commenters were more trusting of Hassabis’ timelines than those from Altman/Dario/Musk, but skeptical that model/code regulation is enforceable—especially against open-source or Chinese frontier models. Others noted real-world cyber impacts already appear dual-use, improving defense while raising attacker sophistication, and questioned whether a US-led standards body is geopolitically credible given current US posture toward international institutions.

    - Commenters questioned the feasibility of regulating frontier AI models or code, comparing it to the difficulty of regulating Bitcoin. One technical concern was that even if U.S. or Western labs submit to mandatory testing, **China could continue releasing increasingly capable open-source models** that are cheaper and close enough to frontier performance to attract global users.
    - A cybersecurity-related comment noted that AI is producing a dual-use effect in banking security: it has materially improved defensive capabilities while also increasing attacker sophistication. The commenter described a shift from mostly basic attacks toward more complex AI-assisted intrusion attempts, suggesting that frontier AI governance may need to account for rapidly improving cyber-offense capabilities.
    - Several commenters challenged the proposed **U.S.-led Frontier AI Standards Body**, arguing that international coordination failure is a major unresolved technical-policy risk. They highlighted problems such as China refusing to participate, labs secretly accelerating despite agreed slowdowns, and the essay’s limited treatment of power concentration, inequality, and job displacement as consequences of AGI deployment.


### 2. AI Hardware Companions and Ambient Robots

  - **[NEW LEAK: OpenAI’s First Device Will Be Moveable, Screenless Speaker Built as AI Companion](https://www.reddit.com/r/OpenAI/comments/1uwkxbc/new_leak_openais_first_device_will_be_moveable/)** (Activity: 764): **A reported leak claims **OpenAI’s first hardware product** is a **moveable, screenless smart-speaker-like AI companion** with onboard **camera/sensors** for environmental context, designed around personality, humanlike interaction, and productivity rather than a traditional display UI. The Bloomberg source itself was not accessible in the provided scrape due to a bot-detection page, so the details remain unverified from the linked article content.** Top comments were skeptical, framing the device as essentially *“reinvented Alexa”* with added surveillance concerns due to the camera/sensor stack. Several commenters joked about uncomfortable camera placement/use cases, reflecting distrust of an always-present AI companion in private spaces.

    - One commenter highlighted a concrete assistive-tech use case: a **moveable AI speaker with a camera-enabled variant** could materially help visually impaired users by providing environmental description, object recognition, and navigation-style assistance. They noted surprise that more consumer AI hardware is not explicitly targeting disability/accessibility markets, suggesting current product strategy may be limited by perceived market size rather than technical feasibility.

  - **[Keio University made these soft, helium-filled flying robots; they can follow you, wake you up, remind you of stuff, and even be your study buddy](https://www.reddit.com/r/singularity/comments/1uwc0oi/keio_university_made_these_soft_heliumfilled/)** (Activity: 1975): **The post highlights **Keio University** soft, helium-filled indoor flying robots—apparently blimp-like “airwhales”—intended for lightweight human-assistance/HRI tasks such as following a user, alarms/wake-up prompts, reminders, and acting as a study companion. The linked Reddit video could not be accessed because Reddit returned **`403 Forbidden`** ([v.redd.it](https://v.redd.it/ekrrg0d3s7dh1)); only the preview image is available ([preview](https://preview.redd.it/jyxujpmnv7dh1.png?width=1200&format=png&auto=webp&s=5890407a504dc2e96774595dcbaaed6efaeb2dc0)).** Top comments were mostly nontechnical: users joked that the robots might be amusing but not broadly useful, attractive to cats, and potentially problematic around hazards like automatic revolving doors.



### 3. AI Cost Controls in Real Deployments

  - **[Well it finally happened: we’re not using models because of cost](https://www.reddit.com/r/singularity/comments/1uwa1mv/well_it_finally_happened_were_not_using_models/)** (Activity: 1537): **A Fortune 500 “AI-first” org that had broadly deployed **GitHub Copilot** and **Claude**, with year-long internal training/demos, is now reducing access—Claude removed, usage limited, demos/training stopped—primarily due to **cost controls**, with architects recommending older/cheaper models. The flagship pilot—using agents to reverse-engineer a legacy application into an “as-written” business-rule specification for a rewrite—failed because agents repeatedly missed subtle code-path details, and day-to-day use showed reliability issues such as generated SQL attempting to `DROP` constraints around insert/delete logic. A top technical comment attributes the budget shock partly to Copilot’s shift toward metered “premium request” / usage-based pricing ([GitHub Docs](https://docs.github.com/en/copilot/concepts/billing/copilot-requests)).** Commenters debate whether this is a structural blocker or a transient phase: one view is that AI is currently in a “sour spot” where capability is near-useful but not dependable enough for complex enterprise workflows while inference/subagent costs remain high; another asks whether the company would resume “AI-first” prioritization if costs fell.

    - Several commenters framed the issue as a pricing-model shift rather than a capability failure: **GitHub Copilot moving to usage-based pricing** reportedly made large organizations re-evaluate AI spend because previously hidden or fixed costs became variable and attributable to heavy usage.
    - A technical cost-performance theme was that current models are in a “sour spot”: capabilities are close to being broadly useful, but workflows with **intense subagent use** can multiply inference calls and make deployments expensive. One commenter argued this may be temporary as model capability per dollar improves, eventually making AI use a financial “no-brainer” for many tasks.
    - Another commenter argued that expensive frontier models do not imply AI is overhyped: organizations can stay on older, cheaper models, and high prices for new models suggest buyers perceive meaningful performance gains rather than “flat performance.” They compared this to historical compute trends where *“one year’s cutting edge is the next year’s bargain bin,”* implying today’s premium models may quickly become commodity-priced as newer models arrive.

  - **[Claude spent +15 EUR of a 2 EUR limit.](https://www.reddit.com/r/ClaudeAI/comments/1uw24bp/claude_spent_15_eur_of_a_2_eur_limit/)** (Activity: 1861): **The image shows a **Claude “Usage credits” billing/limit screen** where a configured **€2.00 monthly spend limit** was exceeded to **€13.79**, displayed as **`690% used`**, with **auto-reload off** and **current balance €0.00** ([image](https://i.redd.it/iepu4woeh5dh1.png)). In context, the post alleges a single summarization request was allowed to complete and bill far beyond the user’s remaining credits/limit, implying the spend cap may function as a soft limit rather than a hard real-time cutoff.** Commenters report similar behavior, suggesting Claude may finish an in-progress turn and charge the full amount even after limits are exceeded. Some frame this as potentially unlawful or deceptive if refunds are denied, while others call it “scummy behavior.”

    - Multiple users report **Anthropic API spend limits/credit controls not acting as hard caps**: one user says a `£15` balance was consumed despite usage credits being turned off, while another reports a `$1` limit resulting in `$42` of charges before detection. A commenter hypothesizes the system may *“always finish the turn”* and bill the full in-flight request even after the configured limit is exceeded, implying spend limits may be enforced only after request completion rather than pre-authorizing against remaining budget.
    - Users describe **refund/support failures after limit overruns**, with one commenter claiming support refused to refund the `$41` overage beyond a `$1` limit. The discussion frames this as a risk for API-only workflows with fixed monthly allocations, e.g. budgeting `$200` for “fable” usage but potentially receiving much larger actual charges if limits are soft rather than hard.


