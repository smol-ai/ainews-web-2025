---
id: 9c3c47c6-79a3-4764-a3e0-5da15e724c97
title: Dia de las Secuelas (StarCoder, The Stack, Dune, SemiAnalysis)
date: '2024-03-01T00:14:08.280260Z'
original_slug: ainews-dia-de-las-secuelas-starcoder-the-stack
description: >-
  **HuggingFace/BigCode** has released **StarCoder v2**, including the
  **StarCoder2-15B** model trained on over **600 programming languages** using
  the **The Stack v2** dataset. This release marks a state-of-the-art
  achievement for models of this size, with opt-out requests excluded from
  training data. A detailed technical report is available, highlighting the
  model's capabilities and training methodology. Additionally, a live event
  featuring **Dylan Patel** discussing GPU economics is announced for San
  Francisco.
companies:
  - hugging-face
  - bigcode
models:
  - starcoder-2
  - starcoder2-15b
topics:
  - code-generation
  - model-training
  - dataset-release
  - model-performance
people:
  - dylan-patel
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 2/28/2024. We checked [**356** Twitter feeds](https://twitter.com/i/lists/1585430245762441216) and **22** Discords (**351** channels, and **9043** messages) for you. Estimated reading time saved (at 200wpm): **860 minutes**. Today's Twitter summary is a big upgrade driven by [Noah](https://twitter.com/thenoahhein) on our team, give him feedback/brickbats.

---

> **Onetime IRL callout**: If you're in SF, join Dylan Patel (aka "that semianalysis guy" who wrote the GPU Rich/Poor essay) for [a special live Latent Space special event tomorrow](https://twitter.com/dylan522p/status/1763281120161140833). Our [first convo](https://www.latent.space/p/semianalysis) was one of last year's top referenced eps.

---

As [hinted last year](https://www.latent.space/p/idefics), HuggingFace/BigCode has finally released [StarCoder v2](https://huggingface.co/bigcode/starcoder2-15b) and [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids). Full [technical report here](https://drive.google.com/file/d/17iGn3c-sYNiLyRSY-A85QOzgzGnGiVI3/view). 





## StarCoder 2: SOTA for size (3B and 15B)

<p><span style="color: rgb(75, 85, 99); font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;">StarCoder2-15B model is a 15B parameter model trained on 600+ programming languages from<span>&nbsp;</span></span><a rel="nofollow" href="https://huggingface.co/datasets/bigcode/the-stack-v2-train" style="box-sizing: border-box; border-width: 0px; border-style: solid; border-color: rgb(229, 231, 235); --tw-border-spacing-x: 0; --tw-border-spacing-y: 0; --tw-translate-x: 0; --tw-translate-y: 0; --tw-rotate: 0; --tw-skew-x: 0; --tw-skew-y: 0; --tw-scale-x: 1; --tw-scale-y: 1; --tw-pan-x: ; --tw-pan-y: ; --tw-pinch-zoom: ; --tw-scroll-snap-strictness: proximity; --tw-gradient-from-position: ; --tw-gradient-via-position: ; --tw-gradient-to-position: ; --tw-ordinal: ; --tw-slashed-zero: ; --tw-numeric-figure: ; --tw-numeric-spacing: ; --tw-numeric-fraction: ; --tw-ring-inset: ; --tw-ring-offset-width: 0px; --tw-ring-offset-color: #fff; --tw-ring-color: rgb(59 130 246 / .5); --tw-ring-offset-shadow: 0 0 #0000; --tw-ring-shadow: 0 0 #0000; --tw-shadow: 0 0 #0000; --tw-shadow-colored: 0 0 #0000; --tw-blur: ; --tw-brightness: ; --tw-contrast: ; --tw-grayscale: ; --tw-hue-rotate: ; --tw-invert: ; --tw-saturate: ; --tw-sepia: ; --tw-drop-shadow: ; --tw-backdrop-blur: ; --tw-backdrop-brightness: ; --tw-backdrop-contrast: ; --tw-backdrop-grayscale: ; --tw-backdrop-hue-rotate: ; --tw-backdrop-invert: ; --tw-backdrop-opacity: ; --tw-backdrop-saturate: ; --tw-backdrop-sepia: ; color: var(--tw-prose-links); text-decoration: underline; font-weight: 500; font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255);">The Stack v2</a><span style="color: rgb(75, 85, 99); font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;">, with opt-out requests excluded. The model uses<span>&nbsp;</span></span><a rel="nofollow" href="https://arxiv.org/abs/2305.13245" style="box-sizing: border-box; border-width: 0px; border-style: solid; border-color: rgb(229, 231, 235); --tw-border-spacing-x: 0; --tw-border-spacing-y: 0; --tw-translate-x: 0; --tw-translate-y: 0; --tw-rotate: 0; --tw-skew-x: 0; --tw-skew-y: 0; --tw-scale-x: 1; --tw-scale-y: 1; --tw-pan-x: ; --tw-pan-y: ; --tw-pinch-zoom: ; --tw-scroll-snap-strictness: proximity; --tw-gradient-from-position: ; --tw-gradient-via-position: ; --tw-gradient-to-position: ; --tw-ordinal: ; --tw-slashed-zero: ; --tw-numeric-figure: ; --tw-numeric-spacing: ; --tw-numeric-fraction: ; --tw-ring-inset: ; --tw-ring-offset-width: 0px; --tw-ring-offset-color: #fff; --tw-ring-color: rgb(59 130 246 / .5); --tw-ring-offset-shadow: 0 0 #0000; --tw-ring-shadow: 0 0 #0000; --tw-shadow: 0 0 #0000; --tw-shadow-colored: 0 0 #0000; --tw-blur: ; --tw-brightness: ; --tw-contrast: ; --tw-grayscale: ; --tw-hue-rotate: ; --tw-invert: ; --tw-saturate: ; --tw-sepia: ; --tw-drop-shadow: ; --tw-backdrop-blur: ; --tw-backdrop-brightness: ; --tw-backdrop-contrast: ; --tw-backdrop-grayscale: ; --tw-backdrop-hue-rotate: ; --tw-backdrop-invert: ; --tw-backdrop-opacity: ; --tw-backdrop-saturate: ; --tw-backdrop-sepia: ; color: var(--tw-prose-links); text-decoration: underline; font-weight: 500; font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255);">Grouped Query Attention</a><span style="color: rgb(75, 85, 99); font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;">,<span>&nbsp;</span></span><a rel="nofollow" href="https://arxiv.org/abs/2205.14135" style="box-sizing: border-box; border-width: 0px; border-style: solid; border-color: rgb(229, 231, 235); --tw-border-spacing-x: 0; --tw-border-spacing-y: 0; --tw-translate-x: 0; --tw-translate-y: 0; --tw-rotate: 0; --tw-skew-x: 0; --tw-skew-y: 0; --tw-scale-x: 1; --tw-scale-y: 1; --tw-pan-x: ; --tw-pan-y: ; --tw-pinch-zoom: ; --tw-scroll-snap-strictness: proximity; --tw-gradient-from-position: ; --tw-gradient-via-position: ; --tw-gradient-to-position: ; --tw-ordinal: ; --tw-slashed-zero: ; --tw-numeric-figure: ; --tw-numeric-spacing: ; --tw-numeric-fraction: ; --tw-ring-inset: ; --tw-ring-offset-width: 0px; --tw-ring-offset-color: #fff; --tw-ring-color: rgb(59 130 246 / .5); --tw-ring-offset-shadow: 0 0 #0000; --tw-ring-shadow: 0 0 #0000; --tw-shadow: 0 0 #0000; --tw-shadow-colored: 0 0 #0000; --tw-blur: ; --tw-brightness: ; --tw-contrast: ; --tw-grayscale: ; --tw-hue-rotate: ; --tw-invert: ; --tw-saturate: ; --tw-sepia: ; --tw-drop-shadow: ; --tw-backdrop-blur: ; --tw-backdrop-brightness: ; --tw-backdrop-contrast: ; --tw-backdrop-grayscale: ; --tw-backdrop-hue-rotate: ; --tw-backdrop-invert: ; --tw-backdrop-opacity: ; --tw-backdrop-saturate: ; --tw-backdrop-sepia: ; color: var(--tw-prose-links); text-decoration: underline; font-weight: 500; font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255);">a context window of 16,384 tokens</a><span style="color: rgb(75, 85, 99); font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;"><span>&nbsp;</span>with<span>&nbsp;</span></span><a rel="nofollow" href="https://arxiv.org/abs/2004.05150v2" style="box-sizing: border-box; border-width: 0px; border-style: solid; border-color: rgb(229, 231, 235); --tw-border-spacing-x: 0; --tw-border-spacing-y: 0; --tw-translate-x: 0; --tw-translate-y: 0; --tw-rotate: 0; --tw-skew-x: 0; --tw-skew-y: 0; --tw-scale-x: 1; --tw-scale-y: 1; --tw-pan-x: ; --tw-pan-y: ; --tw-pinch-zoom: ; --tw-scroll-snap-strictness: proximity; --tw-gradient-from-position: ; --tw-gradient-via-position: ; --tw-gradient-to-position: ; --tw-ordinal: ; --tw-slashed-zero: ; --tw-numeric-figure: ; --tw-numeric-spacing: ; --tw-numeric-fraction: ; --tw-ring-inset: ; --tw-ring-offset-width: 0px; --tw-ring-offset-color: #fff; --tw-ring-color: rgb(59 130 246 / .5); --tw-ring-offset-shadow: 0 0 #0000; --tw-ring-shadow: 0 0 #0000; --tw-shadow: 0 0 #0000; --tw-shadow-colored: 0 0 #0000; --tw-blur: ; --tw-brightness: ; --tw-contrast: ; --tw-grayscale: ; --tw-hue-rotate: ; --tw-invert: ; --tw-saturate: ; --tw-sepia: ; --tw-drop-shadow: ; --tw-backdrop-blur: ; --tw-backdrop-brightness: ; --tw-backdrop-contrast: ; --tw-backdrop-grayscale: ; --tw-backdrop-hue-rotate: ; --tw-backdrop-invert: ; --tw-backdrop-opacity: ; --tw-backdrop-saturate: ; --tw-backdrop-sepia: ; color: var(--tw-prose-links); text-decoration: underline; font-weight: 500; font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255);">a sliding window attention of 4,096 tokens</a><span style="color: rgb(75, 85, 99); font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;">, and was trained using the<span>&nbsp;</span></span><a rel="nofollow" href="https://arxiv.org/abs/2207.14255" style="box-sizing: border-box; border-width: 0px; border-style: solid; border-color: rgb(229, 231, 235); --tw-border-spacing-x: 0; --tw-border-spacing-y: 0; --tw-translate-x: 0; --tw-translate-y: 0; --tw-rotate: 0; --tw-skew-x: 0; --tw-skew-y: 0; --tw-scale-x: 1; --tw-scale-y: 1; --tw-pan-x: ; --tw-pan-y: ; --tw-pinch-zoom: ; --tw-scroll-snap-strictness: proximity; --tw-gradient-from-position: ; --tw-gradient-via-position: ; --tw-gradient-to-position: ; --tw-ordinal: ; --tw-slashed-zero: ; --tw-numeric-figure: ; --tw-numeric-spacing: ; --tw-numeric-fraction: ; --tw-ring-inset: ; --tw-ring-offset-width: 0px; --tw-ring-offset-color: #fff; --tw-ring-color: rgb(59 130 246 / .5); --tw-ring-offset-shadow: 0 0 #0000; --tw-ring-shadow: 0 0 #0000; --tw-shadow: 0 0 #0000; --tw-shadow-colored: 0 0 #0000; --tw-blur: ; --tw-brightness: ; --tw-contrast: ; --tw-grayscale: ; --tw-hue-rotate: ; --tw-invert: ; --tw-saturate: ; --tw-sepia: ; --tw-drop-shadow: ; --tw-backdrop-blur: ; --tw-backdrop-brightness: ; --tw-backdrop-contrast: ; --tw-backdrop-grayscale: ; --tw-backdrop-hue-rotate: ; --tw-backdrop-invert: ; --tw-backdrop-opacity: ; --tw-backdrop-saturate: ; --tw-backdrop-sepia: ; color: var(--tw-prose-links); text-decoration: underline; font-weight: 500; font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255);">Fill-in-the-Middle objective</a><span style="color: rgb(75, 85, 99); font-family: &quot;Source Sans Pro&quot;, ui-sans-serif, system-ui, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;, &quot;Noto Color Emoji&quot;; font-size: 16.8px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;"><span>&nbsp;</span>on 4+ trillion tokens.</span></p>

Since it was only just released, best source on evals is [BigCode for now](https://twitter.com/bigcodeproject/status/1762842312005026258?utm_source=ainews&utm_medium=email): 

![image.png](https://assets.buttondown.email/images/e894f5c7-266d-4874-8c2f-e3221391eab0.png?w=960&fit=max)



## The Stack v2: 10x bigger raw, and 4.5x bigger deduped (900B Tokens)

 ![image.png](https://assets.buttondown.email/images/a8ae0a37-f5ef-44ef-9fe5-027cebc6b04d.png?w=960&fit=max) 






---

**We are experimenting with removing Table of Contents as many people reported it wasn't as helpful as hoped. Let us know if you miss the TOCs, or they'll be gone permanently.**

---

# AI Twitter Summary


**AI and Machine Learning Discussions**

- [François Chollet remarks on the nature of LLMs](https://twitter.com/fchollet/status/1762977717740621839), emphasizing that output mirrors the training data, capturing human thought patterns.
- [Sedielem shares extensive thoughts on diffusion distillation](https://twitter.com/sedielem/status/1762976957728186497), inviting community feedback on the blog post.
- [François Chollet differentiates between current AI capabilities and true intelligence](https://twitter.com/fchollet/status/1762978569624686682), focusing on the efficiency of skill acquisition.
- [Stas Bekman raises concerns about the ML community's dependency on a single hub for accessing weight copies](https://twitter.com/StasBekman/status/1762960092847333647), suggesting the need for a backup hub.

**Executive Shifts and Leadership**

- [Saranormous highlights leadership change at $SNOW](https://twitter.com/saranormous/status/1762957337470324810), welcoming @RamaswmySridhar as the new CEO and applauding his technical and leadership expertise.

**Technology Industry Updates**

- [DeepLearningAI rounds up this week’s AI stories](https://twitter.com/DeepLearningAI/status/1762975968778412094), including Gemini 1.5 Pro's rough week, Groq chips’ impact on AI processing speed, and a discussion on version management in AI development by @AndrewYNg.
- [KevinAFischer celebrates his feature in Tech Crunch](https://twitter.com/KevinAFischer/status/1762960594406383830) as an early user of the Shader app by @shaderapp and @daryasesitskaya.

**Innovation and Technical Insights**

- [Andrew N Carr discusses the potential of fitting 120B parameter models on consumer GPUs](https://twitter.com/andrew_n_carr/status/1762975401482293339) as per the 1.58 Bit paper, emphasizing breakthroughs in VRAM efficiency.
- [Erhartford highlights a real-time EMO lip sync model](https://twitter.com/erhartford/status/1762977518125302167), suggesting its integration for innovative applications.

**Memes/Humor**

- [C_valenzuelab draws a humorous analogy](https://twitter.com/c_valenzuelab/status/1762982681548050900), stating that "airplanes didn't disrupt the bicycle market".
- [KevinAFischer jokes about the economics of using LLMs](https://twitter.com/KevinAFischer/status/1762984942651482419), poking fun at the current state of AI development.
- [KevinAFischer makes a light-hearted comment](https://twitter.com/KevinAFischer/status/1762965650371420520) about ideas being ahead of their time.


**Miscellaneous Observations**

- [Margaret Mitchell questions diversity in news coverage of the Gemini fiasco](https://twitter.com/mmitchell_ai/status/1762971964539637815) - *2806 impressions*
- [Kevin Fischer humorously touches on repeating himself](https://twitter.com/KevinAFischer/status/1762965520180281654) - *732 impressions*
- [Zach talks about the need for fair tax rates for the wealthy](https://twitter.com/zachtratar/status/1762987044522107275) - *492 impressions*

**AI Development and Infrastructure**

- [abacaj mentions the need for backing up weights following an HF outage](https://twitter.com/abacaj/status/1762988118288810094) - *1558 impressions*
- [Together Compute announces the launch of OLMo-7B-Instruct API from @allen_ai](https://twitter.com/togethercompute/status/1762988981845987432) - *334 impressions*
- [A discussion on the ternary BitNet paper's potential to revolutionize model scalability](https://twitter.com/teortaxesTex/status/1762993615750516777) - *42 impressions*

### AI Twitter Narrative

The technical and engineer-oriented Twitter ecosystem has been buzzing with significant discussions spanning AI, blockchain, leadership transitions in tech, and some light-hearted humor. 

Regarding **AI and Machine Learning**, François Chollet’s [reflection on LLMs](https://twitter.com/fchollet/status/1762977717740621839) as mirrors to our inputs, alongside **Daniele Grattarola's deep dive into diffusion distillation**, underscore critical thinking about the essence and future of AI technologies. Reinforcing the importance of diversified safeguarding of machine learning models, [Stas Bekman’s proposition](https://twitter.com/StasBekman/status/1762960092847333647) for a secondary hub for model weights has caught the community's attention, highlighting the community's resilience in facing practical challenges.

In the **leadership and innovation arena**, [the leadership transition at $SNOW](https://twitter.com/saranormous/status/1762957337470324810) garnered significant engagement, reflecting the continuous evolution and admiration for leadership within tech organizations.

**Humor and memes** remain a vital part of the discourse, with tweets like [Cristóbal Valenzuela’s observation](https://twitter.com/c_valenzuelab/status/1762982681548050900) about the non-competition between airplanes and bicycles bringing a light-hearted perspective to innovation and disruption.

On various **miscellaneous observations**, [Margaret Mitchell’s call](https://twitter.com/mmitchell_ai/status/1762971964539637815) for more diverse perspectives in tech reporting highlights the importance of inclusivity and varied viewpoints in shaping our understanding of tech events.

Lastly, discussions around **AI development and infrastructure** saw practical considerations taking the forefront, as noted by [abacaj’s preparation](https://twitter.com/abacaj/status/1762988118288810094) for possible future outages by backing up model weights. This operational resilience mirrors the broader strategic resilience seen across the technical and engineering community.

---

# PART 0: Summary of Summaries of Summaries

<div><h2><strong>ChatGPT Model Evaluations and Data Integrity on TheBloke Discord</strong></h2><ul><li><strong>Detailed ChatGPT Model Comparisons</strong>: Members critically evaluated <strong>ChatGPT models</strong>, including <strong>GPT-4</strong>, <strong>Mixtral</strong>, and <strong>Miqu</strong>, focusing on <strong>API reliability</strong> and comparative performance. Specific concerns were raised about <strong>training data contamination</strong> from other AI outputs, potentially degrading model quality and reliability.</li></ul><h2><strong>Technological Innovations and AI Deployment on Mistral Discord</strong></h2><ul><li><strong>NVIDIA RAG Technical Limitations</strong>: NVIDIA's demo, showcasing <strong>retrieval-augmented generation (RAG)</strong>, was critiqued for its <strong>1024 token context limit</strong> and response coherence issues. The critique extended to NVIDIA's implementation choices, including the use of <strong>LangChain</strong> for RAG's reference architecture, hinting at broader discussions on optimizing <strong>AI model architectures</strong> for better performance.</li></ul><h2><strong>Qualcomm's Open Source AI Models on LM Studio Discord</strong></h2><ul><li><strong>Qualcomm's Contribution to AI Development</strong>: Qualcomm released <strong>80 open source AI models</strong> on <strong>Hugging Face</strong>, targeting diverse applications in <strong>vision</strong>, <strong>audio</strong>, and <strong>speech</strong> technologies. Notable models include <strong>"QVision"</strong> for image processing, <strong>"QSpeech"</strong> for audio recognition, and <strong>"QAudio"</strong> for enhanced sound analysis. These models represent Qualcomm's push towards enriching the AI development ecosystem, offering tools for researchers and developers to innovate in <strong>machine learning applications</strong> across various domains. The release was aimed at fostering advancements in AI modeling and development, specifically enhancing capabilities in vision and audio processing, as well as speech recognition tasks.</li></ul><p>These updated summaries provide a more focused view on the specific areas of interest and discussion within the respective Discord communities. They highlight the depth of technical scrutiny applied to AI models, the identification of performance limitations and potential improvements in AI technologies, and the specific contributions of Qualcomm to the open-source AI landscape, underlining the continuous evolution and collaborative nature of AI research and development.</p></div>

---

# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Spam Alert in General Chat**: Users reported a spam incident involving `@kquant`, with Discord's spam detection system flagging his activity after excessively contacting over 100 people with identical messages.
- **ChatGPT Variants Under Scrutiny**: Diverse experiences with ChatGPT models were discussed, including GPT-4's API reliability and comparisons with Mixtral or Miqu models. Concerns were raised over training data contamination from other AI outputs, potentially compromising quality.
- **Mixed Results in Model Mergers**: Dialogue highlighted the uncertainty in model merging outcomes, emphasizing the role of luck and model compatibility. Merging tactics such as **spherical linear interpolation (slerp)** or **concatenation** were suggested in the specialized channels.
- **Innovative Roleplay with LLMs**: Techniques to enhance character consistency in role-play involve using detailed backstories and traits for LLMs. Specific models like Miqu and Mixtral were favored for these tasks, though longer context length could reduce coherence.
- **Pacing AI Training and Fine-tuning**: Users exchanged training tips, including using [Perplexity AI](https://www.perplexity.ai) and efficient methods like QLoRA to curb hardware demand. The importance of validation and deduplication was stressed, alongside managing model generalization and hallucination.

Links to consider:

- For looking into detailed personalities and character backstories in AI role-play, one might explore the strategy explanations and datasets at [Hugging Face](https://huggingface.co/datasets/maldv/conversation-cixot).
- Searching for efficient training techniques could lead AI engineers to MAX's announcement about their platform aimed at democratizing AI development via an optimized infrastructure, detailed in their Developer Edition Preview blog post [here](https://www.modular.com/blog/announcing-max-developer-edition-preview).



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **NVIDIA's Demo Faces Criticism for RAG Implementation**: The NVIDIA "Chat with RTX" demo showcasing retrieval-augmented generation (RAG) faced criticism for limiting context size to 1024 tokens and issues with coherent responses. Discussions hinted at concerns with NVIDIA's use of LangChain in RAG's reference architecture.

- **Mistral AI Discussions Span Licensing to Open Weights and Hardware Requirements**: Conversations touched on Mistral AI's use of Meta's LLaMa model, anticipation for future open weight models following Mistral-7B, and hardware requirements for running larger models, like Mistral 8x7B, which may need at least 100GB of VRAM. Users considered the use of services like [Together.AI](https://together.ai) for deployment assistance.

- **Model Quantization and Deployment Discussions Highlight Constraints**: Technical discussions included constraining Mistral-7B to specific document responses, the stateless nature of language models, and the limitations of quantized models. Quantization reducing parameter counts for Mistral-7B and the necessity for large VRAM for full precision models were underscored.

- **Mistral Platform Intricacies and Function Calling Discussed**: Users shared experiences and obstacles with Mistral function calls and reported on the necessity for specific message role orders. Some referred to the use of tools like [Mechanician](https://github.com/liebke/mechanician) for better integration with Mistral AI.

- **Educational Tools and the Potential of Specialized Models**: One user showcased an app for teaching economics using Mistral and GPT-4 AI models, while discussions touched on the specialized training of models for tasks like JavaScript optimization. An expressed need for improved hiring strategies within the AI industry surfaced among chats.

The conversations reveal technical discernment among the users, highlighting both enthusiasm for AI's advancements and practical discussions on AI model limitations and ideal deployment scenarios.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Loader Showdown: lm studio vs oobabooga and Jan dot ai**: lm studio was criticized for requiring manual GUI interaction to kickstart the API, making it a non-viable option for automated website applications, prompting engineers to suggest alternatives oobabooga and Jan dot ai for more seamless automation.

- **AI Moderation and OpenAI Feedback**: A message removed in a discussion about **Copilot AI** due to automod censorship led to suggestions to report to Discord mods and submit feedback directly through OpenAI's [Chat model feedback](https://openai.com/form/chat-model-feedback) form, with community members discussing the extent of moderation rules.

- **Mistral's Power and Regulation Query**: The **Mistral** model, known for its powerful, uncensored outputs was compared to **GPT-4**, resulting in a conversation about the impact of European AI regulation on such models. A related [YouTube video](https://www.youtube.com/watch?v=GyllRd2E6fg) was shared, illustrating how to run **Mistral** and its implications.

- **Advancing Chatbot Performance**: Enhancing **GPT-3.5-Turbo** for chatbot applications sparked a debate on achieving performance on par with **GPT-4**, with users discussing fine-tuning techniques and suggesting utilizing actual data and common use cases for improvement.

- **AI Certification vs. Real-world Application**: For those seeking AI specialization, the community highlighted the primacy of hands-on projects over certifications, recommending learning resources such as courses by **Andrew Ng** and **Andrej Karpathy**, available on YouTube.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

**Model Compatibility Queries Spark GPU Discussions**: Engineers engaged in detailed explorations of LLMs, such as Deepseek Coder 6.7B and StarCoder2-15B, and their compatibility with Nvidia RTX 40 series GPUs, discussing optimization strategies for GPUs like disabling certain features on Windows 11. A focus on finding the best-fitting models for hardware specifications was observed, underlined by the launch news of [StarCoder2 and The Stack v2](https://twitter.com/bigcodeproject/status/1762842312005026258), with mentions of LM Studio's compatibility issues, especially on legacy hardware like the GTX 650.

**Hugging Face Outage Disrupts Model Access**: An outage at Hugging Face caused network errors for members trying to download models, affecting their ability to search for models within LM Studio.

**Qualcomm Unveils 80 Open Source Models**: Qualcomm released 80 open source AI models on Hugging Face, targeting vision, audio, and speech applications, potentially enriching the landscape for AI modeling and development.

**LLM Functionality Expansions**: Users exchanged insights on enhancing functionalities within LM Studio, such as implementing an accurate PDF chatbot with Llama2 70B Q4 LLM, seeking guidance on adding image recognition features with models like `PsiPi/liuhaotian_llava-v1.5-13b-GGUF/`, and expressing desires for simplified processes in downloading vision adapter models.

**Hardware Hubris and Hopes**: Discussions thrived around user experiences with hardware, from reminiscing about older GPUs to sharing frustrations over misrepresented specs in an e-commerce setting. One user advised optimizations for Windows 11, while TinyCorp announced a new hardware offering, TinyBox, found [here](https://tinygrad.org). There was also speculation about the potential for Nvidia Nvlink / SLI in model training compared to inference tasks.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Cosmopedia's Grand Release**: **Cosmopedia** was announced, a sizable synthetic dataset with over **25B tokens** and **30M files**, constructed by Mixtral. It is aimed at serving various AI research needs, with the release information accessible through this [LinkedIn post](https://www.linkedin.com/posts/loubna-ben-allal-238690152_today-were-releasing-cosmopedia-the-activity-7165785808883404800-t8o4?utm_source=share&utm_medium=member_desktop).

- **Hugging Face Updates Galore**: The `huggingface_hub` library has a new release **0.21.0** with several improvements, and **YOLOv9** made its debut on the platform, now compatible with **Transformers.js** as per the discussions and platforms like [Hugging Face spaces](https://huggingface.co/spaces/Wauplin/huggingface_hub/discussions/4) and [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-feature-extraction&sort=trending).

- **DSPy Grows Closer to Production**: Exploration of **DSPy** and **Gorilla OpenFunctions v2** is underway to transition from Gradio prototypes to production versions. The tools promise enhanced client onboarding processes for foundation models without prompting, and the discussions and resources can be found in repositories like [stanfordnlp/dspy on GitHub](https://github.com/stanfordnlp/dspy).

- **BitNet Bares Its Teeth**: A new **1-bit Large Language Model, BitNet b1.58**, boasted to preserve performance with impressive efficiency metrics, is discussed with its research available via this [arXiv paper](https://arxiv.org/abs/2402.17764).

- **Inference Challenges and Solutions**: In the field of text inference, an AI professional ran into issues when trying to deploy the [text generation inference repository](https://github.com/huggingface/text-generation-inference) on a CPU-less and non-CUDA system. This highlights typical environmental constraints encountered in AI model deployment.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **AI's Ideogram Stirs Interest**: Engineers discussed the release of a new AI model from **Ideogram**, drawing comparisons with **Stable Diffusion** and shedding light on speculated quality matters pertaining to unseen **Imagen** samples. A user shared a prompt result that sparked a debate on its prompt adherence and aesthetics.

- **Integration of T5 XXL and CLIP in SD3 Discussed**: There have been discussions around the potential integration of **T5 XXL** and **CLIP models** into **Stable Diffusion 3 (SD3)**, with participants expecting advancements in both the precision and the aesthetics of upcoming generative models.

- **Concerns Over AI-Generated Art**: A legal discussion unfolded concerning **AI-generated art** and **copyright** laws, referencing a verdict from China and an article on copyright safety for generative AI, highlighting uncertainty in the space and varied industry responses to DMCA requests.

- **Spiking Neural Networks Back in Vogue?**: Some members considered the potential resurgence of **spiking neural networks** with advanced techniques like time dithering to improve precision, reflecting on historical and current research approaches.

- **State-of-the-Art Icon Generation Model Released**: A new **AI icon generation model** has been released on **Hugging Face**, developed with a personal funding of $2,000 and touted to create low-noise icons at 256px, although scale limitations were acknowledged by its creator.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Emoji Storytelling on GPT-5's No-show**: Community members used a sequence of emojis to express sentiments about GPT-5's absence, oscillating between salutes, skulls, and tears, while revering GPT iterations up to the mythical GPT-9.

- **Dell's Dual Connection Monitors and Docks Intrigue Engineers**: A [YouTube review](https://youtu.be/0TY7J58UEro?si=5UayYH3t3gCC0M_H) of Dell's new 5K monitor and the Dell Thunderbolt Dock WD22TB4 piqued interest for their capabilities to connect multiple machines, with eBay as the suggested source for purchases.

- **1-bit LLMs Unveiled with BitNet B1.58**: The [arXiv paper](https://arxiv.org/abs/2402.17764) revealed **BitNet b1.58** as a 1-bit LLM with performance on par with full-precision models, highlighting it as a cost-effective innovation alongside a mention of [Nicholas Carlini's LLM benchmark](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html).

- **Exploring Alternative Low-Cost LLMs and Fine-Tuning Practices**: Users discussed alternatives to GPT-4, the effect of small training dataset sizes, and the potential use of Directed Prompt Optimization (DPO) to improve model responses.

- **Cutting-Edge Research and New Genomic Model Debut**: Stanford's release of **HyenaDNA**, a genomic sequence model, alongside surprising MMLU scores from CausalLM, and resources on interpretability in AI, such as [Representation Engineering](https://arxiv.org/abs/2310.01405) and [tokenization strategies](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/tokenizer.json), were the hot topics of discussion.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Noam Shazeer on Coding Style**: `@swyxio` highlighted Noam Shazeer's first blog post on [coding style and shape suffixes](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd), which may interest developers who are keen on naming conventions.

- **AI in Customer Service**: Enthusiasm was expressed around data indicating that **LLMs can match human performance in customer service**, potentially handling two-thirds of customer service queries, suggesting a pivot in how customer interactions are managed.

- **Learning with Matryoshka Embeddings**: Members discussed the innovative ["Matryoshka Representation Learning"](https://arxiv.org/abs/2310.07707) paper and its applications in **LLM embeddings with adaptive dimensions**, with potential benefits for compute and storage efficiency.

- **MRL Embeddings Event**: An announcement for an upcoming event by `<@206404469263433728>` where the authors of the **MRL embeddings paper** will attend was made, providing an opportunity for deep discussions on representation learning in the [`#1107320650961518663`](https://lu.ma/rgxvuktv) channel.

- **Representation Engineering Session**: `@ivanleomk` signaled an educational session on **Representation Engineering 101** with `<@796917146000424970>`, indicating a chance to learn and query about engineering effective data representations in the [`#1107320650961518663`](https://discord.com) channel.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Rabbit R1 Activation Assistance**: User `@mithrilman` encountered a non-clickable email link issue when trying to activate the Rabbit R1 promo. `@icelavaman` suggested using the email link and reaching out to support.

- **Podcast Identity Confirmation**: Confusion arose around podcasts using the name "Perplexity AI," leading `@icelavaman` to clarify with the official podcast link, while `@ok.alex` speculated that the name might be used without authorization for attention or financial gain.

- **Comparing AI Model Capabilities**: Users explored the strengths and weaknesses of various AI models like Experimental, GPT-4 Turbo, Claude, and Mistral. There was notably divided opinion regarding Mistral's effectiveness for code queries.

- **Brainstorming Perplexity AI Improvements**: Suggestions for Perplexity AI included exporting thread responses, a feature currently missing but considered for future updates. Issues also included the absence of file upload options and confusion over product name changes.

- **Model Performance Nostalgia and API Errors**: Discussions touched upon glitches in text generation and fond memories of **pplx-70b** being superior to **sonar** models. `@jeffworthington` faced challenges with OpenAPI definitions, suggesting the current documentation might be outdated.

**Links shared**:
- Official Perplexity AI podcasts: [‎Discover Daily by Perplexity](https://podcasts.apple.com/us/podcast/discover-daily-by-perplexity/id1732181427) and [‎Perplexity AI](https://podcasts.apple.com/us/podcast/perplexity-ai/id1725553091).
- Getting started with Perplexity's API: [pplx-api documentation](https://docs.perplexity.ai/docs/getting-started).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Foundation Model Development Cheatsheet Unveiled**: A new resource titled **The Foundation Model Development Cheatsheet** has been released to aid open model developers, featuring contributions from EleutherAI, MIT, AI2, Hugging Face, among others, and focusing on often overlooked yet crucial aspects such as dataset documentation and licensing. The cheatsheet can be accessed as a [PDF paper](https://github.com/allenai/fm-cheatsheet/blob/main/app/resources/paper.pdf) or an [interactive website](https://fmcheatsheet.org/), with additional information in their [blog post](https://blog.eleuther.ai/fm-dev-cheatsheet/) and [Twitter thread](https://twitter.com/AiEleuther/status/1763219826602901518).

- **Scaling Laws and Model Training Discussions Heat Up**: Discourse ranges from inquiries about cross-attention SSM models, stable video diffusion training, and the nuances of *lm-evaluation-harness*, to the status of EleutherAI's Pythia model, and an abstract on a 1-bit Large Language Model (LLM). Notable references include a blog post on [Multiple Choice Normalization in LM Evaluation](https://blog.eleuther.ai/multiple-choice-normalization/) and the research paper on the [Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764).

- **From Open-Sourced Models To Maze Solving Diffusion Models**: The research channel showcased discussions on a variety of AI topics, from open-sourced models and pretraining token-to-model size ratios to diffusion models trained to solve mazes, prompting engineering transfer studies, and the practical challenges of sub 8-bit quantization. Key resources shared include a [Stable LM 2 1.6B Technical Report](https://arxiv.org/abs/2402.17834), and a tweet on training diffusion models to solve mazes by [François Fleuret](https://x.com/francoisfleuret/status/1762866220636807219?s=20).

- **Neox Query for Slurm Compatibility**: User `@muwnd` sought recommendations on running **Neox** with **Slurm** and its compatibility with containers. It was highlighted that Neox's infrastructure does not make assumptions about the user's setup, and a slurm script may be needed for multinode execution.

- **Interpretability Techniques and Norms Explored**: Conversations in the interpretability channel delved into matrix norms and products, RMSNorm layer applications, decoding using tuned lenses, and the proper understanding of matrix norm terminology. For example, the Frobenius norm is the Euclidean norm when the matrix is flattened, while the "2-norm" is the spectral norm or top singular value.

- **Tweaks for LM Eval Harness and Multilingual Upgrades**: Enhancements to the LM Eval harness for chat templates were shared, along with news that higher-quality translations for the Multilingual Lambada have been contributed by `@946388490579484732` and will be included in the evaluation harness. These datasets are made available on [Hugging Face](https://huggingface.co/datasets/marcob/lambada_multilingual).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Confidence in LangChain.js**: `@ritanshoo` raised a question regarding confidence score checks when utilizing LangChain.js for RAG. While an immediate answer was not provided, users were referred to the **[LangChain documentation](https://js.langchain.com/docs/get_started)** for in-depth guidance.
  
- **Integration Queries for LangChain**: Technical discussions highlighted the possibilities of **memory addition to LCEL** and **effective language integration with LangChain in an Azure-hosted environment**. Users were advised to consult official documentation or seek community assistance for specific integration issues.

- **ToolException Workarounds Explored**: `@abinandan sought` advice on retrying a tool after a `ToolException` occurs with a custom tool. The community pointed to **[LangChain GitHub discussions and issues](https://github.com/langchain-ai/langchain/issues/10714)** for potential solutions.

- **LangServe Execution Quirks**: `@thatdc` reported missing intermediate step details when using **langserve**, as opposed to direct invocation from the agent class. They identified a potential glitch in the `RemoteRunnable` requiring a workaround.

- **Summoning Python Template Alchemists**: `@tigermusk` sought assistance creating a Python template similar to the one available on **[Smith LangChain Chat JSON Hub](https://smith.langchain.com/hub/hwchase17/react-chat-json)**, sparking discussions on template generation. 

- **"LangChain in your Pocket" Celebrated**: `@mehulgupta7991` announced their book "LangChain in your Pocket," recently featuring in Google's Best books on LangChain, highlighting resources for LangChain enthusiasts.

- **Beta Testing for AI Voice Chat App**: Pablo, an AI Voice Chat app that integrates multiple LLMs and provides voice support without typing, called for beta testers. Engineers were invited to join the team behind this app, leveraging LangChain technology, with an **[offer for free AI credits](https://testflight.apple.com/join/raZGq35o)**.

- **AI Stock Analysis Chatbot Creation Explained**: A **[video tutorial](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s)** was shared by `@tarikkaoutar`, demonstrating the construction of an AI stock analysis chatbot using LangGraph, Function call, and YahooFinance, catering to engineers interested in multi-agent systems.

- **Groq's Hardware Reveal Generates Buzz**: An introduction to Groq's breakthrough Language Processing Unit (LPU) suitable for LLMs captivated tech enthusiasts, conveyed through a **[YouTube showcase](https://youtu.be/RSzG_v5XIxM)** shared by `@datasciencebasics`.

(Note: The above summary integrates topics and resources from various channels within the Discord guild, focusing on points of interest most relevant to an engineer audience looking for technical documentation, coding integration, and advancement in AI hardware and applications.)



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Jupyter Configuration Chaos**: Users reported issues with **Jupyter notebooks**, highlighting error messages concerning extension links and a **"Bad config encountered during initialization"** without a conclusive solution in the discussion.

- **BitNet b1.58 Breakthroughs**: An [arXiv paper](https://arxiv.org/abs/2402.17764) introduced **BitNet b1.58**, a 1-bit LLM that matches the performance of full-precision models, heralding significant cost-efficiency with an innovative architecture.

- **Sophia Speeds Past Adam**: The [Sophia optimizer](https://arxiv.org/abs/2305.14342), claimed to be twice as fast as Adam algorithms, was shared alongside its [implementation code](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/optim/sophia.py), sparking interest in its efficiency for optimization methods in AI models.

- **DropBP Drops Layers for Efficiency**: A study presented [Dropping Backward Propagation (DropBP)](https://arxiv.org/abs/2402.17812), a method that can potentially reduce computational cost in neural network training by skipping layers during backward propagation without significantly affecting accuracy.

- **Scandinavian Showdown: Mistral vs. ChatGPT 3.5**: A user, **le_mess**, reported that their **7B Mistral model** rivaled **ChatGPT 3.5** in performance for Danish language tasks, using an iterative synthetic data approach for progressive training through 30 iterations and initial human curation.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Groq's Integration Powers Up LlamaIndex**: The **Groq LPU** now supports **LlamaIndex**, including `llama2` and `Mixtral` models, aimed at improving Large Language Model (LLM) generation with a comprehensive [cookbook guide](https://t.co/zBiBlgadVh) provided for streamlining application workflows.
- **LlamaIndex Services Expand and Optimize**: **LlamaParse** reported significant usage leading to a usage cap increase and updates towars uncapped self-serve usage, while a new strategy using LLMs for alpha parameter adjustment in hybrid search has been shared in [this insight](https://t.co/39Lk5nEOoc). Plus, a **RAG architecture** combining structured and unstructured data by `@ClickHouseDB` has been highlighted, which can be read about [here](https://t.co/oy79TexCYR).
- **Technical Insights and Clarifications Heat Up LlamaIndex Discussions**: Indexing the latest **LlamaIndex docs** is in consideration with *mendable* mentioned as a tool for docs, while `@cheesyfishes` comments on an anticipated refactor of `CallbackHandler` in Golang. A combination of **FlagEmbeddingReranker** with **CohereReranker** was identified as a tactic despite the absence of comparison metrics, and `@cheesyfishes` explained that while **LlamaIndex** serves data to LLMs, **Langchain** is a more encompassing library.
- **Model Behaviors Questioned Within AI Community**: There's a discussion about **model decay** with `@.sysfor` noting degrading outputs from their models and `@cheesyfishes` reinforcing that models do not decay but input issues can affect performance. The concern extends to fine-tuned models underperforming when compared to baseline models.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **Claude Encounters a Conversational Hiccup**: **Claude models** from Anthropics were reported to have an error with chats having more than 8 alternating messages. The problem was acknowledged by `@louisgv` with a promise of an upcoming fix.

- **Turn Taking Tweaks for OpenRouter**: `@alexatallah` suggested a workaround for Claude's prompt errors involving changing the initial assistant message to a system message. Development is ongoing to better handle conversations initiated by the assistant.

- **OpenRouter's Rate Limit Relay**: When asked about rate limits for article generation, `@alexatallah` clarified that individually assigned API keys for **OpenRouter** users would have separate limits, presumably allowing adequate collective throughput.

- **Mistral's Suspected Caching Unearthed**: Users noticed repeat prompt responses from **Mistral models** suggesting caching might be at play. `@alexatallah` confirmed the possibility of query caching in Mistral's API.

- **Prepaid Payment Puzzles for OpenRouter**: `@fakeleiikun` raised a question about the acceptance of prepaid cards through **OpenRouter**, and `@louisgv` responded with possible issues tied to Stripe's fraud prevention mechanisms, indicating mixed support.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **Benchmarking Bounties**: [`@hdcharles_74684`](https://github.com/HDCharles) improved a [benchmark script](https://gist.github.com/HDCharles/a7fc12b31702cf963d8453e0da157296) for **Triton** kernels, which may outperform cuBLAS in specific scenarios such as batch sizes greater than 1, pertinent to applications like **sdxl-fast**. In light of potential Triton optimizations, focusing on technologies such as **Torch.compile** could address bottlenecks when handling batch size of 2.
  
- **Triton Turmoil and Triumphs**: Users encountered debugging issues with Triton versions **3.0.0** and **2.2.0**; a workaround involved setting the `TRITON_INTERPRET` environment variable. Moreover, stability concerns were voiced regarding Triton's unpredictable segfaults compared to **CUDA**, prompting a request for comparative examples to understand the inconsistencies.

- **FP8 Intrinsics Intact**: In response to a query based on a [tweet](https://twitter.com/cis_female/status/1763221499551604995), `@zippika` clarified that **FP8 intrinsics** are still documented in the [CUDA math API docs](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html), noting that FP8 is primarily a data format and not universally applied for compute operations.

- **Compiler Conundrums**: In the realm of deep learning, skepticism was expressed about the usefulness of *polyhedral compilation* for optimizing sharding. This ties into the broader discussion about defining **cost functions**, the complexity of mapping DL programs to hardware, and whether top AI institutions are tackling these optimization challenges.

- **Ring Attention Riddles**: A comparison was proposed for validating the correctness and performance of [Ring Attention implementations](https://github.com/lucidrains/ring-attention-pytorch), as potential bugs were noted in the backward pass, and GPU compatibility issues surfaced. User `@iron_bound` suggested there may be breakage in the implementation per [commit history analysis](https://github.com/lucidrains/ring-attention-pytorch/commits/main/), stressing the need for careful code review and debugging.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord Summary

- **European Independence and Open-Weight Ambitions**: Arthur Mensch emphasized the commitment to open-weight models, specifically mentioning **1.5k H100s**, and highlighted a reselling deal with Microsoft. Le Chat and Mistral Large are attracting attention on La Plateforme and Azure, showing growth and a quick development approach. [Here are the details](https://x.com/arthurmensch/status/1762818733016322168?s=46).

- **Starcoder2 Breaks New Ground**: The Stack v2, featuring over **900B+ tokens**, is the powerhouse behind StarCoder2, which flaunts a 16k token context and is trained on more than **4T+ tokens**. It represents a robust addition to the coding AI community with fully open code, data, and models. [Explore StarCoder2](http://hf.co/bigcode/starcoder2-15b).

- **Meta's Upcoming Llama 3**: A report from Reuters indicates that Meta is gearing up to launch **Llama 3** in July, signaling a potential shake-up in the AI language model landscape. The Information provided additional details on this forthcoming release. [Further information available here](http://reut.rs/3TgBgFJ).

- **DeepMind CEO's Insights Captivate Nathan**: Nathan Lambert tuned into a podcast featuring Demis Hassabis of Google DeepMind, covering topics such as **superhuman AI scaling, AlphaZero combining with LLMs, and the intricacies of AI governance**. These insights are accessible on various platforms including [YouTube](https://youtu.be/qTogNUV3CAI) and [Spotify](https://open.spotify.com/episode/6SWbwjYPs5WevIoCCiSByS?si=nCVFSRr7QGGI_STgbrOBDA).

- **Open AI and Personal Perspectives**: The conversation between Nathan and Mike Lambert touched on the nature and importance of open AI and the differing thought models when compared to platforms like Twitter. Additionally, Mike Lambert, associated with Anthropic, expressed a preference to engage in dialogues personally rather than as a company representative.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **A Buzz for Benchmarking Automation**: Engineers `@ampdot` and `@dare.ai` are keen on exploring automated benchmark scripts, with the latter tagging another user for a possible update on such a tool.
- **Springtime Hopes for Llama 3**: `@res6969` predicts a spring release for **Llama 3**, yet hints that the timeline could stretch, while `@potrock` is hopeful for last-minute updates, particularly intrigued by the potential integration of **Gemini ring attention**.
- **The Testing Time Dilemma**: `@jeffreyw128` voices the challenge of time investment needed for comprehensive testing of new **LLMs**, aiming for an adequate "vibe check" on each model.
- **ChatGPT Search Speculation Surfaces**: Rumors of an impending OpenAI update to ChatGPT's web search features were mentioned by `@jeffreyw128`, with `@res6969` seeking more reliable OpenAI intel and curious about resources for deploying **codeinterpreter** in production.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **DiscoLM Template Usage Critical**: `@bjoernp` underscored the significance of utilizing the DiscoLM template for proper chat context tokenization, pointing to the [chat templating documentation](https://huggingface.co/docs/transformers/main/en/chat_templating#introduction) on Hugging Face as a crucial resource.

- **Chunking Code Struggles with llamaindex**: `@sebastian.bodza` encountered severe issues with the llamaindex chunker for code, which is outputting one-liners despite the `chunk_lines` setting, suggesting a bug or a need for tool adjustments.

- **Pushing the Boundaries of German AI**: `@johannhartmann` is working on a German RAG model using Deutsche Telekom's data, seeking advice on enhancing the German-speaking Mistral 7b model reliability, while `@philipmay` delved into generating negative samples for RAG datasets by instructing models to fabricate incorrect answers.

- **German Language Models Battleground**: A debate emerged over whether Goliath or DiscoLM-120b is more adept at German language tasks, with `@philipmay` and `@johannhartmann` weighing in; `@philipmay` posted the [Goliath model card on Hugging Face](https://huggingface.co/alpindale/goliath-120b) for further inspection.

- **Benchmarking German Prompts and Models**: `@crispstrobe` revealed that EQ-Bench now includes German prompts, with the **GPT-4-1106-preview** model leading in performance, and provided a [GitHub pull request](https://github.com/EQ-bench/EQ-Bench/pull/12) link; they mentioned translation scripts being part of the benchmarks, effectively translated by ChatGPT-4-turbo.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **JSON Judo Techniques Remain Hazy**: `@dbreunig` verbalized the common challenge of dealing with **noisy JSON responses**, though specifics on the cleanup techniques or functions were not disclosed.
- **Silencing Claude's Small Talk**: `@justinpinkney` recommended using initial characters like `<rewrite>` based on [Anthropic's documentation](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites) to circumvent **Claude's** default lead-in phrases such as "Sure here's a...".
- **Brevity Battle with Claude**: `@derekpwillis` experimented with several strategies for attaining shorter outputs from Claude, including forcing the AI to begin responses with `{`, but admitted that Claude still tends to include prefatory explanations.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

**An Unexpected Recruitment Approach**: User `.papahh` directly messaged **@1117586410774470818**, indicating a job opportunity and showing enthusiasm for their potential involvement.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Value Hunting Across Species**: `@taodoggy` is inviting collaboration on a project to probe into the **biological and evolutionary origins of shared values among species**, refine value definitions, and explore their manifestation in various cultures. The project overview is accessible via a [Google Docs link](https://docs.google.com/document/d/1A2ZdM1IBv0_5nN1pujyCvmoCGepETmWFRPmAmdjkqqA/edit?usp=drivesdk).



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1212326558795042857) (1070 messages🔥🔥🔥): 

- **Discord Detects Spammer**: Users noticed messages flagged for likely spam in the chat, particularly from `@kquant`, who was reported for messaging over 100 people with the same message, triggering Discord's spam detection system.
- **Exploring ChatGPT Performance**: Users like `@itsme9316` and `@notreimu` discussed their varying experiences with ChatGPT models. Some noted that GPT-4's API was unreliable for them compared to alternatives like Mixtral or Miqu models.
- **Model Merging Conversations**: Various users, including `@itsme9316` and `@al_lansley`, discussed model merging and how it doesn't always result in smarter models. There was consensus that merging often depends on luck and the models' compatibility.
- **Concerns Over Contaminated Training Data**: Users such as `@itsme9316` expressed concerns about modern LLMs potentially being contaminated with outputs from other models like OpenAI's, which could affect quality and reliability.
- **Quantization and Model Performance**: There was discussion led by `@notreimu` and `@aiwaldoh` about the performance differences between high-parameter models with low bit-per-weight (bpw) quantization and smaller models with higher bpw. Users shared varying experiences with different quantized models.

**Links mentioned**:

- [Database Search](https://search.0t.rocks/): Search our database of leaked information. All information is in the public domain and has been compiled into one search engine.
- [A look at Apple’s new Transformer-powered predictive text model](https://jackcook.com/2023/09/08/predictive-text.html): I found some details about Apple’s new predictive text model, coming soon in iOS 17 and macOS Sonoma.
- [Microsoft-backed OpenAI valued at $80bn after company completes deal](https://www.theguardian.com/technology/2024/feb/16/microsoft-openai-valuation-artificial-intelligence): Company to sell existing shares in ‘tender offer’ led by venture firm Thrive Capital, in similar deal as early last year
- [Sad GIF - Sad - Discover &amp; Share GIFs](https://tenor.com/view/sad-gif-7523306793289960933): Click to view the GIF
- [writing-clear.png · ibm/labradorite-13b at main](https://huggingface.co/ibm/labradorite-13b/blob/main/writing-clear.png): no description found
- [And death shall have no dominion](https://poets.org/poem/and-death-shall-have-no-dominion): And death shall have no dominion. / Dead men naked they shall be one
- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO): no description found
- [Uncensored Models](https://erichartford.com/uncensored-models): I am publishing this because many people are asking me how I did it, so I will explain. https://huggingface.co/ehartford/WizardLM-30B-Uncensored https://huggingface.co/ehartford/WizardLM-13B-Uncensore...
- [BioMistral/BioMistral-7B · Hugging Face](https://huggingface.co/BioMistral/BioMistral-7B): no description found
- [NousResearch/Nous-Hermes-2-SOLAR-10.7B · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B): no description found
- [adamo1139 (Adam)](https://huggingface.co/adamo1139): no description found
- [p1atdev/dart-v1-sft · Hugging Face](https://huggingface.co/p1atdev/dart-v1-sft): no description found
- [google/gemma-7b-it · Buggy GGUF Output](https://huggingface.co/google/gemma-7b-it/discussions/38#65d7b14adb51f7c160769fa1): no description found
- [Attack of the stobe hobo.](https://www.youtube.com/watch?v=OS3kvekRDn4&list=PLooFWMJbNy6HbqeniUS1wCYG5agvf5vnT): Full movie. Please enjoy. Rip Jim Stobe.
- [Fred again..: Tiny Desk Concert](https://www.youtube.com/watch?v=4iQmPv_dTI0): Teresa Xie | April 10, 2023When Fred again.. first proposed a Tiny Desk concert, it wasn&#39;t immediately clear how he was going to make it work — not because h...
- [My Fingerprint- Am I Unique ?](https://amiunique.org/fingerprint): no description found
- [GitHub - MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone): Contribute to MooreThreads/Moore-AnimateAnyone development by creating an account on GitHub.
- [adamo1139/rawrr_v2 · Datasets at Hugging Face](https://huggingface.co/datasets/adamo1139/rawrr_v2): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1212329402051076176) (511 messages🔥🔥🔥): 

- **LLM Roleplay Discussion**: Users discussed the effectiveness of using Large Language Models (LLMs) for role-playing characters, including techniques for crafting character identities, such as telling the LLM "you are a journalist" to improve performance. `@nathaniel__` suggested successful strategies involve assigning roles and detailed personalities and `@maldevide` shared a prompt structuring approach using `#define` syntax.
  
- **Character Consistency**: Several users, including `@shanman6991` and `@superking__`, explored whether character consistency can be improved by giving LLMs detailed backstories and personality traits. There was particular interest in techniques to allow characters to lie or scheme convincingly within role-play scenarios.

- **Prompt Engineering Tactics**: `@maldevide` discussed the use of proper names and declarative statements in prompts to guide LLMs into desired patterns of conversation, while `@superking__` provided examples of instruct vs. pure chat mode setups for better model guidance.

- **Model Selection for Roleplay**: Users like `@superking__` indicated a preference for specific models such as miqu and mixtral for role-play purposes, often eschewing the use of system prompts. There was also mention of the potential for models to become less coherent with longer context lengths, and strategies to offset this were discussed.

- **Naming Conventions in LLMs**: `@gryphepadar` and `@maldevide` observed that certain names, like "Lyra" and "Lily", seem to be particularly common in responses when LLMs are prompted to generate character names, leading to some speculation about the training data's influence on these naming trends.

**Links mentioned**:

- [Let Me In Eric Andre GIF - Let Me In Eric Andre Wanna Come In - Discover &amp; Share GIFs](https://tenor.com/view/let-me-in-eric-andre-wanna-come-in-gif-13730108): Click to view the GIF
- [Sad Smoke GIF - Sad Smoke Pinkguy - Discover &amp; Share GIFs](https://tenor.com/view/sad-smoke-pinkguy-depressed-smoking-gif-22804675): Click to view the GIF
- [Why Have You Forsaken Me? GIF - Forsaken Why Have You Forsaken Me Sad - Discover &amp; Share GIFs](https://tenor.com/view/forsaken-why-have-you-forsaken-me-sad-depressed-alone-gif-12497399): Click to view the GIF
- [maldv/conversation-cixot · Datasets at Hugging Face](https://huggingface.co/datasets/maldv/conversation-cixot): no description found
- [Hawk Eye Dont Give Me Hope GIF - Hawk Eye Dont Give Me Hope Clint Barton - Discover &amp; Share GIFs](https://tenor.com/view/hawk-eye-dont-give-me-hope-clint-barton-avengers-gif-14260447): Click to view the GIF
- [GitHub - UltiRTS/PrometheSys.vue](https://github.com/UltiRTS/PrometheSys.vue): Contribute to UltiRTS/PrometheSys.vue development by creating an account on GitHub.
- [GitHub - predibase/lorax: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs](https://github.com/predibase/lorax): Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs - predibase/lorax

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1212407419171504138) (86 messages🔥🔥): 

- **Perplexity AI as a New Tool**: User `@icecream102` suggested trying out [Perplexity AI](https://www.perplexity.ai) as a resource.
- **Budget Training with QLoRA**: `@dirtytigerx` advised that training large language models like GPT can be expensive and suggested using techniques like QLoRA to limit hardware requirements, though noting it would still take many hours of compute.
- **Training and Inference Cost Estimates**: In a discussion on estimating GPU hours for training and inference, `@dirtytigerx` recommended conducting a tiny test run and looking at published papers for benchmarks.
- **Model Training Dynamics Discussed**: `@cogbuji` questioned training a model with a static low validation loss, prompting `@dirtytigerx` to suggest altering the validation split and taking deduplication steps to address discrepancies.
- **Model Generalization and Hallucination Concerns**: `@dirtytigerx` and `@cogbuji` discussed training model generalization and the inevitable problem of hallucination during inference, suggesting the use of retrieval mechanisms and further evaluation strategies.

**Links mentioned**:

[cogbuji/Mr-Grammatology-clinical-problems-Mistral-7B-0.5 · Hugging Face](https://huggingface.co/cogbuji/Mr-Grammatology-clinical-problems-Mistral-7B-0.5): no description found

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1212432912218193990) (6 messages): 

- **Tensor Dimension Misalignment Issue**: `@falconsfly` pointed out that an issue arose due to a **single bit being misplaced or misaligned**, resulting in incorrect tensor dimensions.
- **Appreciation Expressed for Information**: `@222gate` thanked `@falconsfly` for sharing the information about the tensor dimension problem.
- **Query about Slerp or Linear Techniques**: `@222gate` asked if the discussed merging techniques involved **spherical linear interpolation (slerp)** or just linear ties.
- **Reflection on Diffusion Test Techniques**: In response, `@alphaatlas1` mentioned not being certain about `@222gate`'s specific query but shared that their diffusion test used **dare ties** and speculated that a HuggingFace test may have involved dare task arithmetic.
- **Recommendation for Concatenation in Merging**: `@alphaatlas1` suggested trying **concatenation** for anyone doing the *peft* merging, stating it **works well** and noting there's no full-weight merging analogue for it.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1212398390638551080) (8 messages🔥): 

- **Eager for Collaboration**: `@wolfsauge` expresses enthusiasm to learn from `@falconsfly`, anticipating a discussion on fresh ideas for enhancement after dinner.

- **No GPU, No Speed?**: `@dirtytigerx` states that without a GPU, speeding up processes is challenging, offering no alternative solutions for performance improvement.

- **APIs for Acceleration**: `@tom_lrd` suggests using APIs as an alternative to speed up processes, listing multiple services like huggingface, together.ai, and mistral.ai.

- **Looking Beyond Colab for Hosted Notebooks**: Despite `@dirtytigerx` mentioning the lack of hosted notebooks on platforms provided by cloud providers, `@falconsfly` points out that Groq.com offers fast inference.

- **Modular MAX Enters the Game**: `@dirtytigerx` shares news about the general availability of the modular MAX platform, [announcing the developer edition preview](https://www.modular.com/blog/announcing-max-developer-edition-preview) and its vision to democratize AI through a unified, optimized infrastructure.

**Links mentioned**:

[Modular: Announcing MAX Developer Edition Preview](https://www.modular.com/blog/announcing-max-developer-edition-preview): We are building a next-generation AI developer platform for the world. Check out our latest post: Announcing MAX Developer Edition Preview

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1212326538645602325) (992 messages🔥🔥🔥): 

- **NVIDIA's Chat with RTX Demo Criticized**: Users like `@netrve` expressed disappointment with NVIDIA's "Chat with RTX" demo, which was meant to showcase retrieval-augmented generation (RAG) capabilities. The demo, which limited context size to 1024 tokens, faced issues with retrieving correct information and delivering coherent answers. NVIDIA's use of LangChain in the reference architecture for RAG was also questioned.

- **OpenAI and Meta Licensing Discussions**: There was a heated discussion spearheaded by `@i_am_dom` and `@netrve` regarding Mistral AI's usage of Meta's LLaMa model, potential licensing issues, and implications of commercial use. The consensus suggested that an undisclosed agreement between Mistral and Meta was possible, given the seeming compliance with Meta's licensing terms.

- **Conversations about Mistral AI's Open Weight Models**: `@mrdragonfox`, `@tarruda`, and others discussed Mistral AI's commitment to open weight models and speculated about future releases following the Mistral-7B model. The community expressed trust and expectations towards Mistral for providing more open weight models.

- **RAG Implementation Challenges Highlighted**: Several users, including `@mrdragonfox` and `@shanman6991`, discussed the complexities of implementing RAG systems effectively. They mentioned the significant impact of the embedding model on RAG performance and the difficulty in achieving perfection with RAG, often taking months of refinement.

- **Mistral AI and Microsoft Deal Scrutinized**: An investment by Microsoft in Mistral AI raised discussions about the size of the investment and its implications for competition in the AI space. `@ethux` shared information hinting that the investment was minimal, while `@i_am_dom` raised concerns about Microsoft's cautious approach due to potential complexities with open-source models like Miqu.

**Links mentioned**:

- [What Is Retrieval-Augmented Generation aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/): Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard): no description found
- [Klopp Retro GIF - Klopp Retro Dancing - Discover &amp; Share GIFs](https://tenor.com/view/klopp-retro-dancing-liverpool-champions-gif-19224858): Click to view the GIF
- [Basic RAG | Mistral AI Large Language Models](https://docs.mistral.ai/guides/basic-RAG/): Retrieval-augmented generation (RAG) is an AI framework that synergizes the capabilities of LLMs and information retrieval systems. It&#x27;s useful to answer questions or generate content leveraging ...
- [mlabonne/NeuralHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B): no description found
- [Legal terms and conditions](https://mistral.ai/terms/#terms-of-use): Terms and conditions for using Mistral products and services.
- [Microsoft made a $16M investment in Mistral AI | TechCrunch](https://techcrunch.com/2024/02/27/microsoft-made-a-16-): Microsoft is investing €15 million in Mistral AI, a Paris-based AI startup working on foundational models.
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/#json-mode)): We provide client codes in both Python and Javascript.
- [NVIDIA Chat With RTX](https://www.nvidia.com/fr-fr/ai-on-rtx/chat-with-rtx-generative-ai/): Personnalisez et déployez votre chatbot d&#39;IA.
- [Microsoft made a $16M investment in Mistral AI | TechCrunch](https://techcrunch.com/2024/02/27/microsoft-made-a-16-million-investment-in-mistral-ai/amp/): Microsoft is investing €15 million in Mistral AI, a Paris-based AI startup working on foundational models.
- [Mistral Large vs GPT4 - Practical Benchmarking!](https://www.youtube.com/watch?v=IH2htfsciO4): ➡️ One-click Fine-tuning &amp; Inference Templates: https://github.com/TrelisResearch/one-click-llms/➡️ Trelis Function-calling Models (incl. OpenChat 3.5): http...
- [Short Courses](https://www.deeplearning.ai/short-courses/): Take your generative AI skills to the next level with short courses from DeepLearning.AI. Enroll today to learn directly from industry leaders, and practice generative AI concepts via hands-on exercis...

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1212377622764589056) (12 messages🔥): 

- **More Meaningful Error Messages on Mistral**: `@lerela` addressed an issue regarding system limitations, stating that a certain operation is not permitted with the **large** model, but users will now receive a **more meaningful error message**.
- **Discussion on System/Assistant/User Sequence**: `@skisquaw` remarked on having to change the sequence from system/assistant/user to **user/assistant/user** due to the model treating the first user input as a system one, despite a functionality need where assistant prompts follow system commands.
- **Quantization Packs Mistral-7B Parameters**: `@chrismccormick_` inquired about the parameter count of **Mistral-7B**, originally tallying only around 3.5B. They later deduced that **4-bit quantization** likely halves the tensor elements.
- **Large Code Segments Questioned for Mistral**: `@frigjord` contemplated whether querying long code segments, especially more than 16K tokens, might pose a problem for **Mistral** models.
- **Complex SQL Queries with Mistral-7B**: `@sanipanwala` asked about generating complex SQL queries with **Mistral-7B**, and `@tom_lrd` responded affirmatively, providing advice on formulating the queries and even giving an example for creating a sophisticated SQL query.
  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1212354255478853672) (174 messages🔥🔥): 

- **Mistral Deployment Conundrum**: `@arthur8643` inquired about hardware requirements for running **Mistral 8x7B locally**, contemplating a system upgrade. Users `@_._pandora_._` and `@mrdragonfox` advised that his current setup wouldn't suffice, recommending at least 100GB of VRAM for full precision deployment, and suggesting the use of services like [together.ai](https://together.ai) for assistance.

- **Debates on Optimal Server Specs**: `@latoile0221` sought advice on server specifications for token generation, considering a dual CPU setup and RTX 4090 GPU. The user received mixed responses regarding the importance of CPU versus GPU; `@ethux` stressed the GPU's significance for inference tasks while discussions circled around the necessity of substantial VRAM for full precision models.

- **Quantization Qualms and GPU Capabilities**: Various participants expressed that quantized models underperform, with `@frigjord` and `@ethux` noting that quantized versions aren't worthwhile for coding tasks. The consensus emerged that substantial VRAM (near 100GB) is needed to run non-quantized, full-precision models effectively.

- **Self-Hosting, Model Types, and AI Limitations**: Dialogue ensued about the practicalities of self-hosting AI models like **Mixtral**, with mentions of utilizing quant versions and alternatives like GGUF format. Users including `@ethux` and `@sublimatorniq` shared experiences, with a focus on the limitations of quantized models and better performance of full models on high-spec hardware.

- **On the Topic of Specialized AI Models**: The discussion touched on the potential advantages and challenges of training a specialized JS-only AI model. `@frigjord` and `@mrdragonfox` debated the effectiveness and handling of such focused models, with general agreement on the extensive work required to clean and prep datasets for any specialized AI training.

**Links mentioned**:

- [Jurassic Park GIF - Jurassic Park World - Discover &amp; Share GIFs](https://tenor.com/view/jurassic-park-world-velociraptor-clever-gif-25116052): Click to view the GIF
- [starling-lm](https://ollama.com/library/starling-lm): Starling is a large language model trained by reinforcement learning from AI feedback focused on improving chatbot helpfulness.
- [Tags · mixtral](https://ollama.com/library/mixtral/tags): A high-quality Mixture of Experts (MoE) model with open weights by Mistral AI.

  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1212355671144669235) (76 messages🔥🔥): 

- **Typo Alert in Notebook**: `@foxalabs_32486` identified a typo in the `prompting_capabilities.ipynb` notebook, where an extra "or" was present. The correct text should read *"Few-shot learning or in-context learning is when we give a few examples in the prompt..."*
- **Fix Confirmation**: In response to `@foxalabs_32486`'s notice, `@sophiamyang` acknowledged the error and confirmed the fix.
- **Typos Add Human Touch**: `@foxalabs_32486` mused about using occasional typos to make AI-generated content appear more human, sparking a discussion on the ethics of making AI seem human with `@mrdragonfox`.
- **Ethics over Earnings**: `@mrdragonfox` declined projects aimed at humanizing AI beyond ethical comfort, underscoring a preference to choose integrity over financial gain.
- **AI Industry Hiring Challenges**: `@foxalabs_32486` discussed the difficulties in hiring within the AI industry due to a shortage of skilled professionals and the rapid expansion of knowledge required.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1212387019733467189) (15 messages🔥): 

- **Limiting Model Answers to Specific Documents**: `@aaronbarreiro` inquired about constraining a chatbot to only provide information from a specific document, such as one about wines, and not respond about unrelated topics like pizza.
- **The Challenge of Controlling LLMS**: `@mrdragonfox` explained that language models like LLMS will likely hallucinate answers, because they are designed fundamentally as next token predictors, thus a robust system prompt is vital to direct responses.
- **Language Models as Stateless Entities**: `@mrdragonfox` highlighted the stateless nature of language models, meaning they don't retain memory like a human would, and if pushed beyond their token limit—specifically mentioned the 32k context—they will forget earlier information.
- **Strategies to Maintain Context Beyond Limits**: `@mrdragonfox` discussed strategies to circumvent the context limitation, such as using function calling or retrieval-augmented generation (RAG), but acknowledged these methods are more complex and don't work directly out-of-the-box.
- **Fine-Tuning Time Depends on Dataset Size**: When `@atip` asked about the time required to fine-tune a 7B parameter model on H100 hardware, `@mrdragonfox` stated it varies based on dataset size, implying the duration can't be estimated without that information.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1212387381421015090) (7 messages): 

- **Teaching Economics with AI**: `@patagonia50` shared about creating an app for an intermediate microeconomics course that provides instant personalized feedback by making API calls to **gpt-4-vision-preview** and **Mistral** models. The app, which adapts to different questions and rubrics via a JSON file, has been deployed on Heroku and is still being refined, with future plans to expand its capabilities with Mistral AI models.

- **Interest Expressed in Educational App**: `@akshay_1` showed interest in `@patagonia50`'s educational app, asking if there was a GitHub repository available for it.

- **Open Source Plans**: In response to `@akshay_1`, `@patagonia50` indicated that there isn't a GitHub repository yet but plans to create one for the educational app.

- **Request for a Closer Look**: `@akshay_1` expressed a desire for a sneak peek at `@patagonia50`'s educational app, demonstrating enthusiasm for the project.

**Links mentioned**:

- [cogbuji/Mr-Grammatology-clinical-problems-Mistral-7B-0.5 · Hugging Face](https://huggingface.co/cogbuji/Mr-Grammatology-clinical-problems-Mistral-7B-0.5): no description found
- [Use Mistral AI Large Model Like This: Beginner Friendly](https://www.youtube.com/watch?v=Rveib4aYtew.): We learn the features of High Performing Mistral Large and do live coding on Chat Completions with Streaming and JSON Mode. The landscape of artificial intel...

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1212767292195078204) (2 messages): 

- **Seeking the Google Million Context AI**: User `@j673912` inquired about how to access the elusive **Google 1M Context AI**.
- **Insider Connection Required**: `@dawn.dusk` recommended having direct contact with someone from **Deepmind** to gain access.
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1212378077108117574) (41 messages🔥): 

- **Mistral Function Calls Require Adjustments**: `@michaelhunger` discussed challenges with the Mistral function calling mechanism, noting the need for patches and system messages. Specifically, Mistral's behavior contrasts with expectations, often preferring additional tool calls over answering the user's query directly.

- **Clarifying `tool_choice` Behavior**: `@liebke` expressed confusion over the behavior of `tool_choice="auto"` in the context of Mistral's function calling, as the setting does not seem to trigger tool calls as anticipated. `@sophiamyang` suggested that "auto" should work as intended, prompting a request for Liebke's implementation details for further troubleshooting.

- **Inconsistencies in Mistral Function Calling**: `@alexclubs` provided feedback on integrating Mistral Function Calling into Profound Logic, noticing differences from OpenAI's tool behavior and a lack of consistency in when functions are triggered.

- **Reproducibility of Outputs on Mistral's Platform Uncertain**: `@alexli3146` inquired about seedable outputs for reproducibility, while `@foxalabs_32486` and `@sublimatorniq` discussed potential issues and existing settings in the API that may affect it.

- **Mistral Message Roles Must Follow Specific Order**: After discussing error messages encountered with "mistral-large-latest," `@not__cool` discovered that wrapping a user message with two system messages is not supported, as confirmed by `@lerela`. However, `@skisquaw` successfully used the user/assistant format with the system role message in the first user role statement.

**Links mentioned**:

- [Technology](https://mistral.ai/technology/#models): Frontier AI in your hands
- [AI Assistants are the Future | Profound Logic](https://www.profoundlogic.com/ai/): With Profound AI, you can enhance your legacy applications with natural language AI assistants in just 3 steps.
- [AI Assistants are the Future | Profound Logic](https://www.profoundlogic.com/ai/).): With Profound AI, you can enhance your legacy applications with natural language AI assistants in just 3 steps.
- [GitHub - liebke/mechanician: Daring Mechanician is a Python library for building tools that use AI by building tools that AIs use.](https://github.com/liebke/mechanician/tree/main): Daring Mechanician is a Python library for building tools that use AI by building tools that AIs use. - liebke/mechanician
- [mechanician/packages/mechanician_mistral/src/mechanician_mistral/mistral_ai_connector.py at main · liebke/mechanician](https://github.com/liebke/mechanician/blob/main/packages/mechanician_mistral/src/mechanician_mistral/mistral_ai_connector.py): Daring Mechanician is a Python library for building tools that use AI by building tools that AIs use. - liebke/mechanician
- [mechanician/examples/notepad/src/notepad/main.py at main · liebke/mechanician](https://github.com/liebke/mechanician/blob/main/examples/notepad/src/notepad/main.py): Daring Mechanician is a Python library for building tools that use AI by building tools that AIs use. - liebke/mechanician

  

---


### Mistral ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/1212716795262402570) (1 messages): 

- **Mark Your Calendars for Evaluation Talk**: `@sophiamyang` invites everyone to the next office hour on **Mar. 5 at 5pm CET** with a focus on **evaluation and benchmarking**. They express interest in learning about different evaluation strategies and benchmarks used by participants.
  

---


### Mistral ▷ #[le-chat](https://discord.com/channels/1144547040454508606/1211692704363323453/1212337909538234448) (423 messages🔥🔥🔥): 

- **Le Chat Model Limit Discussions**: User `@alexeyzaytsev` inquired about the limits for Le Chat on a free account. Although currently undefined, `@ethux` and `@_._pandora_._` speculated that future restrictions might mimic OpenAI's model, with advanced features potentially becoming paid services.

- **Mistral on Groq Hardware**: `@foxalabs_32486` asked about plans to run Large on Groq hardware, while `@ethux` noted Groq's memory limitations. `@foxalabs_32486` provided a [product brief from Groq](https://groq.com/wp-content/uploads/2022/10/GroqCard%E2%84%A2-Accelerator-Product-Brief-v1.5-.pdf), highlighting potential misconceptions about their hardware's capabilities.

- **Mistral's Market Position and Microsoft Influence**: In an extensive discussion, users `@foxalabs_32486` and `@mrdragonfox` shared their perceptions of Mistral's market positioning and the influence of Microsoft's investment. They touched on topics like strategic hedging, the potential impact on OpenAI, and the speed of Mistral's achievements.

- **Feedback for Le Chat Improvement**: Several users, including `@sophiamyang`, engaged in discussing ways to improve Le Chat. Suggestions included a "thumb down" button for inaccurate responses (`@jmlb3290`), ease of switching between models during conversations (`@sublimatorniq`), features to manage token counts and conversation context (`@_._pandora_._`), preserving messages on error (`@tom_lrd`), and support for image inputs (`@foxalabs_32486`).

- **Debating Efficiency of Low-Bitwidth Transformers**: Users, especially `@foxalabs_32486` and `@mrdragonfox`, debated the implications of a low-bitwidth transformer research paper, discussing potential boosts in efficiency and the viability of quickly implementing these findings. They mentioned the work involved in adapting existing models and the speculative nature of immediate hardware advancements.

**Links mentioned**:

- [Technology](https://mistral.ai/technology/#models).): Frontier AI in your hands
- [Why 2024 Will Be Not Like 2024](https://medium.com/@unravelingentertainment/why-2024-will-be-not-like-2024-8799121ee791): In the ever-evolving landscape of technology and education, a revolutionary force is poised to reshape the way we learn, think, and…
- [Unsloth update: Mistral support + more](https://unsloth.ai/blog/mistral-benchmark#Benchmark%20tables): We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

  

---


### Mistral ▷ #[failed-prompts](https://discord.com/channels/1144547040454508606/1212715819159654451/1212715907898671166) (6 messages): 

- **Instructions for Reporting Failed Prompts**: `@sophiamyang` provided a template requesting details for reporting failed prompts, specifying information like `model`, `prompt`, `model output`, and `expected output`.

- **Witty Math Mistake Report**: `@blueaquilae` humorously flagged an issue regarding mathematics with the **Mistral Large** model with their comment, "math, halfway there (pun intended) on large chat".

- **Tongue-in-Cheek Query Confirmation**: In a playful exchange, `@notan_ai` queries whether a specific example counts as a failed prompt, to which `@blueaquilae` responds, "Synthetic data all the way?"

- **General Failures on le chat**: `@blacksummer99` reports that all versions of Mistral, including **Mistral next**, fail on a prompt given on le chat, without providing specifics.

- **Incomplete Issue Indication**: `@aiwaldoh` mentions "Fondée en 2016?!" possibly pointing out an issue or confusion with the Mistral model's output, but no further details are provided.
  

---


### Mistral ▷ #[prompts-gallery](https://discord.com/channels/1144547040454508606/1212717054302625843/1212717273610063902) (5 messages): 

- **Invitation to Share Prompt Mastery**: User `@sophiamyang` welcomed everyone to share their most effective prompts, emphasizing prompt crafting as an art form and looking forward to seeing users’ creations.

- **Confusion About Channel Purpose**: After user `@akshay_1` simply mentioned "DSPy", `@notan_ai` responded with curiosity about "SudoLang" but expressed confusion regarding the purpose of the channel.

- **Possible Model Mention with Ambiguity**: The model name "Mistral next le chat" was mentioned twice by `@blacksummer99`, however, no further context or details were provided.
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1212334354085318666) (58 messages🔥🔥): 

- **Loader Choices for AI Models**: `@drinkoblog.weebly.com` pointed out that **lm studio** requires manual GUI interaction to start the API, which is impractical for websites. They recommend using alternative loaders such as **oobabooga** or **Jan dot ai** for automation on boot.

- **Automod Censorship on AI Discussions**: `@chonkyman777` reported their message was removed for showcasing problematic behavior by **Copilot AI**, and `@eskcanta` suggested reaching out to Discord mods via Modmail and reporting AI issues directly to OpenAI through their feedback form. Users debated the nuances of moderation and the scope of the rules in place.

- **Concerns Over Mistral and Uncensored Content**: `@dezuzel` shared a [YouTube video](https://www.youtube.com/watch?v=GyllRd2E6fg) discussing **Mistral**, an AI model considered powerful and uncensored. `@tariqali` raised questions about the implications of European AI regulation on Mistral, despite its promoted lack of censorship. `@chief_executive` compared **Mistral Large** to **GPT-4** and found the latter superior for coding tasks.

- **Fine-Tuning GPT-3.5 for Chatbot Use Case**: `@david_zoe` sought advice on fine-tuning **GPT-3.5-Turbo** to perform better than the baseline and maintain conversation flow, but faced challenges matching the performance of **GPT-4**. `@elektronisade` recommended examining common use cases and consulting ChatGPT with actual data for further guidance on fine-tuning.

- **Exploring Certifications for AI Specialization**: `@navs02`, a young developer, inquired about certifications for specializing in AI. `@dezuzel` and `.dooz` advised focusing on real-world projects over certifications and mentioned learning resources including courses by **Andrew Ng** and **Andrej Karpathy** on YouTube.

**Links mentioned**:

- [Chat model feedback](https://openai.com/form/chat-model-feedback): no description found
- [This new AI is powerful and uncensored… Let’s run it](https://www.youtube.com/watch?v=GyllRd2E6fg): Learn how to run Mistral&#39;s 8x7B model and its uncensored varieties using open-source tools. Let&#39;s find out if Mixtral is a good alternative to GPT-4, and lea...

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1212389923689341029) (21 messages🔥): 

- **Confusion Over API and File Uploads**: `@ray_themad_nomad` expressed frustration with the chatbot's inconsistent responses after uploading files and creating custom APIs, noting that methods that worked months ago seem to fail now.
- **Clarifying Document Size Limitations**: `@darthgustav.` pointed out that the chatbot can only read documents within context size, and it will summarize larger files, which spurred a debate with `@fawesum` who suggested that knowledge files can be accessed efficiently even if they are huge.
- **Seed Parameters Causing Inconsistent Outputs**: `@alexli3146` asked if anyone had success with getting reproducible output using the seed parameter, but shared that they haven't.
- **Security Measures with Web Browsing and Code Interpreter**: `@darthgustav.` explained that using python to search knowledge files with the Code Interpreter can disable web browsing in the instance which is a security decision.
- **Proper Channel for Sharing The Memory Game**: `@takk8is` shared a link to "The Memory" but was redirected by `@solbus` to share it in the dedicated channel to avoid it getting lost in the chat.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1212394179557195826) (391 messages🔥🔥): 

- **Prompt Engineering with MetaPrompting**: `@madame_architect` shared their work on annotating "MetaPrompting" research, enhancing their compiled list of prompt architecture papers to 42 total. The article details a method integrating meta-learning with prompts, aimed at improving initializations for soft prompts in NLP models. [MetaPrompting Discussion](https://chat.openai.com/share/9c8c70ca-362e-4d5f-a958-5cef30e7fd6f)

- **LaTeX and Katex in ChatGPT**: Several users, including `@yami1010` and `@eskcanta`, discussed the capabilities of ChatGPT in handling LaTeX and Katex for creating visual data representations, with a focus on math and flowchart diagrams.

- **Curly Brackets Saga in DALL-E 3**: Users such as `@darthgustav.` and `@beanz_and_rice` encountered an issue where DALL-E 3 payloads were not accepting standard curly brackets in JSON strings. They found a workaround by using escape coded curly brackets, which appeared to bypass the parser error.

- **Enhancing ChatGPT Creativity for Artistic Prompts**: When asked about improving creativity in artistic prompts, `@bambooshoots` and `@darthgustav.` suggested a multi-step iterative process and the use of semantically open variables to encourage less deterministic and more imaginative outputs from the AI.

- **Challenges with Custom ChatGPT File Reading**: `@codenamecookie` and `@darthgustav.` discussed issues with Custom ChatGPT's inconsistent ability to read '.py' files from its knowledge. They explored potential solutions such as converting files to plain text and avoiding unnecessary zipping for better AI parsing and responsiveness.

**Links mentioned**:

[Disrupting malicious uses of AI by state-affiliated threat actors](https://openai.com/blog/disrupting-malicious-uses-of-ai-by-state-affiliated-threat-actors): We terminated accounts associated with state-affiliated threat actors. Our findings show our models offer only limited, incremental capabilities for malicious cybersecurity tasks.

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1212394179557195826) (391 messages🔥🔥): 

- **Prompt Engineering Secrets**: `@yami1010` and `@eskcanta` shared insights on using Markdown, LaTeX, and KaTeX in prompts with ChatGPT for creating diagrams and flowcharts. They discussed the effectiveness of different diagram-as-code tools, with mentions of mermaid and mathplotlib, and the peculiarities of dealing with curly brackets in the DALL-E 3 parser.
- **MetaPrompting Annotated**: `@madame_architect` added MetaPrompting to their list of 42 annotated prompt architecture papers. The list, which can be found on the AI-Empower GitHub, is maintained to keep high-quality standards and is useful for researching prompt engineering.
- **The Curly Brackets Saga**: A long discussion revolving around the DALL-E 3 payload’s formatting issues with curly brackets (`{}`, `}`) in JSON strings took place, with multiple users like `@darthgustav.` and `@yami1010` noting failures during image generation. A solution involving Unicode escape codes was found, bypassing the parser error.
- **Custom ChatGPT File Reading**: In a conversation about Custom ChatGPT, `@codenamecookie` expressed confusion about the model's inconsistent ability to read Python files from its 'knowledge'. `@darthgustav.` recommended not zipping the files and converting them to plain text while maintaining Python interpretation, which might help the AI process the files better.
- **Boosting AI Creativity**: For enhancing AI-created artistic prompts, users like `@bambooshoots` and `@darthgustav.` suggested using a multi-step process to develop the scene and elicit more creative responses from GPT-3.5 and GPT-4. The inclusion of semantically open variables and iterative prompting would help provoke less deterministic and more unique outputs.

**Links mentioned**:

[Disrupting malicious uses of AI by state-affiliated threat actors](https://openai.com/blog/disrupting-malicious-uses-of-ai-by-state-affiliated-threat-actors): We terminated accounts associated with state-affiliated threat actors. Our findings show our models offer only limited, incremental capabilities for malicious cybersecurity tasks.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1212336436448075857) (484 messages🔥🔥🔥): 

- **Exploring Model Options**: Users are discussing various LLMs and their compatibility with specific GPUs, with a focus on coding assistance models such as Deepseek Coder 6.7B and StarCoder2-15B. For example, `@solusan.` is looking for the best model to fit an Nvidia RTX 40 series with 12 GB, currently considering Dolphin 2.6 Mistral 7B.

- **LM Studio GPU Compatibility Issues**: Several users like `@jans_85817` and `@kerberos5703` are facing issues running LM Studio with certain GPUs. Discussions revolve around LM Studio's compatibility mainly with newer GPUs, and older GPUs are presenting problems for which users are seeking solutions or alternatives.

- **Hugging Face Outage Impact**: A common issue reported by multiple members like `@barnley` and `@heyitsyorkie` is related to a network error when downloading models due to a Hugging Face outage affecting LM Studio's ability to search for models.

- **Image Recognition and Generation Queries**: Questions regarding image-related tasks surfaced, and `@heyitsyorkie` clarified that while LM Studio cannot perform image generation tasks, it is possible to work with image recognition through Llava models.

- **Hardware Discussions and Anticipations**: Users like `@pierrunoyt` and `@nink1` are discussing future hardware expectations for AI and LLMs, noting that current high-end AI-specific hardware may become more accessible with time.

**Links mentioned**:

- [GroqChat](https://groq.com/): no description found
- [no title found](http://192:168:0:100:1234/v1",): no description found
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/): Find, download, and experiment with local LLMs
- [Stop Shouting Arnold Schwarzenegger GIF - Stop Shouting Arnold Schwarzenegger Jack Slater - Discover &amp; Share GIFs](https://tenor.com/view/stop-shouting-arnold-schwarzenegger-jack-slater-last-action-hero-keep-your-voice-down-gif-21691190): Click to view the GIF
- [BLOOM](https://bigscience.huggingface.co/blog/bloom): Our 176B parameter language model is here.
- [Continue](https://continue.dev/): no description found
- [no title found](https://www.amazon.es/dp/B0CJGD3WYW/?smid=AO867S1490VMY&tag=idealoes-mp-21&linkCode=asn&creative=24634&creativeASIN=B0CJGD3WYW&ascsubtag=2024-02-29_3b7cdfefebea191422f7852137152b5717fd017224c1c4bbdd819d877500d11f&th=1&psc=1): no description found
- [GeForce GTX 650 Ti | Specifications | GeForce](https://www.nvidia.com/en-us/geforce/graphics-cards/geforce-gtx-650ti/specifications/): no description found
- [MaziyarPanahi/dolphin-2.6-mistral-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF · Hugging Face](https://huggingface.co/MaziyarPanahi/dolphin-2.6-mistral-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF): no description found
- [Specifications | GeForce](https://www.nvidia.com/en-us/geforce/graphics-cards/geforce-gtx-650/specifications/): no description found
- [02 ‐ Default and Notebook Tabs](https://github.com/oobabooga/text-generation-webui/wiki/02-%E2%80%90-Default-and-Notebook-Tabs): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui
- [Add support for StarCoder2 by pacman100 · Pull Request #5795 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5795): What does this PR do?  Adds support for StarCoder 2 models that were released recently.
- [bigcode/starcoder2-15b · Hugging Face](https://t.co/fM7GinxJBd): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b21bbx/this_is_pretty_revolutionary_for_the_local_llm/): no description found
- [Anima/air_llm at main · lyogavin/Anima](https://github.com/lyogavin/Anima/tree/main/air_llm): 33B Chinese LLM, DPO QLORA, 100K context, AirLLM 70B inference with single 4GB GPU - lyogavin/Anima
- [GitHub - MDK8888/GPTFast: Accelerate your Hugging Face Transformers 6-7x. Native to Hugging Face and PyTorch.](https://github.com/MDK8888/GPTFast): Accelerate your Hugging Face Transformers 6-7x. Native to Hugging Face and PyTorch. - MDK8888/GPTFast
- [itsdotscience/Magicoder-S-DS-6.7B-GGUF at main](https://huggingface.co/itsdotscience/Magicoder-S-DS-6.7B-GGUF/tree/main): no description found

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1212408835894157312) (61 messages🔥🔥): 

- **Seeking PDF chatbot guidance**: `@solenya7755` is looking to implement an accurate PDF chat bot with LM Studio and llama2 70B Q4 LLM, but experiences inaccuracies with hallucinated commands. `@nink1` suggests extensive prompt work and joining the AnythingLLM discord for further assistance.

- **StarCoder2 and The Stack v2 launch**: `@snoopbill_91704` shares [news](https://twitter.com/bigcodeproject/status/1762842312005026258) about the launch of StarCoder2 and The Stack v2 by ServiceNow, Hugging Face, and NVIDIA, noting a partnership with Software Heritage aligned with responsible AI principles.

- **Qualcomm releases 80 open source models**: `@misangenius` brings attention to [Qualcomm’s release of 80 open source AI models](https://huggingface.co/qualcomm), for vision, audio, and speech applications available on Huggingface.

- **Querying Models that prompt you with questions**: `@ozimandis` inquires about local LLMs that ask questions and has mixed results with different models, while `@nink1` shares success in getting models like dolphin mistral 7B q5 to ask provocative questions.

- **Best setup for business document analysis and writing**: `@redcloud9999` seeks advice on the best LLM setup for analyzing and writing business documents with a high-spec machine. `@heyitsyorkie` advises searching for GGUF quants by "TheBloke" on Huggingface and `@coachdennis.` suggests testing trending models.

**Links mentioned**:

- [qualcomm (Qualcomm)](https://huggingface.co/qualcomm): no description found
- [bigcode/starcoder2-15b · Hugging Face](https://huggingface.co/bigcode/starcoder2-15b): no description found
- [bigcode/the-stack-v2-train-full-ids · Datasets at Hugging Face](https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids): no description found
- [Pioneering the Future of Code Preservation and AI with StarCoder2](https://www.softwareheritage.org/2024/02/28/responsible-ai-with-starcoder2/): Software Heritage&#8217;s mission is to collect, preserve, and make the entire body of software source code easily available, especially emphasizing Free and Open Source Software (FOSS) as a digital c...

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1212348693664239616) (42 messages🔥): 

```html
<ul>
<li><strong>Optimization Tips for Windows 11</strong>: `.bambalejo` advised users to disable certain features like microsheet's core isolation and vm platform on Windows 11 for better performance, and to ensure <em>VirtualizationBasedSecurityStatus</em> is set to 0.</li>
<li><strong>TinyBox Announcement</strong>: `senecalouck` shared a link with details on the TinyBox from TinyCorp, a new hardware offering found <a href="https://tinygrad.org">here</a>.</li>
<li><strong>E-commerce GPU Frustrations and Specs</strong>: `goldensun3ds` recounted a negative experience purchasing a falsely advertised GPU on eBay, opting for Amazon for their next purchase, listing their robust PC specs including dual RTX 4060 Ti 16GB.</li>
<li><strong>Old Hardware Nostalgia</strong>: A string of messages from users like `jans_85817`, `nullt3r`, `heyitsyorkie`, and `666siegfried666`, reminisced about older GPUs; the conversation included insights like the GTX 650 being unfit for modern models, and personal stories of past rigs and upgrades.</li>
<li><strong>Discussion on Nvidia Nvlink / SLI</strong>: Users `dub_ex` and `nullt3r` discussed the effectiveness of Nvidia Nvlink / SLI, concluding it is beneficial for model training but not necessarily for inference.</li>
</ul>
```
  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1212328627962912788) (7 messages): 

- **Inquiring about Image Insertion in LM Studio**: `@heoheo5839` was unsure about how to add an image into LM Studio as the 'Assets' bar wasn't visible. `@heyitsyorkie` explained that to add an image, one must use a model like `PsiPi/liuhaotian_llava-v1.5-13b-GGUF/`, ensure both the vision adapter (mmproj) and gguf of the model are downloaded, after which the image can be inserted in the input box for the model to describe.
  
- **Questions about llava Model Downloads**: `@hypocritipus` queried about the possibility of downloading llava supported models directly within LM Studio, alluding to easier accessibility and functionality.

- **Clarifying llava Model Functionality in LM Studio**: `@wolfspyre` questioned whether downloading llava models is a current functionality, suggesting that it might already be supported within LM Studio.

- **Confirming Vision Adapter Model Use**: In response to `@wolfspyre`, `@hypocritipus` clarified they hadn't tried to use the functionality themselves and were seeking confirmation on whether it was feasible to download both the vision adapter and the primary model simultaneously within LM Studio.

- **Exploring One-Click Downloads for Vision-Enabled Models**: `@hypocritipus` shared an excerpt from the release notes indicating that users need to download a Vision Adapter and a primary model separately. They expressed curiosity about whether there is a one-click solution within LM Studio to simplify this process, where users could download both necessary files with a single action.

**Links mentioned**:

- [Vision Models (GGUF) - a lmstudio-ai Collection](https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1): no description found
- [Tweet from LM Studio (@LMStudioAI)](https://x.com/LMStudioAI/status/1734640355318944190?s=20): Counting penguins can be challenging 🧐🐧  New in LM Studio 0.2.9:   🎉 Local & Offline Vision Models!  In this demo: the small and impressive Obsidian Vision 3B by @NousResearch.

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1212417056729735179) (7 messages): 

- **Gemini vs. ChatGPT in Translation Tasks**: `@hypocritipus` shared their experience using **Gemini and ChatGPT** for translating psychological evaluation reports from Turkish to English, noting that Gemini generally provided better translations.
- **Struggle with Gemini's Overzealous Formatting**: `@hypocritipus` expressed frustration with **Gemini's** tendency to add **unnecessary bullet points** and its habit of hallucinating content beyond the requested translation.
- **ChatGPT to the Rescue, Sort of**: For the final report, `@hypocritipus` had to switch to **ChatGPT** due to Gemini not delivering as expected, though they mentioned that ChatGPT's translation was **inferior**.
- **Accidental Message in Autogen**: `@hypocritipus` humorously noted they posted their experience in the **Autogen** channel by mistake, highlighted by a "LMFAO wrong place for me to post this..." comment.
- **Confusion Cleared Up**: `@johnnyslanteyes` asked for clarification on what `@hypocritipus` meant by "translation" of the reports, which led to the explanation that it was a **language translation** from Turkish to English, not a conversion of medical jargon.
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1212531410066350161) (3 messages): 

- **Dimensionality Details Disclosed**: User `@npcomp_22591` mentioned having positive outcomes using **768 dimensions** for vectors.
- **Vectors 101**: In response to an inquiry from `@bigsuh.eth` on how to check vector dimensions, `@npcomp_22591` briefly explained the process: you can check the **dimensionality of a vector** by examining its length, providing an example output followed by `.length`.
  

---


### LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/) (1 messages): 

jans_85817: i am are waiting that lm studio version for linux
  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1212353471689261138) (1 messages): 

- **Cosmopedia Unleashed**: `@lunarflu` announced the release of **Cosmopedia**, touting it as the largest open synthetic dataset of textbooks, blogposts, and stories created by Mixtral with over **25B tokens** and **30M files**. Available resources linked through [LinkedIn post](https://www.linkedin.com/posts/loubna-ben-allal-238690152_today-were-releasing-cosmopedia-the-activity-7165785808883404800-t8o4?utm_source=share&utm_medium=member_desktop).
  
- **`huggingface_hub` Library Updates**: The new `huggingface_hub` library version **0.21.0** release was highlighted, featuring dataclasses, `PyTorchHubMixin` support, and `audio-to-audio` inference among other updates. Developers can view the full release notes at the [huggingface space](https://huggingface.co/spaces/Wauplin/huggingface_hub/discussions/4).

- **New Methods and Models on the Horizon**: The posts shared exciting developments, including training a **DoRA using diffusers script**, pushing **Figma frames** to a dataset, and the debut of **YOLOv9** on the hub with compatibility confirmed for **Transformers.js**. Additional updates covered `sentence-transformers` v2.4.0, the **LGM Mini** project, and the possibility to run **AWQ models on AMD GPUs**.

- **Innovations in Product**: Google's open LLM **Gemma 7B** is now available on Hugging Chat, `transformers` released a new task guide for mask generation, and a new `image-feature-extraction` tag was introduced, highlighting a model like `google/vit-base-patch16-224-in21k`.

- **Community Collaboration and Contributions**: Community efforts led to the release of datasets such as `#data-is-better-together`'s `10k_prompts_ranked`, and `OpenHermesPreferences`. Furthermore, TTS Arena was launched for testing and rating text-to-speech models, and **Fine-Tuning Gemma Models** guide was made available on Hugging Face's blog.

**Links mentioned**:

- [@Wauplin on Hugging Face: &quot;🚀 Just released version 0.21.0 of the `huggingface_hub` Python library!…&quot;](https://huggingface.co/posts/Wauplin/967130417344883): no description found
- [Tweet from Victor M (@victormustar)](https://x.com/victormustar/status/1760605242574459075): 🤯 This @figma plugin lets you push your figma frames directly into a @huggingface dataset!
- [Tweet from merve (@mervenoyann)](https://x.com/mervenoyann/status/1761502674778824819): YOLOv9 arrived on @huggingface Hub! 🤩  The model checkpoints: https://huggingface.co/merve/yolov9  Try the demo (@kadirnar_ai): https://huggingface.co/spaces/kadirnar/Yolov9  Find demo for YOLOv9 por...
- [Tweet from Xenova (@xenovacom)](https://x.com/xenovacom/status/1761096573755302267): YOLOv9 just released, and now it&#39;s compatible with 🤗 Transformers.js!  That&#39;s right... near real-time object detection running locally in your browser: no server required! 🤯 Try it out yours...
- [Tweet from Omar Sanseviero (@osanseviero)](https://x.com/osanseviero/status/1761024847864275448): Matryoshka Embeddings are here! 🔥  The Sentence Transformers library allows training and running embedding models with embedding sizes that can be shrunk while keeping high quality!  Learn about them...
- [Tweet from dylan (@dylan_ebert_)](https://x.com/dylan_ebert_/status/1760745208793453047): LGM Mini 🧊 Image to Interactive 3D in 5 seconds  https://huggingface.co/spaces/dylanebert/LGM-mini
- [Tweet from Julien Chaumond (@julien_c)](https://x.com/julien_c/status/1760291774348587432): BREAKING:  ↘️ Quoting Victor M (@victormustar)   ✨ Google’s new open LLM Gemma 7B is now available on HuggingChat.
- [Tweet from merve (@mervenoyann)](https://x.com/mervenoyann/status/1760972444829929492): 🤗 transformers has a new task guide for mask generation (also known as zero-shot image segmentation) learn how to use the powerful segment-anything models in this guide  https://huggingface.co/docs/t...
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=image-feature-extraction&sort=trending): no description found
- [DIBT/10k_prompts_ranked · Datasets at Hugging Face](https://huggingface.co/datasets/DIBT/10k_prompts_ranked): no description found
- [@davanstrien on Hugging Face: &quot;The open-source AI community can build impactful datasets collectively!…&quot;](https://huggingface.co/posts/davanstrien/528781527880535): no description found
- [Tweet from Lewis Tunstall (@_lewtun)](https://x.com/_lewtun/status/1762172902252892601): 🪽Introducing OpenHermesPreferences - the largest dataset of ~1 million AI preferences generated by Mixtral and Nous-Hermes-2-Yi-34B 🔥  https://huggingface.co/datasets/argilla/OpenHermesPreferences  ...
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://x.com/reach_vb/status/1761482861176082921): Announcing TTS Arena! 🗣️  *sound on*  One place to test, rate and find the champion of current open models.  A continually updated space with the greatest and the best of the current TTS landscape! ⚡...
- [Introducing the Red-Teaming Resistance Leaderboard](https://huggingface.co/blog/leaderboards-on-the-hub-haizelab): no description found
- [AI Watermarking 101: Tools and Techniques](https://huggingface.co/blog/watermarking): no description found
- [Fine-Tuning Gemma Models in Hugging Face](https://huggingface.co/blog/gemma-peft): no description found
- [Tweet from Bassem Asseh 🤗 (@asseh)](https://x.com/asseh/status/1762077722031911115): .@huggingface worked together with @FetchRewards  to take their document #AI solutions to production on @AWS .  And guess what ? 👉 &#34;With Yifeng’s guidance, Fetch was able to cut its development t...

  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1212327950180155412) (491 messages🔥🔥🔥): 

- **GPU Pricing Queries**: `@zorian_93363` discussed the cost comparison between certain GPUs and a specific 3090 model. They mentioned the possibility of acquiring 100 units for the price of a single 3090 in their location.
- **Increasing Model Performance through Custom Frameworks**: `@ahmad3794` suggested that writing custom frameworks could unleash the potential of 4 teraflops on an 8-bit integrated circuit, offering considerable computing power.
- **Electronics DIY Enthusiasm**: `@zorian_93363` expressed a desire to play with electronics and build computers but lamented the lack of time due to an economic crisis, while appreciating others' skills and abilities to innovate despite challenges.
- **Iran's Resourcefulness Amidst Sanctions**: `@ahmad3794` elaborated on building affordable clusters as a workaround for obtaining high-power technology, which is hard to get in Iran due to sanctions.
- **Accessing GPT Models and UI Challenges**: `@welltoobado` and `@caleb_sol` discussed the possibility and methods of using quantized versions of models for CPU inference without extensive RAM usage, with mentions of llama cpp as a beneficial tool.

**Links mentioned**:

- [GroqChat](https://groq.com/): no description found
- [Morph Studio](https://app.morphstudio.com): no description found
- [Unbelievable! Run 70B LLM Inference on a Single 4GB GPU with This NEW Technique](https://huggingface.co/blog/lyogavin/airllm): no description found
- [Hugging Face](https://apply.workable.com/huggingface/?lng=en): Here at Hugging Face, we’re on a journey to advance and democratize ML for everyone. Along the way, we contribute to the development of technology for the better.
- [2869993 Hail GIF - 2869993 Hail - Discover &amp; Share GIFs](https://tenor.com/view/2869993-hail-gif-12594371): Click to view the GIF
- [Tweet from blob (@moanaris)](https://fxtwitter.com/moanaris/status/1747663326832976137): no description found
- [kopyl/ui-icons-256 · Hugging Face](https://huggingface.co/kopyl/ui-icons-256): no description found
- [Hugging Face – The AI community building the future.](https://huggingface.co/): no description found
- [Kermit Worried GIF - Kermit Worried Oh No - Discover &amp; Share GIFs](https://tenor.com/view/kermit-worried-oh-no-anxious-gif-11565777): Click to view the GIF
- [Boom Explode GIF - Boom Explode Explosions - Discover &amp; Share GIFs](https://tenor.com/view/boom-explode-explosions-gif-17468299): Click to view the GIF
- [Matrix Multiplication Background User&#x27;s Guide - NVIDIA Docs](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc```): no description found
- [Hugging Face – The AI community building the future.](https://huggingface.co): no description found
- [Gradio](https://tencentarc-photomaker.hf.space/): no description found
- [Tweet from Jason (@mytechceoo)](https://fxtwitter.com/mytechceoo/status/1715400853912457532): ChatGPT wrappers when OpenAI is down..
- [cahya/gpt2-small-indonesian-522M · Hugging Face](https://huggingface.co/cahya/gpt2-small-indonesian-522M/): no description found
- [dpaste/15nGx (Python)](https://dpaste.org/15nGx): no description found
- [NCIS ridiculous hacking scene: one keyboard, two typists HD](https://www.youtube.com/watch?v=kl6rsi7BEtk): no description found
- [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF at main](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main): no description found
- [The System Is Down- Strongbad](https://www.youtube.com/watch?v=ILVfzx5Pe-A): Wow.. Really didn&#39;t think this video would be this popular. Apparently people come here when a server to a game is down. Ha! Epic.. Anyway, enjoy! Yes it&#39;s j...
- [‎Hugging Face Outage Impact](https://g.co/gemini/share/790ed3d0665e): Created with Gemini Advanced.
- [The Website is Down #1: Sales Guy vs. Web Dude](https://youtu.be/uRGljemfwUE?si=0mE6_lRmhLjmDgTb): The Website is Down: Sales Guy Vs. Web Dude High QualityThe original video in high resolution.This video won a Webby award!
- [&#39;HumanEval&#39; object has no attribute &#39;dataset&#39; · Issue #131 · bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness/issues/131): When I evaluate human eval with llama 7b, I met this problem: my script accelerate launch /cpfs01/shared/Group-m6/dongguanting.dgt/bigcode-evaluation-harness/main.py --model &quot;/path to my llama7b/...
- [Issues · huggingface/api-inference-community](https://github.com/huggingface/api-inference-community/issues): Contribute to huggingface/api-inference-community development by creating an account on GitHub.
- [Workflow runs · huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference/actions): A blazing fast inference solution for text embeddings models - Workflow runs · huggingface/text-embeddings-inference
- [GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.](https://github.com/comfyanonymous/ComfyUI): The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI
- [Issue with offline mode · Issue #4760 · huggingface/datasets](https://github.com/huggingface/datasets/issues/4760): Describe the bug I can&#39;t retrieve a cached dataset with offline mode enabled Steps to reproduce the bug To reproduce my issue, first, you&#39;ll need to run a script that will cache the dataset im...
- [Issues · huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub/issues): The official Python client for the Huggingface Hub. - Issues · huggingface/huggingface_hub
- [Build software better, together](https://github.com/huggingface/text-embeddings-inference/pkgs/container/text-embeddings-inference/versions?filters%5Bversion_type%5D=tagged&page=1): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [Add PatchModelAddDownscale (Kohya Deep Shrink) node. · comfyanonymous/ComfyUI@bd07ad1](https://github.com/comfyanonymous/ComfyUI/commit/bd07ad1861949007139de7dd5c6bcdb77426919c): By adding a downscale to the unet in the first timesteps this node lets you generate images at higher resolutions with less consistency issues.
- [
Hugging Face status
](https://status.huggingface.co/): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1212448212754112593) (8 messages🔥): 

- **Exploring DSPy and OpenFunctions v2**: User `@n278jm` is investigating **DSPy**, a framework for programming foundation models without prompting, and **Gorilla OpenFunctions v2**, an advanced open-source function calling system for LLMs. They aim to use these tools to improve their client on-boarding process, making the move from Gradio prototypes to production-ready versions.
- **Harness the Power of OpenAI and Hugging Face**: `@davidre95` encourages users to utilize the tools from [OpenAI Chat](https://chat.openai.com/share/e64eee91-62ac-4265-80dc-1facc4d0762e) and [Hugging Face chat room](https://hf.co/chat/r/-ym0Q-L) as resources.
- **Project Collaboration on Invoice Processing**: `@pampkinparty000` invites users dealing with PDF or picture invoices to DM them for a potential collaboration on a project with similar goals.
- **Invoice Storage Advice for Greater Efficiency**: `@pampkinparty000` recommends storing invoices in a vectorized database with metadata for more efficient use of LLMs, suggesting the use of libraries like *llama-index*.
- **Seeking a Research Community in AI**: `@raghadn3` is in search of a community dedicated to writing research papers on Artificial Intelligence.

**Links mentioned**:

- [GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy): DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy
- [Introduction to Gorilla LLM](https://gorilla.cs.berkeley.edu/blogs/7_open_functions_v2.html): no description found

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1212447353693274192) (9 messages🔥): 

- **BitNet b1.58: Efficient LLMs**: `@jessjess84` highlighted the potential of **BitNet b1.58**, a new **1-bit** Large Language Model that promises efficiency without sacrificing performance, detailed in an [arXiv paper](https://arxiv.org/abs/2402.17764). Achieving the same results as full-precision models, it introduces cost-effective latency, memory, throughput, and energy consumption.
  
- **Stable Diffusion Deluxe Debuts**: `@skquark` invited users to try **Stable Diffusion Deluxe**, an extensive multimedia AI toolkit supporting various AI art generators, boasting features for creating images, videos, sound effects, and more. The platform, detailed at [diffusiondeluxe.com](https://diffusiondeluxe.com), integrates numerous pipelines and is designed for ease of use and creative experimentation.
  
- **Looking for Self-Hosting Details**: In response to `@skquark`'s all-in-one multimedia AI app, `@wolfspyre` inquired about self-hosting options, complimenting the project as "super cool" and expressing interest in diving deeper.
  
- **Appreciating 'The Hug'**: `@evergreenking` shared a link to [thehug.xyz](https://thehug.xyz), a site described as "just link art," with `@wolfspyre` following up to ask if it was `@evergreenking`'s creation.

**Links mentioned**:

- [HUG | A Home for Your Art](https://thehug.xyz): Join our global creative community to showcase & sell your art, connect with others, and access creator-friendly grants and education.
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [Uncovering the Origins of Values: A Biology and Cognition-Based Approach for AI Alignment](https://docs.google.com/document/d/1A2ZdM1IBv0_5nN1pujyCvmoCGepETmWFRPmAmdjkqqA/edit?usp=drivesdk): no description found
- [Diffusion Deluxe Home - Stable Diffusion Deluxe](https://diffusiondeluxe.com): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1212410799155453962) (14 messages🔥): 

- **DIY Local LLM Assistant Unveiled**: `@rivridis` developed a **Locally running LLM Assistant** with an assistant mode and real-time editing mode for content editing and creation. The code and details are available on [GitHub](https://github.com/Rivridis/LLM-Assistant).

- **Deploy to Google Cloud Vertex AI Simplified**: `@alvarobartt` wrote a blog post detailing how to deploy models from the HuggingFace Hub to Google Cloud Vertex AI. You can check out the technical post and its step-by-step guide [here](https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai). 

- **Cursor Hero demo v0.3.0**: `@teamy` is developing a UI tool titled **Cursor Hero**, with integrations of ollama and whisper. A demo of the tool can be found in this [YouTube video](https://youtu.be/t1PYks0UTL8).

- **Gantrithor: A Data Annotation Leap**: `@stroggoz` announced an open beta for **Gantrithor**, a rapid, bulk data annotation tool, with a free version limiting datasets to 1000 documents. Learn more and try it out at [Gantrithor](https://www.gantrithor.com/).

- **Starcoder 2: Code & Learn**: `@tonic_1` fixed errors in the example code and announced **Starcoder 2**, available for learning and enjoyment, with a call to collaborate on fine-tuning models. Find the project on [HuggingFace Spaces](https://huggingface.co/spaces/Tonic/starcoder2).

**Links mentioned**:

- [MetaMath Mistral Pro - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/MetaMath-Mistral-Pro): no description found
- [Deploying 🤗 Hub models in Vertex AI](https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai): no description found
- [StarCoder2 - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/starcoder2): no description found
- [Qbeast&#039;s Adventure in AI-Driven Meme Creation - Qbeast](https://qbeast.io/qbeasts-adventure-in-ai-driven-meme-creation/): Learn about AI model selection, fine-tuning, and the role of Qbeast in enhancing meme creativity. Perfect for AI enthusiasts and data engineers seeking insights and innovation.
- [Gantrithor](https://www.gantrithor.com/): no description found
- [Cursor Hero demo v0.3.0](https://youtu.be/t1PYks0UTL8): https://github.com/TeamDman/Cursor-Hero.githttps://discord.gg/psHtde64FJ#rust #bevy #windows #win32
- [this one rly slaps - episode 16 #music #producer](https://youtube.com/shorts/U1fhv5Zc5xk?feature=share): gonna be hard to beat this one
- [GitHub - Rivridis/LLM-Assistant: Locally running LLM with internet access](https://github.com/Rivridis/LLM-Assistant): Locally running LLM with internet access. Contribute to Rivridis/LLM-Assistant development by creating an account on GitHub.
- [SDXL-Lightning: quick look and comparison](https://www.felixsanz.dev/articles/sdxl-lightning-quick-look-and-comparison): With SDXL-Lightning you can generate extremely high quality images using a single step.

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1212732495972274246) (5 messages): 

- **Gradio Queue Function Clarification**: User `@akin8941` inquired about the return type of the `queue()` function in gradio interface, and `@iakhil` clarified that **it does not have a return type** of its own.
- **Too Fast for Comfort**: `@HuggingMod` cautioned `@1122120801903194114` about posting too quickly in the **HuggingFace** Discord, asking to slow down a bit with a friendly reminder emoji.
- **Scheduler Name Puzzle**: `@luihis` expressed difficulty in retrieving the string name of a scheduler due to deprecation warnings. Despite attempts using different properties, the correct string, "DPMSolverSinglestepScheduler," remained elusive.
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1212732013363204126) (4 messages): 

- **Parseq Praise**: User `@whoami02` recommended the use of **Parseq** for its effective symbol recognition capabilities.
- **Personalized Fine-tuning Success**: They also mentioned successfully fine-tuning the model on their specific dataset, which contained images similar to the equations they needed to detect.
- **Resnet Still Rocks**: As for the task of detection, `@whoami02` asserted that **Resnet** stands strong and is good enough for their needs.
- **Slow Your Roll**: `@HuggingMod` advised `@whoami02` to slow down their message posting to adhere to the community guidelines.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1212370281692135435) (14 messages🔥): 

- **Inference Troubles in the Hugging Face Repo**: `@alfred6549` sought assistance for running the [text generation inference repository](https://github.com/huggingface/text-generation-inference) on a machine without a CPU or CUDA, sharing an error they encountered. Despite attempts to disable GPU usage, the local setup still failed.
  
- **Petals Resonate with Users**: User `@ai_noob` simply stated "petals", which received a positive acknowledgment from `@nrs9044`, indicating a shared sentiment or understanding about the term’s context.
  
- **Benchmark Necessities Discussed**: `@vipitis` stressed the importance of testing on larger benchmarks for validity, while `@djpanda1` acknowledged the advice but noted that preliminary tests on several prompts appeared successful.
  
- **Financial Document Insight Quest**: `@hiteshwarsingh1` is exploring ways to extract information from financial documents, considering MapReduce techniques and seeking recommendations for open-source models or approaches suitable for summarization rather than specific information retrieval.
  
- **Improving Information Extraction with LLMs**: `@.sgp` is utilizing mistral 7b with llamacpp for JSON data extraction and expressed interest in incorporating in-context learning to enhance accuracy, requesting resources on the topic.

**Links mentioned**:

- [deepseek-ai/deepseek-coder-6.7b-instruct · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct): no description found
- [Hugging Face](https://github.com/huggingface/): The AI community building the future. Hugging Face has 196 repositories available. Follow their code on GitHub.

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1212732495972274246) (5 messages): 

- **Gradio's `queue()` Function Clarification**: `@akin8941` asked about the return type of the `queue()` function in the Gradio interface, to which `@iakhil` responded that it **doesn't have a return type of its own**.
- **Slow Down Warning by HuggingMod**: A reminder was given by `HuggingMod` directed at `<@1122120801903194114>`, cautioning them to **slow down their message frequency** in the channel.
- **Trouble with Deprecation Notice**: `@luihis` shared a snippet of code and expressed confusion due to a deprecation warning when trying to get the **name of a scheduler as a string**; emphasizes uncertainty even after different attempts at printing the scheduler's class name.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1212390278145769492) (314 messages🔥🔥): 

- **Ideogram Launch Causes Stir**: `@pseudoterminalx` shared a prompt result from the new AI model by Ideogram, triggering discussions on its prompt adherence and aesthetics. There were comparisons to Stable Diffusion and speculations about the potential poor quality of unseen Imagen samples.

- **T5 XXL, CLIP L, and CLIP G in SD3?**: `@thejonasbrothers` and `@devilismyfriend` discussed the integration of T5 XXL and CLIP models in SD3, hinting at the potential for both accuracy and appealing aesthetics in future models.

- **Cascade's Fidelity Questioned**: `@pseudoterminalx` and others critically evaluated Cascade's ability to generate images based on prompts, noting frequent issues with prompt adherence and specificity.

- **AI Generated Art and Copyright Battles**: Users `@progamergov`, `@itali4no`, and others engaged in conversations about the looming legal challenges around AI-generated art, referencing recent cases and the ambivalent approach of Huggingface towards DMCA requests.

- **Stability AI's Silent Many Projects**: `@.undeleted` expressed confusion over the multiplicity of projects with similar goals at Stability AI, each announced similarly but with unclear differences.

**Links mentioned**:

- [Release v0.9.1 - DoRA the explorah · bghira/SimpleTuner](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.1): This release has some breaking changes for users who:  Use RESOLUTION_TYPE=area (resolution_type=area for multidatabackend config) Use crop=false Use crop=true and crop_aspect=preserve  as the prec...
- [panopstor/nvflickritw-cogvlm-captions · Datasets at Hugging Face](https://huggingface.co/datasets/panopstor/nvflickritw-cogvlm-captions): no description found
- [Willys Chocolate Experience Glasgow. Get your Tickets!](https://willyschocolateexperience.com/): INDULGE IN A CHOCOLATE FANTASY LIKE NEVER BEFORE - CAPTURE THE ENCHANTMENT! Tickets to Willys Chocolate Experience are on sale now!  at the willys chocolate experience in Glasgow! Tickets to Willys Ch...
- [China issues world's 1st legally binding verdict on copyright infringement of AI-generated images - Global Times](https://www.globaltimes.cn/page/202402/1307805.shtml): no description found
- [Copyright Safety for Generative AI | Published in Houston Law Review](https://houstonlawreview.org/article/92126-copyright-safety-for-generative-ai): By Matthew Sag. 61 Hous. L. Rev. 295 (2023)

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1212329067265785856) (48 messages🔥): 

- **Spiking Neural Network Speculations**: `@max_voltage` wonders if advancements might lead to a reintroduction of spiking neural networks, proposing time dithering as a technique to enhance precision. `@spirit_from_germany` agrees, reminded of spiking networks by the concept.

- **Contemplating Low Information Density in Models**: `@max_voltage` expresses surprise at the ability to lower information to 1-2 bits per weight in models, indicating a low info density in current models. `@thejonasbrothers` explained this is possible due to the innate sparsity of existing networks, while some weights could be even 1-bit or 0-bit.

- **New AI Image Generator Buzz**: `@vrus0188` shares a Reddit post about a new AI image generator that's reportedly 8 times faster than OpenAI's best tool and can run on modest computers. `@spirit_from_germany` provides a link to the [KOALA image generator site](https://youngwanlee.github.io/KOALA/) for quality testing without cherry-picking.

- **EMO: Creating Expressive Portrait Videos**: The [EMO project](https://humanaigc.github.io/emote-portrait-alive/) is highlighted by `@helium__`, presenting a new audio-driven portrait-video generation method. `@itali4no` remarks on the same authors as the animate anyone paper, indicating a likely absence of released code.

- **AI Icon Generation Model Release**: `@kopyl` announces the release of a state-of-the-art AI model for icon generation, trained with a personal investment of $2000, available via [Hugging Face](https://huggingface.co/kopyl/ui-icons-256). `@chad_in_the_house` praises the model's low noise, although `@kopyl` advises that it only generates images at 256px resolution.

- **Language Model Distillation Learning Inquiry**: `@jh0482` seeks information on distillation learning specifically for embedding language models, discussing concerns related to continuous space targets. `@itali4no` suggests standard distillation methods might apply, but `@jh0482` considers regression towards the target and contrastive learning as potential methods.

**Links mentioned**:

- [KOALA: Self-Attention Matters in Knowledge Distillation of Latent Diffusion Models for Memory-Efficient and Fast Image Synthesis](https://youngwanlee.github.io/KOALA/): SOCIAL MEDIA DESCRIPTION TAG TAG
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364): We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates t...
- [EMO](https://humanaigc.github.io/emote-portrait-alive/): EMO: Emote Portrait Alive - Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions
- [Samsung Develops Industry-First 36GB HBM3E 12H DRAM](https://news.samsung.com/global/samsung-develops-industry-first-36gb-hbm3e-12h-dram): Samsung’s HBM3E 12H achieves industry’s largest capacity HBM with groundbreaking 12-layer stack, raising both performance and capacity by more than 50%  Advanced TC NCF technology enhances vertical de...
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/1b24t06/new_ai_image_generator_is_8_times_faster_than/): no description found
- [GitHub - collabora/WhisperSpeech: An Open Source text-to-speech system built by inverting Whisper.](https://github.com/collabora/WhisperSpeech/?tab=readme-ov-file): An Open Source text-to-speech system built by inverting Whisper. - collabora/WhisperSpeech
- [kopyl/ui-icons-256 · Hugging Face](https://huggingface.co/kopyl/ui-icons-256): no description found
- [UI icons - v1.0 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/327499): SOTA model for generating icons. Motivation: I spent $2000 of my own money to train this model. I was unable to monetize it, so I&#x27;m sharing it with...

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1212329681173610537) (21 messages🔥): 

- **Emoji Reacts Tell a Story**: `@leontello` and `@0xevil` employed emotive emojis, with the former using a salute emoji (`<:o7:1151260455218708480>`) and the latter a skull emoji (`<:dead:1072635189274083409>`), reflecting a sense of conclusion or death, followed by a crying face (`<:f_cry:1159653986681499768>`) in response to the absence of GPT-5.
- **Anticipating Future GPT iterations**: Conversation by `@0xevil` highlighted the community’s anticipation for future GPT versions, mentioning non-existent GPT-6 and responding humorously to `@error.pdf`’s mention of GPT-9 with a surprised emoji (`<:ooo:1133962720232865843>`).
- **Monitor and Dock Recommendations**: `@denovich` shared a [YouTube video](https://youtu.be/0TY7J58UEro?si=5UayYH3t3gCC0M_H) reviewing Dell's new 5K monitor and suggested that Dell offers monitors that can connect to multiple machines simultaneously, while mentioning that their docking stations and a specific model, the Dell Thunderbolt Dock WD22TB4, are worth considering and can be found on eBay.
- **Anticipations on Y Combinator's Batch Focus**: `@0xevil` pondered whether Y Combinator’s latest batch predominantly featured companies offering GPT-wrapper services, observing similarities with existing products and innovations in areas like transcription and code generation from design.
- **Speculations and Shared Resources Surrounding GPT Patents and Applications**: `@0xevil` mulled over the GPT-6 patent possibly discussed in broader circles and noted the integration of AI agents with music generation, while `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be) demonstrating how to fine-tune the Gemma model using Unsloth.

**Links mentioned**:

- [Oppenheimer Oppenheimer Movie GIF - Oppenheimer Oppenheimer movie Oppenheimer explosions - Discover &amp; Share GIFs](https://tenor.com/view/oppenheimer-oppenheimer-movie-oppenheimer-explosions-oppenheimer-blue-oppenheimer-orange-gif-4256076026225346812): Click to view the GIF
- [Finetune Gemma 7B with Unsloth](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be): We will take a look at how to finetune Gemma model using unslothhttps://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollT...
- [One Month with the Best Monitor in the World: The New Dell 40&quot; 5K120 HDR U4025QW](https://youtu.be/0TY7J58UEro?si=5UayYH3t3gCC0M_H): Dave spends a month with the brand new Dell 5K120 HDR monitor.  For my book on life on the Spectrum: https://amzn.to/49sCbbJFollow me on Facebook at http://f...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1212746283467546684) (6 messages): 

- **1-bit Revolution in LLMs**: `@deki04` shared an [arXiv paper](https://arxiv.org/abs/2402.17764) introducing **BitNet b1.58**, a new 1-bit Large Language Model that achieves comparable performance to full-precision models while being more cost-effective. The model presents a "new scaling law" for designing high-performance, yet cost-efficient LLMs.
  
- **Curiosity Piqued by BitNet**: `@deki04` expressed surprise about the existence of 1-bit LLMs, not having encountered this concept before. 

- **Scaling Laws Under the Microscope**: `@sherlockzoozoo` commented that multiplicative scaling laws are interesting, presumably in the context of the 1-bit LLM, and noted that additive scaling doesn't perform well with increasing model size.

- **New LLM Benchmark Released**: `@tarruda` shared a link to [Nicholas Carlini's benchmark](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html) for Large Language Models, highlighting its unique tests that include a range of complex tasks and the use of a dataflow domain specific language for easy test additions.

- **Benchmark Results on Mistral vs GPT-4**: Following the benchmark share, `@tarruda` mentioned a [YouTube video](https://www.youtube.com/watch?v=IH2htfsciO4) where someone tested the benchmark on various models, including some 7B models like Mistral and GPT-4.

**Links mentioned**:

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [
      My benchmark for large language models
    ](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html): no description found
- [Mistral Large vs GPT4 - Practical Benchmarking!](https://www.youtube.com/watch?v=IH2htfsciO4): ➡️ One-click Fine-tuning &amp; Inference Templates: https://github.com/TrelisResearch/one-click-llms/➡️ Trelis Function-calling Models (incl. OpenChat 3.5): http...

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1212327517206093845) (205 messages🔥🔥): 

- **Ragtag Ruminations on RAG**: `@natefyi_30842` discussed the use of an LLM to create Q&A pairs that are then fine-tuned and combined with RAG for better context understanding.
- **Issues with Service Providers and Fine-Tuning**: `@teknium` commented that fine-tuning providers are facing issues due to conflicts between fine-tune mixing and scaled inference code, making local GGUF setups the only reliable option currently.
- **Troubles with Gemini 2B Fine-Tuning**: `@lmmint` asked the community if anyone was successful in fine-tuning the Gemini 2B and mentioned high-quality data as a requirement.
- **CausalLM's Impressive MMLU Score**: `@nonameusr` expressed surprise at CausalLM's high MMLU benchmark and shared a link provided by `@giftedgummybee` to the Hugging Face model [CausalLM/34B-preview](https://huggingface.co/CausalLM/34B-preview).
- **Excitement Around the Release of HyenaDNA**: Discussions surrounding Stanford's introduction of **HyenaDNA**—long-range genomic model with 1 million token capacity—generated buzz, with `@euclaise` suggesting "fill in the middle" (FIM) might be suitable for DNA sequences over autoregressive models.

**Links mentioned**:

- [Tweet from undefined](https://x.com/RealJosephus?t=p5kYoitoAq5wfe_NBl0-Ig&s=09): no description found
- [HyenaDNA: learning from DNA with 1 Million token context](https://hazyresearch.stanford.edu/blog/2023-06-29-hyena-dna): HyenaDNA is a long genomic sequence model trained on the Human Reference Genome with context length of up to 1 million tokens.
- [CausalLM/34B-preview · Hugging Face](https://huggingface.co/CausalLM/34B-preview): no description found
- [qualcomm (Qualcomm)](https://huggingface.co/qualcomm): no description found
- [Embedding - GPT4All Documentation](https://docs.gpt4all.io/gpt4all_python_embedding.html): no description found
- [OpenAI Five defeats Dota 2 world champions](https://openai.com/research/openai-five-defeats-dota-2-world-champions): OpenAI Five is the first AI to beat the world champions in an esports game, having won two back-to-back games versus the world champion Dota 2 team, OG, at Finals this weekend. Both OpenAI Five and De...
- [Tweet from TechCrunch (@TechCrunch)](https://x.com/techcrunch/status/1762942326391906352?s=46): Tim Cook says Apple will ‘break new ground’ in GenAI this year https://tcrn.ch/3Ig8TAX
- [UniProt](https://www.uniprot.org/uniprotkb/Q99ZW2/entry#sequences): no description found
- [sordonia (Alessandro Sordoni)](https://huggingface.co/sordonia): no description found
- [supertrainer2000/supertrainer2k/optim/adalite.py at master · euclaise/supertrainer2000](https://github.com/euclaise/supertrainer2000/blob/master/supertrainer2k/optim/adalite.py): Contribute to euclaise/supertrainer2000 development by creating an account on GitHub.
- [GitHub - nestordemeure/question_extractor: Generate question/answer training pairs out of raw text.](https://github.com/nestordemeure/question_extractor): Generate question/answer training pairs out of raw text. - nestordemeure/question_extractor
- [BAAI/bge-base-en-v1.5 · Hugging Face](https://huggingface.co/BAAI/bge-base-en-v1.5): no description found
- [Models: Remove system prompt of Nous-Hermes-2-Mistral-7b-DPO by ThiloteE · Pull Request #2054 · nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all/pull/2054/files)): Describe your changes  Adds &quot;accepts various system prompts&quot; Removes system prompt fix whitespace  Checklist before requesting a review   I have performed a self-review of my code.  If it is...
- [CausalLM/34b-beta · Hugging Face](https://huggingface.co/CausalLM/34b-beta): no description found
- [Models: Remove system prompt of Nous-Hermes-2-Mistral-7b-DPO by ThiloteE · Pull Request #2054 · nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all/pull/2054/fil): Describe your changes  Adds &quot;accepts various system prompts&quot; Removes system prompt fix whitespace  Checklist before requesting a review   I have performed a self-review of my code.  If it is...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1212326803356651580) (45 messages🔥): 

- **Seeking GPT-4 level on a budget**: `@natefyi_30842` sought a cheaper alternative to GPT-4 that can prevent the inclusion of provided subsequent book chunks in its responses, finding `Mixtral Instruct` to work fairly well despite its limitations. The conversation suggests that only GPT-4 behaves as desired in this context.

- **Fine-tuning a question of quantity**: Discussing the significance of the training dataset size, `@natefyi_30842` wondered if a hundred entries would suffice as opposed to millions, and `@teknium` succinctly replied with "5k".

- **DPO tactics in model training discussed**: In pursuit of improving model answers, `@natefyi_30842` considered generating wrong examples for Directed Prompt Optimization (DPO), meanwhile, users discussed when DPO might be more effective.

- **Choosing separators for text manipulation**: `@natefyi_30842` pondered the efficacy of using standard or unique tokens as separators, such as emojis vs. `%XYZ%`, for adding elements to text in model inputs; `@natefyi_30842` shared a [link to a tokenizer](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/tokenizer.json) for context.

- **Interpretability and engineering representations**: Max_paperclips discussed the exciting field of representations engineering, citing a favorite post and referring to work such as [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) and the corresponding [Github code for the paper](https://github.com/andyzoujm/rep).

**Links mentioned**:

- [Bowing Thank You GIF - Bowing Thank You Tom And Jerry - Discover &amp; Share GIFs](https://tenor.com/view/bowing-thank-you-tom-and-jerry-take-a-bow-chasing-gif-20784169): Click to view the GIF
- [
    
      
        Representation Engineering Mistral-7B an Acid Trip
      
    
  ](https://vgel.me/posts/representation-engineering/): no description found
- [Metas Llama 3 is set to release in July and could be twice the size](https://the-decoder.com/metas-llama-3-is-set-to-release-in-july-and-could-be-twice-the-size/): Meta&#039;s next open-source language model, Llama 3, is scheduled for release in July and is intended to be on par with GPT-4.

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1212456766210838558) (3 messages): 

Here's the summary based on the messages provided:

- **QT Node-X Twitter Updates**: QT Node-X's Twitter shared a series of posts [QT Node-X Tweet 1](https://twitter.com/qtnx_/status/1762894467399332276), [QT Node-X Tweet 2](https://twitter.com/qtnx_/status/1762895167768375407), and [QT Node-X Tweet 3](https://twitter.com/qtnx_/status/1762895953944514791), though the content of the tweets was not provided in the messages.
  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1212343969552273418) (57 messages🔥🔥): 

- **Noam Shazeer's Blog Debut**: `@swyxio` shared the first blog post by Noam Shazeer, discussing coding style, titled [Shape Suffixes: Good Coding Style](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).
- **Customer Satisfaction and LLMs**: `@eugeneyan` expressed appreciation for a data point indicating that LLMs are on par with humans in customer service satisfaction and can handle two-thirds of customer service queries.
- **Skepticism on AI News**: `@swyxio` flagged an overhyped news piece, suggesting skepticism when something seems too good, referencing the Klarna AI assistant story on [Fast Company](https://www.fastcompany.com/91039401/klarna-ai-virtual-assistant-does-the-work-of-700-humans-after-layoffs).
- **Discussion on LLM Paper Club**: `@swyxio` alerted users to a special Matryoshka Embeddings presentation, while `@osanseviero` and `@swyxio` referenced additional materials on this topic, including a blog post on [HuggingFace](https://huggingface.co/blog/matryoshka) and a [YouTube channel](https://www.youtube.com/@EfficientNLP) with simplified LLM technique explanations.
- **Insights on Lakehouses and Data Engineering**: In response to `@quicknick123` seeking resources on lakehouses, `@swyxio` recommended an in-depth guide on table formats, query engines, and the utility of Spark published by [Airbyte](https://airbyte.com/blog/data-lake-lakehouse-guide-powered-by-table-formats-delta-lake-iceberg-hudi).

**Links mentioned**:

- [no title found](https://www.]): no description found
- [Tweet from Noam Shazeer (@NoamShazeer)](https://x.com/noamshazeer/status/1762733550892401030?s=46&t=90xQ8sGy63D2OtiaoGJuww): https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd Check out my first blog post.
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147): Learned representations are a central component in modern ML systems, serving a multitude of downstream tasks. When training such representations, it is often the case that computational and statistic...
- [Tweet from murat 🍥 (@mayfer)](https://x.com/mayfer/status/1762764909371183292?s=46&t=90xQ8sGy63D2OtiaoGJuww): wow, highly recommend checking out all the samples: https://humanaigc.github.io/emote-portrait-alive/  ↘️ Quoting AK (@_akhaliq)   Alibaba presents EMO: Emote Portrait Alive  Generating Expressive Por...
- [Conviction ](https://www.conviction.com/startups): no description found
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [Efficient NLP](https://www.youtube.com/@EfficientNLP): Efficient NLP Consulting  My name is Bai Li, I&#39;m a machine learning engineer and PhD in natural language processing. I can help you build cost-effective and efficient NLP systems. Reach me at:  Em...
- [Data Lake / Lakehouse Guide: Powered by Data Lake Table Formats (Delta Lake, Iceberg, Hudi) | Airbyte](https://airbyte.com/blog/data-lake-lakehouse-guide-powered-by-table-formats-delta-lake-iceberg-hudi): Explains the open-source data lakes and their power with data lake table formats. What’s the difference between a lakehouse and when you need one.
- [Tweet from Hamel Husain (@HamelHusain)](https://x.com/HamelHusain/status/1762873030496428164?s=20): Something smells really wrong about the Klarna news it’s a bit too much made for TV?  https://www.fastcompany.com/91039401/klarna-ai-virtual-assistant-does-the-work-of-700-humans-after-layoffs
- [Tweet from Rowan Cheung (@rowancheung)](https://x.com/rowancheung/status/1763087469585498383?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): It&#39;s been a huge day for AI with announcements from Alibaba, Lightricks, Ideogram, Apple, Adobe, OpenAI, and more.  The 7 most important developments that happened:  1. Alibaba researchers unveile...
- [🪆 Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): no description found
- [Jonathan Ross at Web Summit Qatar](https://youtu.be/IixoaS5ckBA?si=iTQFG-k_SQd6OP8H): Groq CEO &amp; Founder, Jonathan Ross, on Center Stage at #WebSummitQatar2024, discussing how to make AI Real.X (fka Twitter): @WebSummitQatarInstagram: @WebSumm...

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1212465665076764712) (3 messages): 

- **Replicate CEO in the Podcast Spotlight**: `@swyxio` announced the release of a new podcast episode featuring the CEO of Replicate. The tweet with the link to the episode can be found [here](https://twitter.com/swyx/status/1762906839505846418).
- **MRL Embeddings Paper Club Meeting**: `@swyxio` gave a heads-up about an upcoming event led by `<@206404469263433728>` in the [`#1107320650961518663`](https://lu.ma/rgxvuktv) channel, where the authors of the MRL embeddings paper will be present. The event cover can be viewed [here](https://images.lumacdn.com/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-covers/mq/b7a9e5d5-cbd9-4546-a668-972d498d2186).
- **Deep Dive into Representation Engineering**: `@ivanleomk` flagged an upcoming session with `<@796917146000424970>` on **Representation Engineering 101** in the [`#1107320650961518663`](https://discord.com) channel, inviting members to participate and engage with questions.

**Links mentioned**:

[LLM Paper Club (West Edition!) · Luma](https://lu.ma/rgxvuktv): This week we&#x27;ll be covering the paper - Matryoshka Representation Learning ( https://arxiv.org/abs/2205.13147 ) with two of the co-authors Gantavya Bhatt and Aniket Rege. We have moved...

  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1212488066716012646) (165 messages🔥🔥): 

- **Matryoshka Dolls Embrace AI**: User `@akusupati` shared the paper titled ["Matryoshka Representation Learning"](https://arxiv.org/abs/2310.07707) and discussed its potential for creating **LLM embeddings with adaptive dimensions**. It's a technique that could offer varying levels of abstraction, potentially saving on compute and storage.
  
- **Making sense of MRL**: `@swyxio` and others engaged in a discussion trying to grasp the **quirks of Matryoshka Representation Learning (MRL)**, including insightful comparisons to PCA on embeddings and how this technique involves adding the loss of models at varying dimensions for optimized learning.

- **Deployment Insights and Applications**: Participants like `@ivanleomk` and `@gulo0001` offered practical information and **demonstrations of embedding models** incorporating MRL. They discussed adaptations and provided resources like a [Supabase blog](https://supabase.com/blog/matryoshka-embeddings) and [HuggingFace blog](https://huggingface.co/blog/matryoshka) that help understand the real-world use of these models.

- **Curiosity Reigns in Matryoshka Exploration**: `@punnicat`, presumably one of the authors, was present to field questions and clarify concepts around **Matryoshka Embeddings**, especially concerning dimensionality and the granularity of embeddings during training and their implications for models.

- **Engagement with Authors and Resources**: The session marked a presence of curious minds asking questions about **Matryoshka Embeddings and the broader implications for transformer models** with users like `@swyxio` and `@cakecrusher` discussing potential applications and improvements. The authors were open to sharing slides and further details like `@punnicat` who can be contacted on Twitter.

**Links mentioned**:

- [Matryoshka Representation Learning (MRL) from the Ground Up | Aniket  Rege](https://aniketrege.github.io/blog/2024/mrl/): no description found
- [Nextra: the next docs builder](https://llm-paper-club-asia-notes.vercel.app/): Nextra: the next docs builder
- [MatFormer: Nested Transformer for Elastic Inference](https://arxiv.org/abs/2310.07707): Transformer models are deployed in a wide range of settings, from multi-accelerator clusters to standalone mobile phones. The diverse inference constraints in these scenarios necessitate practitioners...
- [
    
      
        Representation Engineering Mistral-7B an Acid Trip
      
    
  ](https://vgel.me/posts/representation-engineering/#How_do_we_make_one?_Is_it_hard?'): no description found
- [Matryoshka embeddings: faster OpenAI vector search using Adaptive Retrieval](https://supabase.com/blog/matryoshka-embeddings): Use Adaptive Retrieval to improve query performance with OpenAI&#x27;s new embedding models
- [Matrioska Loop GIF - Matrioska Loop Bored - Discover &amp; Share GIFs](https://tenor.com/view/matrioska-loop-bored-tired-russian-gif-7433508): Click to view the GIF
- [AdANNS: A Framework for Adaptive Semantic Search](https://arxiv.org/abs/2305.19435): Web-scale search systems learn an encoder to embed a given query which is then hooked into an approximate nearest neighbor search (ANNS) pipeline to retrieve similar data points. To accurately capture...
- [🪆 Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): no description found
- [NeuML/pubmedbert-base-embeddings-matryoshka · Hugging Face](https://huggingface.co/NeuML/pubmedbert-base-embeddings-matryoshka): no description found
- [Representation Engineering 101](https://tana.pub/OG9hf2MA4tNS/representation-engineering-101): no description found

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1212330758447046686) (157 messages🔥🔥): 

- **Activation Woes for Rabbit R1 Promo**: User `@mithrilman` required assistance activating the Rabbit R1 promo. `@icelavaman` provided step-by-step instructions, emphasizing the need to use the email link, and suggested contacting support for further help, especially since the email button appeared bugged and non-clickable.

- **Podcast Curiosities and Clarity**: `@_paradroid` raised a question about podcasts posting under the name "Perplexity AI," prompting `@icelavaman` to clarify the official podcast link while `@ok.alex` stated that unauthorized use of the Perplexity AI name is likely for attention or money.

- **Understanding AI Model Preferences**: New user `@outrerim` asked about strengths and weaknesses of different AI models, and `@jaicraft` outlined core use-cases for Experimental, GPT-4 Turbo, Claude, and Mistral models, though opinions differed with users like `.claidler` and `naivecoder786` favoring Mistral for code queries.

- **Discussing Perplexity's Capabilities and Limitations**: `@brknclock1215` described Perplexity's AI as excellent for internet-based information handling and answering questions rapidly, but highlighted its limitations such as parsing large files and image generation, understanding it's less optimized for such tasks.

- **Concerns and Solutions for Perplexity Service Issues**: Users `@stevvie` and `@dv8s` encountered confusions regarding the absence of file upload options and name changes from "Copilot" to "Pro," while `@moyaoasis` suggested the addition of a feature for exporting Perplexity thread responses, a function not yet available but considered for future implementation.

**Links mentioned**:

- [Tweet from Perplexity (@perplexity_ai)](https://x.com/perplexity_ai/status/1762606713239130453?s=46): More on Mistral Large 👇https://www.perplexity.ai/search/Mistral-Large-Overview-Fw.QrWxvR9e9NRuDxB1wzQ
- [‎Discover Daily by Perplexity on Apple Podcasts](https://podcasts.apple.com/us/podcast/discover-daily-by-perplexity/id1732181427): ‎News · 2024
- [‎Perplexity AI on Apple Podcasts](https://podcasts.apple.com/us/podcast/perplexity-ai/id1725553091): ‎News · 2024
- [‎Stuff You Should Know About AI on Apple Podcasts](https://podcasts.apple.com/us/podcast/stuff-you-should-know-about-ai/id1722322183): ‎Business · 2024

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1212352475894521886) (13 messages🔥): 

- **Librem5 Explores BurpSuite Community Edition**: `@librem5` shared a [Perplexity link](https://www.perplexity.ai/search/burpsuite-community-versus-XOmmYWeFS2.dpO0FPQFyRg) examining the differences between BurpSuite Community Edition and an unspecified alternative.
- **Muscle Building Plan crafted by AI**: `@commuting5048` requested a muscle-building plan optimized with a focus on protecting arms from over-fatigue, and shared the [resulting Perplexity search](https://www.perplexity.ai/search/Create-a-customized-mxLrpmM8QnSfpDbVeYkI_g#0). They expressed satisfaction with GPT-4's detailed workout including sets and reps.
- **Ourdigital Investigates Digital Analytics with Perplexity**: `@ourdigital` utilized Perplexity to gather and organize information for digital analytics and performance marketing, sharing his findings in a [Perplexity link](https://www.perplexity.ai/search/What-are-some-1DA8EQPJQK67Zk0qm1M2tg).
- **Exploring Mistral's Capabilities**: Several users, including `@manbearpig86`, `@rhysd21`, and `@dailyfocus_daily`, were looking into comparisons between Mistral and other models like ChatGPT, as reflected in their shared [Perplexity search links](https://www.perplexity.ai/search/mistral-vs-chatgpt-HFXN1aGyTaOZV_CwE0CNAQ#1), [another comparison](https://www.perplexity.ai/search/mistral-vs-chatgpt-2MI6cSqrSOOJmsOGlC3KNg), and a [Starcoder announcement](https://www.perplexity.ai/search/Starcoder-2-Announcement-siMrjqpzSxuXAkkWLe49EQ#0). 
- **Podcast Prompt Crafting and AI Future Discussions**: `@_paradroid` shared a [Perplexity link](https://www.perplexity.ai/search/You-will-act-hEljiMC4SqWMacvlhk4Njw) for crafting a podcast prompt for "48 Hours of AI" and another link discussing Russia's preparation for future challenges, likely with AI, using a ResearchGPT prompt ([ResearchGPT prompt link](https://www.perplexity.ai/search/Task-instructions-You-mrQfNL47S.y_inbGP2cdUQ)).
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1212354113116045374) (28 messages🔥): 

- **Glitch Hunt in Text Generation**: `@thedigitalcat` pointed out that glitches often occur when the system attempts to generate source information during text production. Other users like `@brknclock1215` and `@clay_ferguson` contributed to the discussion, suggesting that the issue could relate to the implementation of sources and the inference layer's approach.
  
- **Sonnar Medium's Weather Query Passion**: `@brknclock1215` humorously continued to test **sonar-medium-online** with weather-related queries, reporting inconsistent behaviors related to the retrieval system and making observations about the presence of "responsive" elements in system messages.

- **The Nostalgia for pplx-70b**: Amidst discussions on model performance, `@thedigitalcat` humorously suggested that everyone will eventually agree that **pplx-70b** was superior to **sonar** models, with `@lazysucker` expressing agreement.

- **The API Conundrum**: `@jeffworthington` encountered an error when using an OpenAPI definition from the provided documentation and queried whether a newer version should be referenced, indicating potential issues with the existing API definitions.

- **Seeking Perplexity's API for Voice Chat**: `@tom_primozic` inquired about using **Perplexity AI**'s functionality through an API for a voice chat application, noting discrepancies in response quality between the website and `sonar-medium-online` model.

**Links mentioned**:

[Getting Started with pplx-api](https://docs.perplexity.ai/docs/getting-started): You can access pplx-api using HTTPS requests. Authenticating involves the following steps:Start by visiting the Perplexity API Settings page. Register your credit card to get started. This step will n...

  

---



### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1212781444720627783) (1 messages): 

- **Launch of Foundation Model Development Cheatsheet**: `@hailey_schoelkopf` announced the release of **The Foundation Model Development Cheatsheet**, a resource to assist new open model developers. The cheatsheet was a collaborative effort featuring contributors from EleutherAI, MIT, AI2, Hugging Face, and other institutions, aiming to provide an overview of resources for responsible open model development.
- **The Cheatsheet Champions Open Model Pioneers**: Highlighting the importance of open model development, `@hailey_schoelkopf` pointed out the release of fully transparent models such as the Pythia model suite by EleutherAI, Amber by the LLM360 project, and AI2's OLMo, emphasizing the growth of openly available models since April 2023.
- **Focus on Dataset Documentation and Licensing**: The new resource focuses on important and underdiscussed areas in model development like dataset documentation and licensing practices, which are crucial for creating open models.
- **Where to Find the Cheatsheet**: The **Foundation Model Development Cheatsheet** can be accessed as a [PDF paper](https://github.com/allenai/fm-cheatsheet/blob/main/app/resources/paper.pdf) or viewed as an [interactive website](https://fmcheatsheet.org/). Updates and additional context are available in their [blog post](https://blog.eleuther.ai/fm-dev-cheatsheet/) and [Twitter thread](https://twitter.com/AiEleuther/status/1763219826602901518).
  

---


### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1212386645391708210) (34 messages🔥): 

- **Seeking Cross-Attention SSM Model**: `@_michaelsh` inquired about models with cross-attention similar to BERT for sequence classification; `@stellaathena` suggested models could be trained as encoders and later mentioned **StripedHyena**, which alternates attention and SSM layers. `@frazermc` favored `adaLN0` with `mamba`, and although there wasn't a pretrained mamba for sequence classification readily available, it was suggested that one could train a classification head on an existing checkpoint.

- **Stable Video Diffusion Inquiry**: `@clashluke` was looking for guidance on how to train/fine-tune the stable video diffusion model, looking to retain its v-prediction while noting it uses `EulerDiscrete` without a `get_velocity` function for training.

- **Understanding lm-evaluation-harness**: Several users, including `@slowturtle_p`, `@hailey_schoelkopf`, and `@maya_liv`, discussed nuances of the lm-evaluation-harness evaluation tool, including score normalization, model substitution with custom code, and potential TensorRT support. `@stellaathena` provided a link to a blog post for further clarification on multiple-choice normalization.

- **EleutherAI Pythia Model Status**: Question from `@mistobaan` about the status of EleutherAI/pythia-13m model, to which `@catboy_slim_` clarified it is still available if referring to the 14m variant.

- **Various Discussion and Announcements**: Users like `@canadagoose1` shared logistical challenges and announcements about talks, `@gaindrew` highlighted an abstract of a research paper introducing a 1-bit Large Language Model, `@tastybucketofrice` and `@hailey_schoelkopf` celebrated user engagement with specific datasets, and `@ilovescience` noted automated downloads likely from using `lm-eval-harness`.

**Links mentioned**:

- [Multiple Choice Normalization in LM Evaluation](https://blog.eleuther.ai/multiple-choice-normalization/): There are multiple ways of evaluating multiple choice tasks on autoregressive LMs like GPT-3/Neo/J. This post lays out the current prevalent normalization methods.
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [Oogway Master Oogway GIF - Oogway Master Oogway Kung Fu Panda - Discover &amp; Share GIFs](https://tenor.com/view/oogway-master-oogway-kung-fu-panda-gif-26485559): Click to view the GIF
- [Meet](https://meet.google.com/hjc-obwu-kjf): Real-time meetings by Google. Using your browser, share your video, desktop, and presentations with teammates and customers.
- [Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/978)): A framework for few-shot evaluation of language models. - Issues · EleutherAI/lm-evaluation-harness

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1212385555602604142) (63 messages🔥🔥): 

- **Open Source Models Galore**: @maxmatical shared a Twitter link to some open-sourced models with accompanying data, posting a tweet from [BigCodeProject](https://twitter.com/BigCodeProject/status/1762842312005026258).

- **Pretraining Token Queries**: In a discussion initiated by @leegao_ about the pretraining token-to-model size ratio, @stellaathena clarified, "*There are no rules*," regarding the expectations of tokens for pretraining models. @maxmatical provided a [link to a paper on arXiv](https://arxiv.org/abs/2305.16264) discussing pretraining with constrained data.

- **Navigating Mazes with Diffusion Models**: @.the_alt_man highlighted a diffusion model trained to solve mazes, sharing tweets from @francoisfleuret and @ArnaudPannatier. @uwu1468548483828484 also chimed in, relating it to prior work on solving mazes with variable depth neural networks.

- **Prompt Engineering Transferability Discourse**: @thatspysaspy asked if there's been study on prompt engineering transfer from small to big models; @catboy_slim_ replied with personal experiences, noting that while generic engineering transfers reasonably well, complex instructions tend to be tightly coupled with specific models. A systematic study with statistical measures seems to be an untapped area.

- **The Challenges of Sub 8 Bit Quantization**: A series of messages from @kd90138 and @clock.work_ expressed skepticism about the practicality and scaling potential of 1-bit Large Language Models given current hardware trends and geopolitical concerns impacting chip manufacturing.

**Links mentioned**:

- [Stable LM 2 1.6B Technical Report](https://arxiv.org/abs/2402.17834): We introduce StableLM 2 1.6B, the first in a new generation of our language model series. In this technical report, we present in detail the data and training procedure leading to the base and instruc...
- [Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/#learning-concrete-scores-with-score-entropy): no description found
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264): The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the...
- [LeoLM: Igniting German-Language LLM Research | LAION](https://laion.ai/blog/leo-lm/.): &lt;p&gt;We proudly introduce LeoLM (&lt;strong&gt;L&lt;/strong&gt;inguistically &lt;strong&gt;E&lt;/strong&gt;nhanced &lt;strong&gt;O&lt;/strong&gt;pen &lt;strong&gt;L&lt;/strong&gt;anguage &lt;stron...
- [Tweet from François Fleuret (@francoisfleuret)](https://x.com/francoisfleuret/status/1762866220636807219?s=20): We train a discrete diffusion denoising model to find paths in a maze. The visualization of the evolution of x_0|x_t (last message in the thread) is very cool IMO.  ↘️ Quoting Arnaud Pannatier (@Arnau...

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1212456356146053211) (3 messages): 

- **Inquiring About Animation Creation**: `@.the_alt_man` asked how a certain animation was made, expressing curiosity about the method or tool used.
- **`imageio` for GIFs**: In response, `@kyo_takano` mentioned that `imageio` was used to create the GIF animation. `@.the_alt_man` followed up for confirmation to clarify that the animation was indeed created with `imageio`.
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1212342612238012427) (15 messages🔥): 

- **Matrix Norms and Products Simplified**: `@wendlerc` explained that matrix vector & matrix matrix products, as well as matrix norms, are shorthand for computing and summing up important cosines. The matrix-2-norm is specifically the matrix norm associated with the vector 2-norm.
- **Decoding Details in RMSNorm Implementation**: `@wendlerc` clarified a subtle detail that their paper does not explicitly mention: the final decoding step involves an RMSNorm layer application to `h` before matrix multiplication. They described a computational split of this process for ease in cosine calculations between resulting expressions.
- **Unpacking the Tuned Lens Decoding Process**: `@wendlerc` and `@mrgonao` discussed the mechanism of decoding using a tuned lens in neural networks. They considered whether `logits = U RMSNormlayer(tunedlens(h))` accurately represents the tuned lens's activity.
- **Implementation Nuances of Tuned Lens and Notation**: Throughout the conversation, `@wendlerc` addressed the practical aspects of porting their implementation to consider the tuned lens's effect, highlighting the necessity of substituting `h` with `tunedlens(h)`.
- **Understanding Matrix Norm Terminology**: `@norabelrose` clarified the terminology around matrix norms, stating that the Frobenius norm relates to the Euclidean norm of the matrix when flattened, whereas the "2-norm" of a matrix refers to its spectral norm or top singular value.
  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1212467835247263796) (19 messages🔥): 

- **Tinkering with LM Eval Harness**: `@paganpegasus` inquired about integrating instruction/chat formatting into the LM Eval harness or considering finetuning on examples with existing eval harness formatting.

- **Custom Model Modification for Hallucination Leaderboard**: `@pminervini` shared a snippet of code from their approach to incorporate chat templates into the LM Eval harness for the hallucinations leaderboard, by extending the `HFLM` class.

- **Awaiting Progress on Proposed Modifications**: `@asuglia` updated `@981242445696221224` on the status of modifications being identified for a project, noting other tasks had taken precedence.

- **Improving Multilingual Lambada Translations**: `@hailey_schoelkopf` mentioned that `@946388490579484732` contributed new, higher-quality translations to replace poor quality ones, and the changes will be integrated into the eval harness. The updated dataset includes additional languages, and is available on [Hugging Face](https://huggingface.co/datasets/marcob/lambada_multilingual).

- **Implementing EQ-Bench**: `@pbevan1` sought advice on implementing EQ-Bench, a benchmark for emotional intelligence in language models, especially tasks that handle multiple answers for a single prompt. `@hailey_schoelkopf` pointed to the Truthfulqa_mc2 task as an example.

**Links mentioned**:

- [src/backend/huggingface_generate_until.py · hallucinations-leaderboard/leaderboard at main](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/blob/main/src/backend/huggingface_generate_until.py#L9): no description found
- [GitHub - EQ-bench/EQ-Bench: A benchmark for emotional intelligence in large language models](https://github.com/EQ-bench/EQ-Bench/tree/main_v2_1): A benchmark for emotional intelligence in large language models - EQ-bench/EQ-Bench
- [marcob/lambada_multilingual · Datasets at Hugging Face](https://huggingface.co/datasets/marcob/lambada_multilingual): no description found

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1212555902704549909) (2 messages): 

- **Choosing Between Encoder-Decoder and Decoder-Only Models**: User `@jerry0478` inquired about when to use **cross-attention conditioning** as seen in encoder-decoder models compared to embedding tokens in input for **decoder-only models**.
- **Flamingo vs. LLaMA Architecture Decisions**: `@jerry0478` contrasted "llama-style" architectures with "flamingo-style" ones, probing the community on intuition for optimal application scenarios of each.
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1212493712865763359) (2 messages): 

- **Inquiring about Neox and Slurm**: `@muwnd` asked for the recommended method to run **Neox** with **Slurm** and **Containers**, suspecting that `--launcher_args` might be the way but noted it seems unavailable in Neox.
- **Tip on Neox Infrastructure**: `@triggerhappygandhi` clarified that Neox does not assume any specifics about the infrastructure, and containers need to be set up in advance. A **slurm script** exists for using Slurm to run Neox on multinode.
  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1212357015112777739) (89 messages🔥🔥): 

- **Seeking Confidence Score Insight**: User `@ritanshoo` inquired about checking the confidence score when using LangChain.js for RAG. Kapa.ai did not have an immediate answer but **referred** to the **LangChain documentation** (<https://js.langchain.com/docs/get_started>) for further exploration.

- **Contemplating Memory Integration with LCEL**: Both `@marknicholas` and `@pcube__` discussed different aspects of LangChain usage. `@marknicholas` wanted to **add memory** to LCEL, and `@pcube__` inquired about **which language integrates best** with LangChain for a server using azure hosted LLM as an API endpoint. Kapa.ai suggested **consulting official documentation** or reaching out to the community for specific guidance.

- **Handling Tool Exceptions in Custom Applications**: `@abinandan` requested a way to retry a tool if `ToolException` is thrown when using a **custom tool**. Kapa.ai **highlighted workarounds** from LangChain's GitHub discussions and encouraged checking **LangChain's GitHub issues** for more streamlined solutions (<https://github.com/langchain-ai/langchain/issues/10714>).

- **Using Shopify as an Automated Agent/Tool**: User `@erikk4` sought **automation solutions** for customer support tasks related to **Shopify**, such as checking order statuses or canceling orders. They considered "front desk" agents routing issues to specific tools and queried the community for tools beyond LangChain that might facilitate this process.

- **Deployment Issues and Adding Functionality with LangChain**: Users conveyed challenges with LangChain's deployment and functionality. `@hanumantgarad_25732` experienced an `AttributeError` when using `SQLDatabase.from_databricks` outside a Databricks notebook. `@kamakshi08` asked about **using the JSON parser** with **LLaMA** from Ollama, wondering how it integrates with **multimodal models**.

**Links mentioned**:

- [no title found](https://js.langchain.com>)): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/9BHf9tdSSd): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Revolutionizing AI Interactions: Integrating Function Calling with Mistral](https://medium.com/@arapbiisubmissions/blog-title-revolutionizing-ai-interactions-integrating-function-calling-with-mistral-8486d1841e50): Introduction
- [Querying a SQL DB | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/cookbook/sql_db): We can replicate our SQLDatabaseChain with Runnables.
- [JSON parser | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/model_io/output_parsers/types/json): This output parser allows users to specify an arbitrary JSON schema and
- [Docusaurus | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/document_loaders/docusaurus#filtering-sitemap-urls>)): Docusaurus is a static-site generator which
- [Custom Agent Class fails with object has no attribute &#39;is_single_input&#39; · Issue #18292 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/18292): Checked other resources I added a very descriptive title to this issue. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and di...
- [Groq: Insanely Fast Inference 🚀 | World&#39;s First Language Processing Unit (LPU)](https://youtu.be/RSzG_v5XIxM): In this video, I will explain about Groq who introduced World&#39;s first Language Processing Unit (LPU) designed for AI applications (LLMs). I will show you how...
- [Deployment | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/deployments#outline>)).): In today&#x27;s fast-paced technological landscape, the use of Large Language Models (LLMs) is rapidly expanding. As a result, it is crucial for developers to understand how to effectively deploy thes...
- [langchainjs/langchain/src/retrievers/score_threshold.ts at e24d2dedbe7ff93db33a5809e604143d60113028 · langchain-ai/langchainjs](https://github.com/langchain-ai/langchainjs/blob/e24d2de/langchain/src/retrievers/score_threshold.ts#L24>)): 🦜🔗 Build context-aware reasoning applications 🦜🔗. Contribute to langchain-ai/langchainjs development by creating an account on GitHub.
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/10714>).): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/2024>))): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
- [GenAI Summit San Francisco 2024](https://www.eventbrite.com/e/genai-summit-san-francisco-2024-tickets-796934722207?aff=eemailordconf&utm_campaign=order_confirm&ref=eemailordconf&utm_medium=email&utm_source=eventbrite&utm_term=viewevent): This summit is an extraordinary convergence of the brightest minds in Generative AI, encapsulating the spirit of the future. #AI_ARE_ALL

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1212755246955237459) (3 messages): 

- **LangServe Agent Troubles**: `@thatdc` reported an issue where their agent is not returning the **intermediate steps of execution** when using **langserve**; however, it works fine when invoking directly from the agent class. They deduced the problem might be with the API server setup by langserve.
- **Deep Dive into the Tech Snag**: `@thatdc` believes to have found the problem in the `RemoteRunnable` object where the `_decode_response` method seems to lose the intermediate steps by executing `serializer.loadd(obj["output"])`. They're in search of a workaround for this issue.
  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1212394712061714482) (2 messages): 

- **Invitation to Join the Discord Party**: `@davisson0429` posted a [Discord invite link](https://discord.gg/9BHf9tdSSd) for users to join, accompanied by a lengthy series of separator characters.
- **Seeking Python Template Wisdom**: `@tigermusk` inquired about generating a template in Python code that resembles the one found at [Smith LangChain Chat JSON Hub](https://smith.langchain.com/hub/hwchase17/react-chat-json).

**Links mentioned**:

- [LangSmith](https://smith.langchain.com/hub/hwchase17/react-chat-json): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/9BHf9tdSSd): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1212351604972322846) (4 messages): 

- **"LangChain in your Pocket" Hits the Shelves**: User `@mehulgupta7991` celebrated the listing of their debut book "LangChain in your Pocket" under Google's Best books on LangChain.

- **Flood of Discord Invites**: `@davisson0429` shared an invite link to a Discord server with a string of obscured characters following the URL, and an @everyone tag, possibly indicating a call to join.

- **Calling All Learners**: User `@silvermango9927` shared a Google Form link soliciting feedback on interest in various topics such as Machine Learning, Data Science, and Web Development, as part of a validation process for a project they are considering.

- **Voices of the Future**: `@beaudjango` introduced "Pablo," an AI Voice Chat app that supports multiple LLMs and voices without the need for typing, inviting beta testers to join with an [offer for free AI credits](https://testflight.apple.com/join/raZGq35o). They mentioned looking for engineers willing to join their team using LangChain.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/9BHf9tdSSd): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Join the Pablo - AI Voice Chat beta](https://testflight.apple.com/join/raZGq35o): Available on iOS
- [Product Idea Validation Form](https://forms.gle/j48JLAeJWZRryX7c8): Hi, thank you so much for filling in this form and giving a response.   The idea : Creating a lab (course) that teaches in a project-based manner compared to all of the conventional longer video-heavy...

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1212351332900413450) (4 messages): 

- **Question on LangGraph Capabilities**: User `@tigermusk` inquired whether `workflow.compile()` is a runnable object in LangGraph.
- **Spam Alert**: `@davisson0429` posted an unrelated and spammy invite link to an external Discord server filled with severe text repetition.
- **Groq's LPU Breakthrough Showcased**: `@datasciencebasics` shared a [YouTube video titled "Groq: Insanely Fast Inference 🚀 | World's First Language Processing Unit (LPU)"](https://youtu.be/RSzG_v5XIxM) highlighting the introduction of the world's first Language Processing Unit designed for AI applications, showcasing its potential for LLMs.
- **LangGraph + YahooFinance Tutorial**: `@tarikkaoutar` provided a [video guide](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s) explaining how to create an AI stock analysis chatbot using LangGraph, Function call, and YahooFinance, enhancing understanding of multi-agent applications.

**Links mentioned**:

- [Join the ONE PERCENT CLUB Discord Server!](https://discord.gg/9BHf9tdSSd): Check out the ONE PERCENT CLUB community on Discord - hang out with 16193 other members and enjoy free voice and text chat.
- [LangGraph + Function Call+ YahooFinance =  Multi-Agent Application](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s): #chatbot #animation #trading  #ai #machinelearning #datascience In this video, you will make an AI stock analysis chatbot with LangGraph, Function call and C...
- [Groq: Insanely Fast Inference 🚀 | World&#39;s First Language Processing Unit (LPU)](https://youtu.be/RSzG_v5XIxM): In this video, I will explain about Groq who introduced World&#39;s first Language Processing Unit (LPU) designed for AI applications (LLMs). I will show you how...

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1212327480304730142) (44 messages🔥): 

- **Trouble in Jupyter Town**: `@nruaif` shared a log indicating issues with Jupyter notebooks, showing error messages related to extensions being linked and a **Bad config encountered during initialization**. `@nanobitz` chimed in asking if it was a template or Jupyter issue.
  
- **BitNet b1.58 Makes Waves**: `@_dampf` shared an [arXiv paper on BitNet b1.58](https://arxiv.org/abs/2402.17764), a 1-bit LLM that promises significant cost-efficiency with performance matching full-precision models. `@nanobitz` mentioned it's not just a quantization method but a new architecture.

- **Axolotl User Survey Outreach**: `@caseus_` is seeking feedback through a [questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSeyJkTk7sCYWpCNfKNNpnlMQlT9XU2nt_TJCzP4GSZBT0vrRA/viewform) to improve understanding of axolotl users. `@dreamgen` suggested making the form more concise to get more responses.

- **Mistral Office Hours Announcement**: `@casper_ai` shared an invite to the next [Mistral AI office hour](https://discord.gg/mistralai?event=1204405056825327677).

- **Alpaca Formatting for Inferences**: `@j_sp_r` inquired about formatting inferences to match the training instruction format, and `@caseus_` responded that specifying `chat_template: alpaca` in the axolotl YAML will handle it.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/mistralai?event=1204405056825327677): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...
- [TinyBox packs a punch with six of AMD's fastest gaming GPUs repurposed for AI &mdash; new box uses Radeon 7900 XTX and retails for $15K, now in production](https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production): Startup wants to offer high AI performance using Radeon RX 7900 XTX.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b29eax/me): no description found
- [Axolotl End User Questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSeyJkTk7sCYWpCNfKNNpnlMQlT9XU2nt_TJCzP4GSZBT0vrRA/viewform): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1212533327379308576) (9 messages🔥): 

- **KTO Trainer Implementation Inquiry**: `@giftedgummybee` shared a [link to Huggingface's documentation](https://huggingface.co/docs/trl/main/en/kto_trainer) on the Kahneman-Tversky Optimization (KTO) Trainer and asked `@257999024458563585` if there are any plans to implement it. `@caseus_` responded affirmatively, suggesting they might work on it the following week unless someone else takes it up earlier.
- **Sophia: A Speedy Optimizer**: `@casper_ai` discussed the potential of [Sophia optimizer](https://arxiv.org/abs/2305.14342) being twice as fast as Adam algorithms and supplied the [implementation link](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/optim/sophia.py) (not torch) for Sophia, highlighting its advantage in efficiency over traditional optimization methods.
- **Innovative Training with DropBP**: `@suikamelon` brought up a study on [Dropping Backward Propagation (DropBP)](https://arxiv.org/abs/2402.17812), which reduces computational costs of neural network training while preserving accuracy by dropping layers during backward propagation.
- **Starcoder2 Training Support**: `@faldore` inquired about support for [Starcoder2](https://github.com/bigcode-project/starcoder2?tab=readme-ov-file#training), providing a link to its GitHub repository.

**Links mentioned**:

- [DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation](https://arxiv.org/abs/2402.17812): Training deep neural networks typically involves substantial computational costs during both forward and backward propagation. The conventional layer dropping techniques drop certain layers during tra...
- [KTO Trainer](https://huggingface.co/docs/trl/main/en/kto_trainer): no description found
- [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342): Given the massive cost of language model pre-training, a non-trivial improvement of the optimization algorithm would lead to a material reduction on the time and cost of training. Adam and its variant...
- [levanter/src/levanter/optim/sophia.py at main · stanford-crfm/levanter](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/optim/sophia.py): Legible, Scalable, Reproducible Foundation Models with Named Tensors and Jax - stanford-crfm/levanter
- [GitHub - bigcode-project/starcoder2: Home of StarCoder2!](https://github.com/bigcode-project/starcoder2?tab=readme-ov-file#training): Home of StarCoder2! Contribute to bigcode-project/starcoder2 development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1212386061637124126) (22 messages🔥): 

- **Pondering Plausible Intentions**: `@nafnlaus00` floated the idea of prompting a sophisticated language model to **generate intentionally wrong answers** that seem plausible but contain flaws leading to incorrect conclusions, though no further discussion ensued.
- **Tool Swap Troubles**: `@stoicbatman` contemplated switching from **Runpod** to **Vast AI** due to cost concerns and sought the community's experience comparison; `@nanobitz` responded noting that although cheaper, **Vast AI** doesn't abstract machine details and offers variable machine quality.
- **Confusing Commit Conundrums**: `@karisna` expressed disappointment that their commit to rewrite documentation for **axolotl** wasn't accepted and pointed out a possible oversight where **WSL2 setup** for Windows isn't sufficiently emphasized; however, `@nanobitz` replied looking to clarify if the documentation issue had been addressed.
- **Benchmarks for the Brainy**: `@jovial_lynx_74856` inquired about running benchmarks on a model finetuned with **Axolotl**, and `@nanobitz` suggested looking at **lm_eval_harness** on Github, affirming there's no direct integration for benchmarking within Axolotl itself.
- **Save Setting Snafu**: Concerned about a saving discrepancy, `@duke001.` asked why setting `saves_per_epoch` to 4 and `num_epochs` to 4 resulted in only 4 checkpoints instead of the expected 16; `@nanobitz` hinted at a resolution suggesting an adjustment to the save limit.

**Links mentioned**:

[axolotl/src/axolotl/core/trainer_builder.py at 6b3b271925b2b0f0c98a33cebdc90788e31ffc29 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/6b3b271925b2b0f0c98a33cebdc90788e31ffc29/src/axolotl/core/trainer_builder.py#L887): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1212754289269809224) (11 messages🔥): 

- **Mistral Model Rivals ChatGPT 3.5**: `@le_mess` shared that their **7B Mistral model** matches the performance of **ChatGPT 3.5** for Danish tasks.
- **Performance Strengthens Through Iterative Training**: `@le_mess` improved their models by using a **synthetic data approach** and training over **30 iterations**, enhancing responses over time without relying on **GPT-4**.
- **Initial Human Curation Leads to Scalable Model Training**: `@le_mess` curated the first 1000 responses manually, then employed models to generate more data. Subsequent models were trained to identify high-quality responses for further training cycles.
  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1212428111434743878) (4 messages): 

- **Groq Accelerates LlamaIndex**: The `@GroqInc` LPU now officially integrates with LlamaIndex and supports `llama2` and `Mixtral` models for efficient LLM generation. They announced this development with a [cookbook guide](https://t.co/zBiBlgadVh), for streamlining application workflows.

- **LlamaParse Sees Soaring Usage**: `@llama_index` reports significant usage of LlamaParse, leading to important updates, such as working towards uncapped self-serve usage and temporarily increasing the usage cap from 1k pages. Details can be found at this [update link](https://t.co/tsfAEjziku).

- **Optimizing Hybrid Search with LLMs**: A new strategy for better retrieval in hybrid search uses LLMs to categorize queries with few-shot examples and subsequently adjust the alpha parameter. `@llama_index` shares insights into this approach in [their latest tweet](https://t.co/39Lk5nEOoc).

- **RAG for Structured and Unstructured Data**: `@llama_index` introduced a blog post by `@ClickHouseDB` showcasing a RAG architecture suited for queries involving both unstructured and structured data, housed in the same database. Interested readers can delve into this integration [here](https://t.co/oy79TexCYR).
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1212386022370054204) (75 messages🔥🔥): 

- **Exploring LlamaIndex Documentation Indexing**: `@vaguely_happy` proposed setting up a service to index the latest LlamaIndex docs, which prompted `@cheesyfishes` to mention *mendable* on docs and `@whitefang_jr` informing about LlamaParse not currently sending page numbers but work is in progress to add page numbers and labels.

- **Clarification on Callbacks in Golang**: As `@sansmoraxz` questioned the use of `CallbackHandler` with native types, `@cheesyfishes` assured a refactor is in progress for callbacks and advised holding off on concerns for the moment due to expected improvements.

- **Debating Reranker Models**: In a discussion initiated by `@richard1861` regarding the superior reranking model between Colbert and Cohere, `@.sysfor` shared code and suggested using both the FlagEmbeddingReranker and CohereReranker together, despite having no formal metrics to compare their performance.

- **Visualizing ReActAgent Pipelines/DAGs**: `@mrpurple9389` inquired about visualizing the graph for ReActAgent, and while `@cheesyfishes` clarified that ReActAgent lacks a visual graph, `@mrpurple9389` further explored visualizing the agent if replicated using pipelines/DAGs.

- **Discussions on LlamaIndex vs. Langchain and Compatibility**: `@tr1ckydev` sought clarification on the differences between LlamaIndex and Langchain, with `@cheesyfishes` explaining that LlamaIndex focuses on connecting data to LLMs while Langchain is more of a comprehensive library. Follow-up queries included compatibility inquiries, indicating that LlamaIndex can be integrated with various vector databases and LLM platforms.

**Links mentioned**:

- [Introducing LlamaCloud and LlamaParse — LlamaIndex, Data Framework for LLM Applications](https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b): LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).
- [Arize Phoenix - Phoenix](https://docs.arize.com/phoenix/): no description found
- [Ollama - Llama 2 7B - LlamaIndex 🦙 v0.10.14](https://docs.llamaindex.ai/en/stable/examples/llm/ollama.html): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1212488361206489140) (5 messages): 

- **Model Decay Woes**: User `@.sysfor` expressed concerns that their models have been generating **insane responses** recently, questioning whether models decay over time with the hypothesis that nothing else has changed in the setup.
- **Cheesyfishes to the Rescue**: `@cheesyfishes` clarified that models **do not decay** over time, but **longer inputs or inputs not structured as instructions** could potentially lead to issues with the model's responses.
- **Observable Decline in Fine-tuned Performance**: Further to the decay question, `@.sysfor` noticed issues specifically with the "better" fine-tuned models, while running tests to compare against baseline models.
  

---



### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1212407499781832796) (49 messages🔥): 

- **Claude Models Prompt Errors**: `@quentmaker` reported an error when a chat has more than 8 alternating messages between user and assistant, affecting various Anthropics' Claude models. `@louisgv` acknowledged the issue and promised a fix is in the works.

- **OpenRouter Addressing Turn Order Issues**: `@alexatallah` suggested a temporary workaround for the prompt issue by changing the first assistant message to a system message. Meanwhile, development is underway to handle conversations that begin with a message from the assistant.

- **Rate Limit Discussions for OpenRouter**: `@gunpal5_43100` inquired about rate limits when using OpenRouter for generating large numbers of articles. `@alexatallah` clarified that each user with their own API key would have separate rate limits, which cumulatively should provide sufficient throughput.

- **Caching Concerns with Mistral**: Several users, including `@natefyi_30842` and `@spaceemotion`, observed similarities in responses when repeating prompts to Mistral models, leading to speculation of caching behavior by the API. `@alexatallah` confirmed that Mistral's API might cache queries.

- **Compatibility with Prepaid Cards**: `@fakeleiikun` asked about OpenRouter's support for prepaid cards, particularly those provided by e-wallet apps. `@louisgv` indicated that while some prepaid cards might work, virtual cards from unsupported banks might not be accepted due to Stripe's fraud prevention measures.

**Links mentioned**:

- [no title found](https://bluegpt.app)): no description found
- [OpenRouter](https://openrouter.ai/docs#limits): Build model-agnostic AI apps

  

---



### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1212531021627785216) (10 messages🔥): 

- **Benchmark Script Enhanced**: `@hdcharles_74684` improved a benchmark script for comparing **Triton** kernel performance, which could be beneficial for **int8 weight-only linear kernels** potentially outperforming cuBLAS for batch sizes greater than 1, impacting **sdxl-fast**. The [script is available on GitHub](https://gist.github.com/HDCharles/a7fc12b31702cf963d8453e0da157296), and contains various kernels, including **fast kernel for bs=1**, **int4 tinygemm**, and **uint4x2 triton kernel**.
- **PR to cuda-mode/lectures Suggested**: `@marksaroufim` suggested `@hdcharles_74684` make a pull request to the [cuda-mode lectures](https://github.com/cuda-mode/lectures) repository on GitHub to make the benchmark script easily accessible.
- **Potential Triton Optimizations Discussed**: `@chhillee` mentioned that **Torch.compile** could efficiently handle batch size of 2, which could alleviate the main bottleneck in question.
- **Tensor Performance Fixed on Radeon**: `@iron_bound` reported a significant improvement in **tensor performance** on **Radeon RX 7900 XTX** graphics card after fixing an issue with **WMMA hooks** in **mlir/llvm**.
- **Debugging Issue with Triton Versions**: `@kierandidi` encountered an issue with the Triton debugger in versions **3.0.0** and **2.2.0** regarding the `interpret` argument. `@andreaskoepf` and `@marksaroufim` confirmed that the method was deprecated and suggested setting `TRITON_INTERPRET` environment variable as a workaround.
- **Feedback on Triton's Stability**: `@andreaskoepf` shared experiences of instabilities with **Triton** compared to **CUDA**, citing unexplained segfaults and inconsistent results. `@marksaroufim` requested an example to compare the situations before and after the segfaults, following similar feedback observed on Twitter.

**Links mentioned**:

- [GitHub - cuda-mode/lectures: Material for cuda-mode lectures](https://github.com/cuda-mode/lectures): Material for cuda-mode lectures. Contribute to cuda-mode/lectures development by creating an account on GitHub.
- [script for comparing performance of several linear triton kernels across several shapes](https://gist.github.com/HDCharles/a7fc12b31702cf963d8453e0da157296): script for comparing performance of several linear triton kernels across several shapes - linear_triton_kernels.py

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1212784318867312650) (6 messages): 

- **Inquiry about GPU Intrinsics**: User `@drexalt` asked if a claim made in a [tweet](https://twitter.com/cis_female/status/1763221499551604995) was true, seeking clarification from fellow CUDA MODE Discord members.
- **Response to FP8 Intrinsics Query**: `@zippika` clarified that the claim in question was false and provided a [link to the CUDA math API docs](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html) that still lists FP8 intrinsics.
- **Clarifying the Purpose of FP8**: `@zippika` underlined that FP8 serves mainly as a data format rather than being extensively used for computations.

**Links mentioned**:

[CUDA Math API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html#group__CUDA__MATH__FP8__MISC): no description found

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1212416868195897365) (13 messages🔥): 

- **No Appetite for Polyhedral**: `@chhillee` expresses skepticism about the utility of *polyhedral compilation* in optimizing sharding for deep learning, suggesting that the key question is defining the **cost function**.

- **Search Space Skepticism**: In a discussion with `@andreaskoepf`, `@chhillee` likens the challenge of finding optimal shardings in deep learning to the ongoing developments in new **ML architectures**.

- **Contemplating Optimal Mappings**: `@gogators.` muses that the space of valid mappings from **deep learning programs to hardware** may be smaller and less complex than the space of all possible deep learning programs.

- **DL Program Optimization Not So Trivial**: `@gogators.` backtracks from describing the process of finding efficient mappings of deep learning computations as "trivial," while expressing surprise if **top AI institutions** aren't already investigating this area.

- **Debating Deep Learning Computability**: `@telepath8401` humorously challenges `@gogators.`'s initial use of "trivial," prompting a clarification about the feasibility of optimizing operation mappings given homogeneity and explicit dependencies in **deep learning operators**.
  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1212509988145340416) (15 messages🔥): 

- **New Ring Attention Implementations**: `@andreaskoepf` shared [lucidrains' implementation of Ring Attention](https://github.com/lucidrains/ring-attention-pytorch) with custom Triton kernels and proposed to [compare its correctness and performance](https://github.com/cuda-mode/ring-attention/issues/11) with another implementation by zhuzilin.
- **Backward Pass Bug Hunt**: `@andreaskoepf` mentioned that Phil pointed out an issue with the backward pass, which might need fixing, as discussed in [this GitHub issue](https://github.com/lucidrains/ring-attention-pytorch/issues/4#issuecomment-1970029318).
- **GPU Compatibility Troubles**: `@nthanhtam.` and `@jamesmel` reported problems when running the Ring Attention implementation on GPUs, while `@ericauld` noted the assertion script works on CPU.
- **Code Inconsistencies and Errors**: `@ericauld` observed multiple errors in the code when trying to run it with Melvin's suggestions, such as typos and missing imports, which led to additional Triton-related issues.
- **Commit History Suggests Problems**: `@iron_bound` hinted that something might have broken in lucidrains' Ring Attention implementation by referring to the [commit history on GitHub](https://github.com/lucidrains/ring-attention-pytorch/commits/main/).

**Links mentioned**:

- [GitHub - lucidrains/ring-attention-pytorch: Explorations into Ring Attention, from Liu et al. at Berkeley AI](https://github.com/lucidrains/ring-attention-pytorch): Explorations into Ring Attention, from Liu et al. at Berkeley AI - lucidrains/ring-attention-pytorch
- [Commits · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/commits/main/): Explorations into Ring Attention, from Liu et al. at Berkeley AI - Commits · lucidrains/ring-attention-pytorch
- [A ring attention with flash attention kernel implementation · Issue #4 · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/issues/4#issuecomment-1970029318): Hi! Thank you for your work on implementing the ring attention in pytorch! I&#39;ve just tried to implement a ring_flash_attn_qkvpacked_func (corresponding to flash_attn_qkvpacked_func in flash attent...
- [Compare ring-flash-attention &amp; ring-attention-pytorch · Issue #11 · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/issues/11): lucidrains &amp; zhuzilin were hard working the last days and have completed the following two ring-attention implementations: lucidrains/ring-attention-pytorch zhuzilin/ring-flash-attention Create a ...

  

---



### Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1212377918651637780) (10 messages🔥): 

- **Arthur Mensch Sets the Record Straight**: `@arthurmensch` clarified misconceptions about their recent announcements, reiterating the commitment to open-weight models with 1.5k H100s, a reselling agreement with Microsoft, and maintaining independence as a European company with global ambitions. He highlighted the growing interest for Le Chat and Mistral Large on La Plateforme and Azure, with a plan to iterate quickly. [Check out the clarifications](https://x.com/arthurmensch/status/1762818733016322168?s=46).

- **Nathan Endorses Public Clarifications**: After the tweet from `@arthurmensch`, `@natolambert` expressed approval, describing the act of providing such public clarifications on social media as "def legit vibes".

- **Announcing StarCoder2 and The Stack v2**: `@BigCodeProject` launched StarCoder2, a model trained with a 16k token context and a massive 4T+ token repository-level information, built upon The Stack v2 which contains over 900B+ tokens. The code, data, and models are fully open and available, marking a significant contribution to the community. [Discover StarCoder2](http://hf.co/bigcode/starcoder2-15b).

- **Meta Prepares to Launch Llama 3**: A tweet from `@Reuters` reported that Meta plans to release a new AI language model dubbed Llama 3 in July, which could signify another major competition in the AI field. The details were reported by The Information. [Read more from Reuters](http://reut.rs/3TgBgFJ).

- **G 1.5 Pro with Extended Context Coming to Nathan**: `@natolambert` announced excitement for getting access to G 1.5 Pro with a 1 million token context, planning to use it for processing podcasts and other content, and mentioned a potential article workshop based on the experience, if there's interest.

**Links mentioned**:

- [Tweet from BigCode (@BigCodeProject)](https://fxtwitter.com/BigCodeProject/status/1762842312005026258): Introducing: StarCoder2 and The Stack v2 ⭐️  StarCoder2 is trained with a 16k token context and repo-level information for 4T+ tokens. All built on The Stack v2 - the largest code dataset with 900B+ t...
- [Tweet from Arthur Mensch (@arthurmensch)](https://x.com/arthurmensch/status/1762818733016322168?s=46): Clarifying a couple of things since we’re reading creative interpretations of our latest announcements: - We’re still committed to leading open-weight models! We ask for a little patience, 1.5k H100s ...
- [Tweet from Reuters (@Reuters)](https://x.com/reuters/status/1762894264462176676?s=46): Meta plans launch of new AI language model Llama 3 in July, The Information reports http://reut.rs/3TgBgFJ

  

---


### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1212434435472105533) (30 messages🔥): 

- **Nathan Lambert Tunes into Demis Hassabis**: `@natolambert` shared an episode of a podcast with [Demis Hassabis, CEO of Google DeepMind](https://open.substack.com/pub/dwarkesh/p/demis-hassabis?r=68gy5&utm_medium=ios), discussing **superhuman AI scaling, AlphaZero training atop LLMs, and AI governance**. The podcast can be watched on [YouTube](https://youtu.be/qTogNUV3CAI) or listened to on platforms like [Apple Podcasts](https://podcasts.apple.com/us/podcast/demis-hassabis-scaling-superhuman-ais-alphazero-atop/id1516093381?i=1000647410338) and [Spotify](https://open.spotify.com/episode/6SWbwjYPs5WevIoCCiSByS?si=nCVFSRr7QGGI_STgbrOBDA).

- **Considering Openness in AI Discussions**: `@natolambert` and `@mike.lambert` discussed the merits of having open conversations about **completely open AI** and the differences in mental models as opposed to conversations on platforms like Twitter.

- **Name Coincidence Among Users**: User `@xeophon.` inquired if `@natolambert` and `@mike.lambert` were related due to the similarity in their last names; it was confirmed to be a coincidence.

- **Anthropic Association Confirmation**: `@mike.lambert` confirmed employment at **Anthropic** and took a stance on sharing information in the chat, indicating a preference to engage in discussions as themselves, not as a representative of their employer.

- **The Quest for the LAMB Emoji**: `@natolambert` humorously lamented the lack of an appropriate emoji for "LAMB," expressing frustration with the search results pointing to a steak emoji 🥩.

**Links mentioned**:

[Demis Hassabis - Scaling, Superhuman AIs, AlphaZero atop LLMs, Rogue Nations Threat](https://open.substack.com/pub/dwarkesh/p/demis-hassabis?r=68gy5&utm_medium=ios): &quot;scaling is an artform&quot;

  

---



### LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1212537914081284147) (2 messages): 

- **Inquiry About Benchmark Automation**: `@ampdot` asked if a benchmark is available as an automated script, showing interest in trying out such a tool.
- **Enthusiasm for Benchmark Automation**: `@dare.ai` also expressed interest in the automated benchmark script and is looking forward to trying it out, tagging `<@757392677280022549>` for a potential response.
  

---


### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1212419404843974686) (4 messages): 

- **Anticipated Spring Launch for Llama 3**: User `@res6969` expressed that their expectation was for **Llama 3** to be released in spring, suggesting that the current timeline is further than anticipated.
- **Possible Last-Minute Improvements for Llama 3**: `@potrock` expressed hope that the delay of **Llama 3** might be due to a last-minute attention update, hinting at improvements that could be included in the release.
- **Enthusiasm for Gemini Ring Attention**: `@potrock` mentioned that the incorporation of **Gemini ring attention** would be a cool feature for Llama 3, indicating interest in this specific attention mechanism.
  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1212650486927331358) (1 messages): 

- **Time Crunch for LLM Testing**: User `@jeffreyw128` expressed a desire to test new **LLMs** but emphasized the significant effort required to **"get a good vibe check on each"** due to time constraints.
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1212538529276629162) (3 messages): 

- **ChatGPT Search Update Rumors**: `@jeffreyw128` mentioned **rumors** that OpenAI might be updating their web search in ChatGPT this week, seeking confirmation from others.
- **In Search of OpenAI Insights**: User `@res6969` acknowledged not having heard such rumors and expressed a need to find better sources for OpenAI-related information.
- **Looking for codeinterpreter Production Resources**: `@res6969` inquired if anyone had resources on using **codeinterpreter** in production environments, indicating an interest in practical applications.
  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1212327082026209332) (6 messages): 

- **DiscoLM Template Clarification**: User `@bjoernp` pointed out the importance of using the DiscoLM template for chat context tokenization, referencing the Hugging Face documentation on [chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating#introduction).

- **Issues with llamaindex Chunker for Code**: `@sebastian.bodza` reported that the llamaindex chunker for code was significantly malfunctioning, producing one-liners and disregarding the `chunk_lines` option.

- **Sanity Check on Training German RAG Models**: `@johannhartmann` is creating a German dataset for Retrieve-and-Generate (RAG) tasks, utilizing Deutsche Telekom's Wikipedia content-question pairs, and sought feedback on the approach to improve reliability of German-speaking Mistral 7b models.

- **Goliath versus DiscoLM for German Language Tasks**: `@philipmay` questioned if Goliath is the superior model for German language skills and shared a link to its model card on Hugging Face. The discussion evolved with `@johannhartmann` suggesting that DiscoResearch/DiscoLM-120b might perform better due to its training on German content.

- **Advice on Generating Negative Samples for Datasets**: `@philipmay` suggested a successful method to generate negative samples by directing a language model to alter given answers to be factually incorrect, for the purposes of building a more effective dataset for RAG training.

**Links mentioned**:

- [alpindale/goliath-120b · Hugging Face](https://huggingface.co/alpindale/goliath-120b): no description found
- [Templates for Chat Models](https://huggingface.co/docs/transformers/main/en/chat_templating#introduction)): no description found

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1212858233925013587) (1 messages): 

- **German Prompts in EQ-Bench**: `@crispstrobe` shared that EQ-Bench now supports German prompts, showing strong correlation with various benchmarks like MMLU and Arena Elo. Link to the GitHub pull request is [here](https://github.com/EQ-bench/EQ-Bench/pull/12).

- **GPT-4 Leads in Performance**: According to a comparison shared by `@crispstrobe`, **GPT-4-1106-preview** scored 81.91 in the EQ-Bench German prompts evaluation, outperforming other models including GPT-3.5, various Mistral versions, and `discolm-german-laser`.

- **Evaluating German Language Models**: The message lists EQ-Bench scores for different models, highlighting that even a model like `german-assistant-v7` has a score of 35.48 which could indicate a baseline for German language model performance.

- **Translation Scripts Included**: `@crispstrobe` also mentioned including translation scripts with the benchmarks, stating that these were set up quickly and have the potential for further improvement, such as manual review by a student.

- **Automatic Translation with GPT-4**: The German prompts were automatically translated using **ChatGPT-4-turbo**, showing that sophisticated models can facilitate the translation of test or training sets, a process that can be adapted or changed to other translation services like "free Gemini".

**Links mentioned**:

[Build software better, together](https://github.com/EQ-bench/EQ-Bench/pull/12):): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1212553675050844220) (4 messages): 

- **Struggle Against Verbose JSON Responses**: User `@dbreunig` mentioned the frequent need to clean up **noisy json responses** but did not elaborate on the specific methods or function used.
- **Tackling Claude's Introductory Phrases**: User `@justinpinkney` shared a tip on how to avoid intro sentences like "Sure here's a..." from Claude by using the initial characters control, referencing [Anthropic's documentation](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites). They suggested starting with `<rewrite>` or enforcing the response to initiate with `{`.
- **Claude's Tenacious Explanations**: User `@derekpwillis` acknowledged trying various methods to make Claude deliver less verbose outputs, such as forcing the AI to start with `{`, yet Claude persists in providing explanations before the actual content.

**Links mentioned**:

[Ask Claude for rewrites](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites): If Claude gives a response that is close to, but not quite what you&#x27;re looking for, you can ask Claude to rewrite it. In Slack this can be as simple as telling Claude to &quot;Try again&quot; aft...

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be
  

---


### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1192042724480794685/1212786676825329695) (1 messages): 

- **Recruitment Inquiry in DMs**: User `.papahh` reached out to `@1117586410774470818` with a direct message, hinting at a potential job opportunity and expressing interest in the recipient's participation.
  

---



### Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1212532669376630824) (1 messages): 

- **Exploring the Roots of Cross-Species Values**: `@taodoggy` is seeking collaborators for a project aiming to **understand the biological and evolutionary origins of values shared across species**, refine the definition of values, and analyze how these are expressed in various cultures. They provided a brief overview with a [Google Docs link](https://docs.google.com/document/d/1A2ZdM1IBv0_5nN1pujyCvmoCGepETmWFRPmAmdjkqqA/edit?usp=drivesdk).

**Links mentioned**:

[Uncovering the Origins of Values: A Biology and Cognition-Based Approach for AI Alignment](https://docs.google.com/document/d/1A2ZdM1IBv0_5nN1pujyCvmoCGepETmWFRPmAmdjkqqA/edit?usp=drivesdk): no description found

  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1212713293693718598) (1 messages): 

- **AI Engineer Recruitment Advice Sought**: User `@peterg0093` is looking to start recruiting AI engineers in the UK and requests examples of good job descriptions to avoid deviating from any standard language in the field. He encourages users to reach out if they have useful references or resources.
  
