#!/bin/bash
# One-liner to tag AI news issues using chisel CLI
# Usage: ./chisel-tag.sh [file] or defaults to latest issue

f="${1:-$(ls -t src/content/issues/*.md | head -n 1)}"
core=$(awk '/^---$/{c++;next} c==1 && /^(description|companies|models|topics|people):/{exit} c==1' "$f")
body=$(awk '/^---$/{c++;next}c>=2' "$f")
tags=$(chisel --model sonnet -p "Extract AI news tags. CRITICAL: Output ONLY raw YAML (no markdown code fences, no backticks, no \`\`\`yaml, no preamble text). Start directly with 'description: >-':
description: >-
  1-3 sentences with **bold** for companies/models/facts, *italics* for quotes. Ignore '> AI News for...' header.
companies:
  - lowercase-hyphenated (prefer: openai, anthropic, google-deepmind, meta-ai-fair, x-ai, deepseek, together-ai, figure-ai, langchain-ai)
models:
  - with-versions (prefer: gpt-4o, claude-3-opus, llama-3-70b, gemini-1.5-pro, deepseek-v3)
topics:
  - specific-terms (prefer: fine-tuning, reasoning, agents, multimodality, benchmarking, long-context, function-calling, vision)
people:
  - twitter-handles (prefer: sama, karpathy, ylecun, swyx, gdb)

Content:
$(printf '%.5000s' "$body")" | sed -E '/^```/d' | sed '/^```yaml$/d' | sed '/^```yml$/d')
{ echo '---'; echo "$core"; echo "$tags"; echo '---'; echo; echo "$body"; } > "$f"

