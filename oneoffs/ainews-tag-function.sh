#!/bin/bash
# Bash function version - add this to your ~/.zshrc or ~/.bashrc
# Usage: ainews-tag [file] or defaults to latest issue

ainews-tag() {
  local f="${1:-$(ls -t src/content/issues/*.md | head -n 1)}"
  local core body tags
  
  # Extract core frontmatter (id, title, date)
  core=$(awk '/^---$/{c++;next} c==1 && /^(description|companies|models|topics|people):/{exit} c==1' "$f")
  
  # Extract body content (after second ---)
  body=$(awk '/^---$/{c++;next}c>=2' "$f")
  
  # Generate tags using chisel
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
$(printf '%.5000s' "$body")" | sed '/^```/d' | sed '/^```yaml$/d' | sed '/^```yml$/d')
  
  # Reconstruct file
  { echo '---'; echo "$core"; echo "$tags"; echo '---'; echo; echo "$body"; } > "$f"
  
  echo "Tagged: $f"
}

# If sourced directly, run it
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ainews-tag "$@"
fi

