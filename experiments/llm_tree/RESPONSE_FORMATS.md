# Response Format Strategies for LLM Evaluation

This document outlines different strategies for ensuring parseable, automated evaluation of LLM responses in navigation tasks.

## Current Implementation: Direct Node ID

**Approach:** The model responds with the node ID it wants to move to.

**Prompt format:**
```
You are currently at Node 15. Your goal is to reach Node 31.
Which node do you want to move to? You can move to: 7, 30, 31
Respond with ONLY the node number you want to move to.
Next node:
```

**Expected response:**
```
31
```

**Advantages:**
- Tests true graph understanding (not just procedural tree rules)
- Simple to parse (extract first number)
- Explicit invalid moves (attempting non-connected edges)
- Natural for models to understand
- Works with any graph structure, not just binary trees

**Parsing strategy:**
```python
def parse_node_id(response):
    patterns = [
        r'(?:move to node|to node|node)\s*[:\-]?\s*(\d+)',
        r'(?:move to|to)\s+(\d+)',
        r'^(?:node\s+)?(\d+)$',
        r'\b(\d+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None
```

**Failure modes:**
- Model cannot parse response → counted as failure (task ends)
- Model responds with node ID not in neighbor list → stays at current node, receives feedback, can continue
- Max steps exceeded → counted as failure

---

## Alternative Strategies

### 1. Structured Format with Keywords

**Prompt:**
```
Respond in this exact format:
ACTION: MOVE_UP / MOVE_LEFT / MOVE_RIGHT
```

**Expected response:**
```
ACTION: MOVE_LEFT
```

**Advantages:**
- Clear structure
- Easy to parse with regex

**Disadvantages:**
- Assumes binary tree structure (not general graph)
- Models may not follow exact format
- Doesn't test graph understanding

---

### 2. Multiple Choice

**Prompt:**
```
Choose one option:
A) Move to Node 7
B) Move to Node 30
C) Move to Node 31

Respond with only the letter (A, B, or C).
```

**Expected response:**
```
B
```

**Advantages:**
- Very constrained output space
- Easy to parse
- Models often good at multiple choice

**Disadvantages:**
- Limited to small number of neighbors
- Extra verbosity in prompt
- Doesn't scale to nodes with many neighbors

---

### 3. JSON Format

**Prompt:**
```
Respond with a JSON object:
{"next_node": <node_id>, "reasoning": "<optional>"}
```

**Expected response:**
```json
{"next_node": 30, "reasoning": "Moving closer to target"}
```

**Advantages:**
- Structured, parseable format
- Can include reasoning for analysis
- Industry-standard format

**Disadvantages:**
- Models may not generate valid JSON
- More verbose
- Requires JSON parsing (can fail)
- Some models struggle with exact formatting

---

### 4. Few-Shot Prompting

**Approach:** Include examples in the prompt

**Prompt:**
```
Example 1:
Current: Node 5. Can move to: 2, 10, 11
Response: 10

Example 2:
Current: Node 10. Can move to: 5, 20, 21
Response: 20

Now your turn:
Current: Node 15. Can move to: 7, 30, 31
Response:
```

**Advantages:**
- Shows exact expected format
- Often improves model performance
- Can combine with any strategy above

**Disadvantages:**
- Longer prompts
- May bias model behavior
- Takes up context window

---

### 5. Constrained Decoding

**Approach:** Use logit bias or grammar constraints to force valid outputs

**Implementation:**
```python
# Force model to only generate node IDs from valid neighbors
allowed_tokens = [tokenizer.encode(str(n))[0] for n in valid_neighbors]
outputs = model.generate(
    inputs,
    force_tokens=allowed_tokens  # Pseudocode
)
```

**Advantages:**
- Guarantees valid output
- No parsing errors
- 100% parseable

**Disadvantages:**
- Requires framework support
- More complex implementation
- May not be available for all models
- Doesn't test model's ability to follow instructions

---

### 6. System Prompt (Instruction-Tuned Models)

**Approach:** Use system message for format specification

**Implementation:**
```python
messages = [
    {"role": "system", "content": "You are a graph navigation agent. Always respond with exactly one number: the node ID you want to move to. No explanation."},
    {"role": "user", "content": "Current: Node 15. Can move to: 7, 30, 31"}
]
```

**Advantages:**
- Separates task from format
- Works well with chat models
- Can reinforce throughout conversation

**Disadvantages:**
- Only works with instruction-tuned models
- Requires chat template support
- May still not be followed

---

## Comparison Table

| Strategy | Parseable | Tests Understanding | Scalable | Simple | Robust |
|----------|-----------|-------------------|----------|--------|--------|
| **Direct Node ID** (current) | ✅ High | ✅ Yes | ✅ Yes | ✅ Yes | ✅ High |
| Structured Keywords | ✅ High | ❌ No | ❌ Limited | ✅ Yes | ⚠️ Medium |
| Multiple Choice | ✅ Very High | ⚠️ Partial | ❌ No | ⚠️ Medium | ✅ High |
| JSON Format | ⚠️ Medium | ✅ Yes | ✅ Yes | ❌ No | ❌ Low |
| Few-Shot | ⚠️ Depends | ✅ Yes | ✅ Yes | ⚠️ Medium | ⚠️ Medium |
| Constrained Decoding | ✅ Perfect | ❌ No | ✅ Yes | ❌ No | ✅ Very High |
| System Prompt | ⚠️ Depends | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Medium |

---

## Recommendations

### For General Graph Understanding (Current)
✅ **Use Direct Node ID** - Best balance of simplicity, robustness, and testing true graph understanding.

### For Maximum Parseability
Consider **Multiple Choice** for smaller graphs or **Constrained Decoding** if framework supports it.

### For Detailed Analysis
Add **Few-Shot examples** to current approach to improve consistency.

### For Production Systems
Consider **System Prompt + JSON** for chat models with error handling and retries.

---

## Implementation Notes

The current implementation uses **Direct Node ID with robust parsing**:

1. **Multiple regex patterns** to catch variations
2. **Fallback parsing** for natural language responses
3. **Explicit validation** against neighbor list
4. **Clear failure categorization**:
   - `Could not parse node ID` - parsing failure
   - `Invalid move: Node X not connected` - wrong edge
   - `Target reached` - success
   - `Max steps exceeded` - timeout

This approach provides the best signal for evaluating whether models truly understand graph structure vs. just following procedural rules.

### Error Recovery

The system now allows models to recover from invalid moves:
- **Invalid edge attempt:** Model stays at current node and receives feedback
- **Feedback message:** "Invalid move: Node X is not connected to Node Y. You remain at Node Y."
- **Can continue:** Model gets another chance to make a valid move
- **Tests learning:** Can the model use feedback to correct its understanding?

This makes evaluation more realistic - agents should be able to recover from mistakes rather than failing immediately.
