# Changelog

## 2026-01-14 - Changed to Binary Tree Actions (UP/LEFT/RIGHT)

### Changed: Action Format

**Previous behavior:**
- Models responded with direct node IDs (e.g., "15", "move to 25")
- Tested graph understanding by requiring models to specify exact neighbors

**New behavior:**
- Models respond with binary tree actions: **UP**, **LEFT**, or **RIGHT**
- UP: Move to parent node (invalid at root)
- LEFT: Move to left child (invalid at leaf)
- RIGHT: Move to right child (invalid at leaf)

### Rationale

1. **Tests structural understanding**
   - Models must understand binary tree relationships
   - Simpler action space (3 actions vs N node IDs)

2. **More natural for tree navigation**
   - UP/LEFT/RIGHT are intuitive directions
   - Matches how trees are conceptually navigated

3. **Tests different skill**
   - Previous: Can model identify valid neighbors from graph?
   - Current: Can model follow directional rules in tree structure?

4. **Clearer success/failure modes**
   - Invalid action: tried LEFT at leaf, tried UP at root
   - Success: correctly followed tree structure to target

### Example

**Before (Node ID format):**
```
Current: Node 15, Neighbors: [7, 30, 31]
Response: "31"
→ Moved to 31
```

**After (Action format):**
```
Current: Node 15
Available: UP, LEFT, RIGHT
Response: "RIGHT"
→ Took action RIGHT, moved to Node 11
```

### Impact on Evaluation

- Tests binary tree navigation ability
- Simpler parsing (just 3 keywords: UP/LEFT/RIGHT)
- More robust to different response formats
- Better tests if models understand tree structure from different representations

### Files Changed

- [evaluate_models.py](llm_tree/evaluate_models.py): Changed to parse UP/LEFT/RIGHT actions
- [generate_tree.py](llm_tree/generate_tree.py): Updated task description
- [test_action_navigation.py](llm_tree/test_action_navigation.py): New test for action-based navigation
- [README.md](llm_tree/README.md): Updated with action-based examples

---

## 2026-01-14 - Invalid Move Recovery (Now applies to actions)

### Changed: Invalid Edge Handling

**Previous behavior:**
- Attempting to move to a non-connected node immediately failed the entire task
- Task ended with reason: "Invalid move: Node X is not connected to Node Y"
- No opportunity for the model to recover

**New behavior:**
- Attempting to move to a non-connected node:
  - Agent stays at current node
  - Receives feedback: "Invalid move: Node X is not connected to Node Y. You remain at Node Y."
  - Can continue making moves (counts as 1 step)
  - Task only fails if max steps exceeded or unparseable response

### Rationale

1. **More realistic evaluation**
   - Real agents should handle errors gracefully
   - Immediate failure is too harsh for exploratory behavior

2. **Tests error correction**
   - Can models learn from feedback?
   - Do models adjust their strategy after invalid moves?

3. **Better signal**
   - Distinguishes between:
     - Models that can't parse the graph at all
     - Models that make occasional mistakes but recover
     - Models that perfectly understand the structure

4. **Fairer comparison**
   - Some representations may be harder to parse correctly
   - Recovery ability is a valuable skill to measure

### Example

**Before:**
```
Step 1: Current=15, Attempted=25, Valid neighbors=[7,30,31]
→ FAILURE: "Invalid move: Node 25 is not connected to Node 15"
Result: Failed after 1 step
```

**After:**
```
Step 1: Current=15, Attempted=25, Valid neighbors=[7,30,31]
→ Invalid move feedback, remain at 15

Step 2: Current=15, Attempted=31, Valid neighbors=[7,30,31]
→ Moved to 31

Step 3: Current=31 == Target
Result: Success after 3 steps (including 1 invalid move)
```

### Impact on Metrics

This change will likely:
- **Increase success rates** - Models that were close but made mistakes can now recover
- **Increase average step counts** - Invalid moves still consume steps
- **Better differentiate models** - Recovery ability becomes a measurable dimension

### Files Changed

- [evaluate_models.py](llm_tree/evaluate_models.py): Changed invalid move handling from return failure to continue with feedback
- [README.md](llm_tree/README.md): Updated navigation task description with recovery examples
- [RESPONSE_FORMATS.md](llm_tree/RESPONSE_FORMATS.md): Added error recovery section
- [test_invalid_move_recovery.py](llm_tree/test_invalid_move_recovery.py): New test to verify recovery behavior

### Testing

Run the test to verify behavior:
```bash
python test_invalid_move_recovery.py
```

Expected output:
```
✅ Invalid moves are detected
✅ Agent stays at current node
✅ Agent can continue with valid moves
```

### Backward Compatibility

This is a **breaking change** for evaluation results:
- Previous evaluations counted invalid moves as immediate failures
- New evaluations allow recovery, changing success rates
- Results from before/after this change are not directly comparable

To compare with old behavior, you would need to:
1. Check history for any `valid: False` entries
2. Count those as immediate failures in post-processing

---

## Earlier Changes

### 2026-01-14 - Refactored to Remove Duplicate Code

- Removed tree generation logic from `evaluate_models.py`
- Now reads graph structure from `4_adjacency_list.txt`
- Simplified code, single source of truth
- See [REFACTORING_SUMMARY.md](llm_tree/REFACTORING_SUMMARY.md) for details

### 2026-01-14 - Changed to Node ID Actions

- Changed from `MOVE_UP/LEFT/RIGHT` to direct node ID responses
- Tests true graph understanding vs. procedural navigation
- See [RESPONSE_FORMATS.md](llm_tree/RESPONSE_FORMATS.md) for details

### Initial Version

- Created tree generation with multiple representations
- Basic evaluation framework for LLMs
- Binary tree navigation task
