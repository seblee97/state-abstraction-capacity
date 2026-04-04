# Action-Based Binary Tree Navigation

## Overview

The evaluation system uses **action-based navigation** where models respond with structural commands (UP/LEFT/RIGHT) rather than specific node IDs.

## Action Types

### UP
- **Moves to:** Parent node
- **Valid when:** Not at root
- **Invalid when:** At root (index 0)
- **Example:** From Node 14 (left child of root) → UP → Node 9 (root)

### LEFT
- **Moves to:** Left child node
- **Valid when:** Node has a left child
- **Invalid when:** Node is a leaf or has no left child
- **Example:** From Node 9 (root) → LEFT → Node 14

### RIGHT
- **Moves to:** Right child node
- **Valid when:** Node has a right child
- **Invalid when:** Node is a leaf or has no right child
- **Example:** From Node 9 (root) → RIGHT → Node 13

## Implementation Details

### Tree Structure Reconstruction

The system reconstructs the binary tree structure from the adjacency list:

1. **Find Root**: Node with exactly 2 neighbors (no parent, only children)
2. **Build Index Map**: Use BFS to assign tree indices (0 for root, 2i+1 for left child, 2i+2 for right child)
3. **Action Validation**: Check tree indices to determine valid actions

### Action Parsing

Simple keyword matching:
```python
def parse_action(response_text):
    response_upper = response_text.upper()
    if "UP" in response_upper:
        return "UP"
    elif "LEFT" in response_upper:
        return "LEFT"
    elif "RIGHT" in response_upper:
        return "RIGHT"
    return None
```

## Comparison with Node ID Navigation

| Aspect | Node ID Navigation | Action Navigation |
|--------|-------------------|-------------------|
| **Response format** | "15", "move to 25" | "UP", "LEFT", "RIGHT" |
| **Action space** | N nodes | 3 actions |
| **Tests** | Graph neighbor identification | Tree structure understanding |
| **Parsing** | Extract numbers | Keyword matching |
| **Natural for** | General graphs | Binary trees |
| **Complexity** | Higher (must know specific IDs) | Lower (directions only) |

## Example Navigation Session

```
Step 1:
Current: Node 9 (root)
Available: LEFT, RIGHT
Goal: Node 11

Model: "LEFT"
→ Valid! Moved to Node 14

Step 2:
Current: Node 14
Available: UP, LEFT, RIGHT
Goal: Node 11

Model: "RIGHT"
→ Valid! Moved to Node 15

Step 3:
Current: Node 15 (leaf)
Available: UP
Goal: Node 11

Model: "LEFT"
→ Invalid! LEFT not available. Staying at Node 15.

Step 4:
Current: Node 15 (leaf)
Available: UP
Goal: Node 11

Model: "UP"
→ Valid! Moved to Node 14

Step 5:
Current: Node 14
Available: UP, LEFT, RIGHT
Goal: Node 11

Model: "LEFT"
→ Valid! Moved to Node 11

Success! Reached target in 5 steps (including 1 invalid action).
```

## Error Handling

### Invalid Action Recovery

When a model attempts an invalid action:
1. **Stay at current node** - position doesn't change
2. **Provide feedback** - explain why action is invalid
3. **List valid actions** - show what's available
4. **Continue** - model gets another chance (counts as 1 step)

Example feedback:
```
Invalid action: LEFT is not available from Node 15.
Available actions: UP.
You remain at Node 15.
```

### Failure Modes

**Task fails only when:**
1. **Unparseable response** - can't extract UP/LEFT/RIGHT
2. **Max steps exceeded** - took more than 20 steps
3. **Model error** - exception during generation

**Task does NOT fail when:**
- Invalid action attempted (recoverable)
- Model takes sub-optimal path (still exploring)

## Testing

Run the test suite:
```bash
python test_action_navigation.py
```

Tests verify:
- ✅ Root has LEFT/RIGHT but not UP
- ✅ Internal nodes have UP/LEFT/RIGHT
- ✅ Leaves have only UP
- ✅ UP returns to parent correctly
- ✅ LEFT/RIGHT reach correct children
- ✅ Navigation paths work correctly

## Advantages

### 1. **Simpler for Models**
- Only need to understand 3 actions
- Don't need to remember specific node IDs
- More robust to different naming conventions

### 2. **Tests Structural Understanding**
- Models must understand parent/child relationships
- Tests if representations convey tree structure
- Shows if models grasp binary tree concept

### 3. **Robust Parsing**
- Just keyword matching (UP/LEFT/RIGHT)
- Works with various response formats:
  - "LEFT"
  - "I'll go LEFT"
  - "Move LEFT to the left child"
  - "Action: LEFT"

### 4. **Natural Task Framing**
- Matches how humans think about tree navigation
- Clearer instructions for models
- More intuitive than node ID memorization

## Limitations

### 1. **Binary Tree Specific**
- Only works for binary trees
- Not generalizable to arbitrary graphs
- Assumes exactly 2 children per internal node

### 2. **Requires Tree Reconstruction**
- Must infer tree structure from adjacency list
- Depends on correct root identification
- Assumes tree is properly formed

### 3. **Less Challenging**
- Simpler than arbitrary graph navigation
- Doesn't test spatial reasoning with node IDs
- May be "too easy" for some models

## Future Extensions

### Multi-way Trees
- Add actions: CHILD_1, CHILD_2, ..., CHILD_N
- Support non-binary trees
- Test on trees with variable branching factors

### Directed Actions
- Add: PARENT, ANCESTOR, DESCENDANT
- Test more complex navigation primitives
- Evaluate deeper structural understanding

### Hybrid Approach
- Combine actions with node IDs
- "LEFT to Node 14"
- Tests both structural and spatial reasoning

## Related Files

- [evaluate_models.py](evaluate_models.py) - Main evaluation with action parsing
- [test_action_navigation.py](test_action_navigation.py) - Test suite
- [generate_tree.py](generate_tree.py) - Updated task descriptions
- [README.md](README.md) - User-facing documentation
- [CHANGELOG.md](CHANGELOG.md) - Change history
