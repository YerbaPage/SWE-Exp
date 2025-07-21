<div align="center">
  <img src="assets/SWE-Exp3.png" alt="SWE-Exp Logo" width="200"/>
</div>

# SWE-Exp
SWE-Exp: Experience-Driven Software Issue Resolution

A software engineering experimental framework based on Large Language Models (LLMs) for automated code repair and optimization, featuring experience learning and transfer capabilities.

## 🎯 Core: Experience Learning System

The `moatless/experience` module implements a sophisticated three-stage experience-driven approach to software issue resolution.

## 📋 Stage 1: Issue Type Extraction

### Issue Type Generator (`exp_agent/extract_verified_issue_types_batch.py`)
Automatically categorizes software issues to enable effective experience matching.

**Key Features:**
- **Intelligent Classification**: Analyzes problem statements to extract issue types
- **Batch Processing**: Handles large datasets efficiently with resume capability
- **Structured Output**: Generates categorized issue type mappings

```python
from moatless.experience.exp_agent.extract_verified_issue_types_batch import IssueAgent
from moatless.experience.prompts.exp_prompts import (
    issue_type_system_prompt,
    issue_type_user_prompt
)
from moatless.completion.completion import CompletionModel

# Initialize completion model
completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)

# Initialize issue type extractor
issue_agent = IssueAgent(
    completion=completion_model,
    system_prompt=issue_type_system_prompt,
    user_prompt=issue_type_user_prompt
)

# Extract issue types for instances
# python extract_verified_issue_types_batch.py --start 0 --end 100
```

**Workflow:**
1. **Problem Analysis**: Parse issue descriptions and code context
2. **Type Classification**: Generate structured issue type categories
3. **Batch Storage**: Save results in JSON format for efficient lookup
4. **Merge Processing**: Combine batch results into unified dataset

## 📊 Stage 2: Experience Generation

### Experience Generator (`exp_agent/exp_agent.py`)
Transforms past successful/failed resolution attempts into reusable experience knowledge.

**Key Features:**
- **Trajectory Processing**: Analyzes historical resolution paths from search trees
- **Dual Extraction**: Separates perspective insights and modification patterns
- **Success/Failure Analysis**: Different prompts for successful vs. failed attempts
- **Trajectory Selection criteria**: The successful trajectory is the shortest, and the failed trajectory is the longest.
- **Structured Experience**: Converts trajectories into structured knowledge base

```python
from moatless.experience.exp_agent.exp_agent import ExpAgent
from moatless.experience.prompts.exp_prompts import (
    encode_success_perspective_system_prompt,
    encode_failed_perspective_system_prompt,
    encode_success_modify_system_prompt
)
from moatless.completion.completion import CompletionModel

# Initialize completion model
completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)

# Initialize experience generator
exp_agent = ExpAgent(
    completion=completion_model,
    success_per_system_prompt=encode_success_perspective_system_prompt,
    failed_per_system_prompt=encode_failed_perspective_system_prompt,
    success_mod_system_prompt=encode_success_modify_system_prompt,
    issue_type_path='<PROJECT_ROOT>/tmp/verified_issue_types_final.json'
)

# Process instance to extract experiences
exp_tree = {}
instance_id = "<INSTANCE_ID>"
tree_path = "<TRAJECTORY_PATH>/trajectory.json"
tree = SearchTree.from_file(tree_path)

# Extract perspective and modification experiences
rollout, patch = get_success_rollout_with_patch(tree, eval, False)
trajectory = get_trajectory(rollout)
perspective_exp = exp_agent.encode_perspective(instance_id, rollout=trajectory[0], patch=patch, flag='success')

rollout, patch = get_success_rollout_with_patch(tree, eval, True)
trajectory = get_trajectory(rollout)
modify_exp = exp_agent.encode_modify(instance_id, rollout=trajectory[0], patch=patch)
```

**Generation Process:**
- **Trajectory Loading**: Load search tree trajectories from resolution attempts
- **Success/Failure Classification**: Determine outcome and select appropriate prompts
- **Pattern Extraction**: Extract both perspective insights and modification strategies
- **Experience Structuring**: Format into standardized experience objects
- **Knowledge Base Update**: Add new experiences to searchable repository

## 🔍 Stage 3: Experience Reuse

### Two-Phase Experience Selection

#### Phase 1: Similarity-Based Filtering
**Automated Pre-filtering using Issue Type + Description Similarity**

```python
from moatless.experience.exp_agent.select_agent import SelectAgent
from moatless.experience.prompts.exp_prompts import (
    select_exp_system_prompt,
    select_exp_user_prompt
)

# Initialize experience selector
select_agent = SelectAgent(
    completion=completion_model,
    instance_id="<INSTANCE_ID>",
    select_system_prompt=select_exp_system_prompt,
    user_prompt=select_exp_user_prompt,
    exp_path='<PROJECT_ROOT>/tmp/verified_experience_tree.json',
    train_issue_type_path='<PROJECT_ROOT>/tmp/verified_issue_types_final.json',
    test_issue_type_path='<PROJECT_ROOT>/tmp/verified_issue_types_final.json',
    persist_dir=''
)

# Phase 1: Automatic similarity-based filtering (internal method)
# Phase 2: Agent evaluation and final selection
old_experiences = select_agent.select_workflow(n=5)
```

**Similarity Matching Logic:**
- **Issue Type Alignment**: Match categorical problem types
- **Semantic Similarity**: Use `multilingual-e5-large-instruct` embeddings
- **Cosine Distance**: Calculate description similarity scores
- **Top-K Selection**: Return 10 most relevant candidates

#### Phase 2: Agent-Based Evaluation
**Intelligent Final Selection by LLM Agent**

```python
# The select_workflow method internally handles both phases:
# 1. Similarity-based filtering using issue types and embeddings
# 2. LLM agent evaluation for final selection

# Generate generalized experiences for current context
new_experiences = select_agent.generalize_workflow(
    old_experiences=old_experiences,
    type='perspective',
    history=None,
    cur_code=None,
    instruction=None
)
```

**Agent Evaluation Process:**
- **Context Analysis**: Deep understanding of current problem
- **Experience Assessment**: Evaluate relevance and applicability
- **Strategy Selection**: Choose most promising resolution approaches
- **Confidence Scoring**: Rank experiences by expected effectiveness

### Complete Experience Selection Pipeline

```python
# Full pipeline: Issue Type → Similarity → Agent Selection
experiences = select_agent.select_workflow(n=3)
```

## 🔧 Main Workflow (`workflow.py`)

The main execution pipeline that orchestrates the entire system:

### Workflow Steps

1. **Environment Setup**
   ```python
   # Load instance and create repository
   instance = get_moatless_instance(instance_id=instance_id)
   repository = create_repository(instance)
   ```

2. **Agent Initialization**
   ```python
   # Configure multi-agent system
   agent = ActionAgent(...)
   discriminator = AgentDiscriminator(n_agents=5, n_rounds=3)
   instructor = Instructor(...)
   ```

3. **Experience Integration**
   ```python
   # Two-phase experience selection
    select_agent = SelectAgent(completion=completion_model, instance_id=instance_id,
                            select_system_prompt=select_exp_system_prompt,
                            user_prompt=select_exp_user_prompt, 
                            exp_path='.json', 
                            train_issue_type_path='.json', 
                            test_issue_type_path='.json', 
                            persist_dir=experience_path)
    old_experiences = select_agent.select_workflow(n=1)
   ```

4. **Search Tree Execution**
   ```python
   # Run intelligent search with experience guidance
    search_tree = SearchTree.create(
        message=instance["problem_statement"],
        assistant=agent,
        instructor=instructor,
        file_context=file_context,
        value_function=value_function,
        discriminator=discriminator,
        feedback_generator=feedback_generator,
        max_finished_nodes=max_finish_nodes,
        max_iterations=max_iterations,
        max_expansions=max_expansions,
        max_depth=max_depth,
        persist_path=persist_path,
    )
   finished_node = search_tree.run_search(select_agent, old_experiences)
   ```

### Usage

```bash
# Step 1: Extract issue types
python moatless/experience/exp_agent/extract_verified_issue_types_batch.py \
  --start 0 --end 500

# Step 2: Generate experiences from trajectories
# Run the main script in exp_agent.py (modify paths as needed)
python moatless/experience/exp_agent/exp_agent.py

# Step 3: Run experience-driven resolution
python workflow.py --instance_ids instances.txt --max_iterations 10
```

## 🏗️ Project Structure

```
SWE-Exp/
├── workflow.py                           # Main execution pipeline
├── moatless/
│   ├── experience/                      # 🎯 Experience learning core
│   │   ├── exp_agent/                  # Experience processing agents
│   │   │   ├── extract_verified_issue_types_batch.py  # Issue classification
│   │   │   ├── select_agent.py         # Two-phase experience selection
│   │   │   └── exp_agent.py            # Experience generation from trajectories
│   │   ├── prompts/                    # Instruction templates
│   │   │   ├── exp_prompts.py          # Experience selection prompts
│   │   │   └── agent_prompts.py        # Agent interaction prompts
│   │   ├── instructor.py               # Guidance generation
│   │   └── get_save_json.py            # Data persistence utilities
│   ├── actions/                        # Code operation primitives
│   ├── agent/                          # Multi-agent framework
│   ├── search_tree.py                  # Tree search optimization
│   └── ...                             # Other modules
└── verified_dataset_ids.txt             # Verified test instances
```

## 🙏 Acknowledgements

We would like to thank Albert Örwall for open-sourcing SWE-Search, which serves as the foundation for our framework, SWE-Exp. This framework is built upon and references the excellent work at [@aorwall/moatless-tools](https://github.com/aorwall/moatless-tools/tree/main).