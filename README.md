<div align="center">
  <img src="assets/SWE-Exp3.png" alt="SWE-Exp Logo" width="300"/>
</div>

# SWE-Exp
SWE-Exp: Experience-Driven Software Issue Resolution

A software engineering experimental framework based on Large Language Models (LLMs) for automated code repair and optimization, featuring experience learning and transfer capabilities.

## 📚 Table of Contents

<div align="center">
  
| Section | Description |
|---------|-------------|
| [🎯 Core: Experience Learning System](#-core-experience-learning-system) | Overview of the four-stage experience framework |
| [📋 Stage 1: Issue Type Extraction](#-stage-1-issue-type-extraction) | Automatic categorization of software issues |
| [📊 Stage 2: Experience Generation](#-stage-2-experience-generation) | Transform trajectories into reusable knowledge |
| [🔍 Stage 3: Experience Reuse](#-stage-3-experience-reuse) | Two-phase experience selection and application |
| [🔧 Main Workflow](#-main-workflow-workflowpy) | Complete execution pipeline and usage |
| [🏗️ Project Structure](#️-project-structure) | Repository organization and modules |
| [📋 Requirements](#-requirements) | Dependencies and environment setup |
| [🙏 Acknowledgements](#-acknowledgements) | Credits and references |

</div>

<div align="center">
  <img src="assets/method.png" alt="SWE-Exp Method Overview" width="800"/>
  <p><em>Figure 1: SWE-Exp Method Overview - Four-Stage Experience-Driven Framework</em></p>
</div>

## 🎯 Core: Experience Learning System

The `moatless/experience` module implements a sophisticated four-stage experience-driven approach to software issue resolution.

Before using experiences, you first need to use this framework without experience to generate some trajectories.

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
old_experiences = select_agent.select_workflow(n=1)
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
- **Strategy Selection**: Choose most promising comprehension experiences
- **Confidence Scoring**: Rank experiences by expected effectiveness


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

### Search Tree Four-Stage Process

The search tree implements a sophisticated four-stage iterative process for problem resolution:

1. **🎯 Selection Stage**
   - **Node Selection**: Choose the most promising leaf node for expansion
   - **Strategy**: Uses value function scores and exploration strategies
   - **Criteria**: Balance between exploitation of high-value paths and exploration of new possibilities

2. **🔄 Expansion Stage**
   - **Action Generation**: Generate possible actions using the ActionAgent
   - **Experience Integration**: Incorporate relevant past experiences into action planning
   - **Instruction Guidance**: Use Instructor to provide contextual guidance for action selection

3. **📊 Evaluation Stage**
   - **Value Assessment**: Evaluate the quality of generated actions using ValueFunction
   - **Feedback Integration**: Incorporate feedback from FeedbackGenerator
   - **Scoring**: Assign scores to determine action viability and success probability

4. **🏆 Discrimination Stage**
   - **Multi-Agent Consensus**: Use multiple agents to evaluate and rank solutions
   - **Trajectory Selection**: Choose the best resolution path among alternatives
   - **Quality Assurance**: Ensure selected solutions meet quality criteria through collaborative judgment

### Usage

#### Complete Pipeline Execution

```bash
# Step 1: Extract issue types from problem statements
python moatless/experience/exp_agent/extract_verified_issue_types_batch.py \
  --start 0 --end 500

# Step 2: Generate experiences from historical trajectories
# Run the main script in exp_agent.py (modify paths and your trajectory as needed)
# The input must be in the SearchTree format
python moatless/experience/exp_agent/exp_agent.py

# Step 3: Run experience-driven resolution workflow
python workflow.py --instance_ids instance_id.txt --max_iterations 20
```

#### Main Workflow Execution

The primary entry point for running SWE-Exp is through `workflow.py`:

```bash
# Basic usage with instance ID file (without experience)
python workflow.py --instance_ids instance_id.txt

# Enable experience-driven resolution
python workflow.py --instance_ids instance_id.txt --experience

# Advanced usage with custom parameters
python workflow.py \
  --instance_ids instance_id.txt \
  --max_iterations 20 \
  --max_finished_nodes 3 \
  --max_expansions 3 \
  --experience
```

**Parameters:**
- `--instance_ids`: Path to text file containing instance IDs (one per line)
- `--max_iterations`: Maximum number of search tree iterations (default: 10)
- `--max_finished_nodes`: Maximum number of completed solution nodes (default: 3)  
- `--max_expansions`: Maximum number of expansions per state (default: 3)
- `--experience`: Enable experience-driven resolution (flag to activate experience learning and transfer)
  - **Without `--experience`**: Uses standard search tree resolution without historical experience guidance
  - **With `--experience`**: Activates the three-stage experience system for enhanced problem-solving

**Input Format:**
Create an `instance_id.txt` file with one instance ID per line:
```
django__django-12345
flask__flask-6789
requests__requests-1011
```

**Output:**
- **Trajectory Files**: Saved to `tmp_verified/trajectory/{instance_id}/`
- **Experience Files**: Saved to `tmp_verified/experience/{instance_id}/`
- **Prediction Results**: Appended to `prediction_verified.jsonl`

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