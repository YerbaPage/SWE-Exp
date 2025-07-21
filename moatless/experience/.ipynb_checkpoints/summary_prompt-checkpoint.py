# 这个是从issue里面提取的
summarize_prompt = '''
Your task is to summarize a human-reported GitHub issue in natural language.
Your output should strictly follow the format below.
1. ...
2. ...

DO NOT SPEAK ANY REDUNDANT WORDS (like ’json’, ’output’, etc.)
An example is given below:
{example}
Below is the issue for you to summarize:
Repository: 
{repo}

Issue:
{description}

Output:
'''

example1 = '''
Repository:
django

Problem statement:
Form Field’s __deepcopy__ does not (deep)copy the error messages.
Description
	
The __deepcopy__ method defined for the formfields (​https://github.com/django/django/blob/146086f219d01dbb1cd8c089b5a5667e396e1cc4/django/forms/fields.py#L200) performs a shallow copy of self and does not include additional treatment for the error_messages dictionary. As a result, all copies of the same field share the same dictionary and any modification of either the dictionary or the error message itself for one formfield is immediately reflected on all other formfiels.
This is relevant for Forms and ModelForms that modify the error messages of their fields dynamically: while each instance of the specific form (e.g., ProfileForm) is expected to have a set of fields “sealed” away from other instances of the same ProfileForm (​https://github.com/django/django/blob/146086f219d01dbb1cd8c089b5a5667e396e1cc4/django/forms/forms.py#L95), in fact all these instances share the same error messages, resulting in incorrectly raised errors.
Confirmed for versions of Django going back to 1.11.


Output:
1. The deepcopy method in Django form fields only performs a shallow copy, so the error_messages dictionary is shared across all copies.
2. This causes changes to error messages in one field to affect all instances, leading to incorrect behavior in dynamic forms.

'''

example2 = '''
Repository:
astropy

Problem statement:
Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
       [False,  True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True, False],
       [False, False, False,  True]])
```

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True,  True],
       [False, False,  True,  True]])
```
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?


Output:
1. The issue reports that the separability_matrix function incorrectly computes separability when dealing with nested compound models.
2. In nested models, such as combining Pix2Sky_TAN with a compound model, the expected diagonal separability is lost, leading to shared inputs and outputs, which appears to be a bug.
'''



# for planner

planner_prompt = '''
**Aim**: Generate a step-by-step solution for fixing a software issue.

**Instructions**:

1. **Problem Analysis**: Clearly describe the bug.
2. **Code Review**: Identify the relevant code files, classes, and methods involved in the bug.
    - **Step**: Examine the code context, including surrounding functions or data structures.
3. **Root Cause**: Determine the fundamental cause of the bug (e.g., logical error, missing validation).
4. **Solution Design**: Propose a solution, such as modifying existing code or adding new logic.
    - **Step**: Provide a detailed explanation of the changes needed.
5. **Implementation**: Modify the code accordingly and ensure compatibility with other components.
    - **Step**: Include the updated code snippets and any configuration changes.

Please only output the list, do not output any other text. Your output should follow the format below:

1. ...
2. ...
'''
