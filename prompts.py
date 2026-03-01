
def build_agent_message_sequential_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if role == "planner":
        user_prompt = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""
    
    elif role == "critic":
        user_prompt = f"""
Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""
    
    elif role == "refiner":
        user_prompt = f"""
Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""
    
    elif role == "judger":
        if args.task in ['gsm8k', 'aime2024', 'aime2025']:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        
        elif args.task in ["arc_easy", "arc_challenge", "gpqa", 'medqa']:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif args.task in ["mbppplus", "humanevalplus"]:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block.

Now, reason step by step and output the final answer inside ```python
YOUR_PYTHON_CODE
```.
"""

        elif args.task in ["winogrande"]:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        else: 
            raise NotImplementedError(f"Task {args.task} not implemented in v5 judger prompt.")
        
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_agent_message_hierarchical_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if args.task in ['gsm8k', 'aime2024', 'aime2025']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:

        if args.task == "medqa":

            if role == "planner":
                user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""
            elif role == "critic":
                user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}     

Your response:
"""
            elif role == "refiner":
                user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:       
"""
            elif role == "judger":

                user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""

        else:
            if role == "planner":
                user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""
    
            elif role == "critic":
                user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}     

Your response:
"""
    
            elif role == "refiner":
                user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:       
"""
            elif role == "judger":

                user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 
    
Input Question: {question}

Your response:
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_sequential_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["text_mas"], "only for text_mas method"
    assert "qwen" in args.model_name.lower(), "only for qwen models"

    # truncate context if needed
    ctx = context[: args.text_mas_context_length]

    if role == "planner":
        user_content = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""

    elif role == "critic":
        user_content = f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{question}

## Plan from Planner Agent:
{ctx}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""

    elif role == "refiner":
        user_content = f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{question}

## Original Plan and Critic Feedback:
{ctx}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:
"""

    elif role == "judger":
        task = getattr(args, "task", None)

        if task in ["gsm8k", "aime2024", "aime2025"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["mbppplus", "humanevalplus"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
"""
            
        elif task in ["winogrande"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        else:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and present your final answer clearly at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    assert method in ["text_mas"], "this prompt only for text_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"
    
    if args.task in ['gsm8k', 'aime2024', 'aime2025']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":

            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer in markdown python code block.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_single_agent(question: str, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert args.method in ["baseline"], "this prompt only for baseline method (single agent)"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    task = args.task

    if task in ["gsm8k", "aime2024", "aime2025"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


# ============= Document Extraction Dataset Prompts =============

# DocRED relation definitions (all 96 Wikidata properties)
DOCRED_RELATIONS_FULL = """ALL VALID RELATIONS (use these IDs):

- P6: head of government
- P17: country
- P19: place of birth
- P20: place of death
- P22: father
- P25: mother
- P26: spouse
- P27: country of citizenship
- P30: continent
- P31: instance of
- P35: head of state
- P36: capital
- P37: official language
- P39: position held
- P40: child
- P50: author
- P54: member of sports team
- P57: director
- P58: screenwriter
- P69: educated at
- P86: composer
- P102: member of political party
- P108: employer
- P112: founded by
- P118: league
- P123: publisher
- P127: owned by
- P131: located in the administrative territorial entity
- P136: genre
- P137: operator
- P140: religion
- P150: contains administrative territorial entity
- P155: follows
- P156: followed by
- P159: headquarters location
- P161: cast member
- P162: producer
- P166: award received
- P170: creator
- P171: parent taxon
- P172: ethnic group
- P175: performer
- P176: manufacturer
- P178: developer
- P179: series
- P190: sister city
- P194: legislative body
- P205: basin country
- P206: located in or next to body of water
- P241: military branch
- P264: record label
- P272: production company
- P276: location
- P279: subclass of
- P355: subsidiary
- P361: part of
- P364: original language of work
- P400: platform
- P403: mouth of the watercourse
- P449: original network
- P463: member of
- P488: chairperson
- P495: country of origin
- P527: has part
- P551: residence
- P569: date of birth
- P570: date of death
- P571: inception
- P576: dissolved, abolished or demolished
- P577: publication date
- P580: start time
- P582: end time
- P585: point in time
- P607: conflict
- P674: characters
- P676: lyrics by
- P706: located on terrain feature
- P710: participant
- P737: influenced by
- P740: location of formation
- P749: parent organization
- P800: notable work
- P807: separated from
- P840: narrative location
- P937: work location
- P1001: applies to jurisdiction
- P1056: product or material produced
- P1198: unemployment rate
- P1336: territory claimed by
- P1344: participant of
- P1365: replaces
- P1366: replaced by
- P1376: capital of
- P1412: languages spoken, written or signed
- P1441: present in work
- P3373: sibling"""

def build_extraction_prompts_sequential(dataset: str, role: str, question: str, item: dict, method=None, args=None):
    """
    Unified extraction prompts for DocRED/CORD/FUNSD/FinER datasets (Sequential mode).
    """
    import json
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist. When asked to extract, output ONLY valid JSON without any thinking process or explanatory text."
    
    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"
    
    template = json.loads(item.get("extract_template", "{}"))
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    chunk_info = item.get("chunk_info", "")
    
    # Dataset-specific instructions
    if dataset == "docred":
        task_desc = "document-level relation extraction using Wikidata property IDs."
        focus_areas = "named entities (persons, organizations, locations, dates, etc.) and their relationships"
        
        output_constraint = DOCRED_RELATIONS_FULL
    
    elif dataset == "cord":
        task_desc = "receipt/invoice information extraction. Extract structured purchase information."
        focus_areas = "menu items (names, prices, counts), subtotal, total amount, tax, payment details"
        output_constraint = "Fill the 'menu' array with all items, and populate subtotal/total/tax fields with exact numeric values."
    
    elif dataset == "funsd":
        task_desc = "form understanding and key-value extraction. Extract form fields and their relationships."
        focus_areas = "form entities (questions, answers, headers, other text), spatial relationships between fields"
        output_constraint = "Fill 'entities' array with all text elements and their labels, 'relations' array with question-answer pairs."
    
    elif dataset == "finer":
        task_desc = "fine-grained financial entity recognition. Identify and classify financial domain entities."
        focus_areas = "financial terms (stock codes, rates, accounting items, metrics, dates, amounts, company names)"
        output_constraint = "Fill 'entities' array with all financial entities, their types, and positions in text."
    
    else:
        task_desc = "document information extraction."
        focus_areas = "key information elements in the document"
        output_constraint = "Fill all schema fields with extracted information."
    
    if role == "planner":
        user_prompt = f"""You are a Document Scanner Agent (Phase 1: Information Discovery).

Task: {task_desc}

Document Section {chunk_info}:
{question}

Instructions:
- Carefully read the document section
- Identify ALL relevant information: {focus_areas}
- Note context and relationships between information elements
- Store findings in latent format (do NOT output final JSON yet)
- Be thorough and precise

Begin scanning:
"""
    
    elif role == "critic":
        user_prompt = f"""You are a Document Validator Agent (Phase 2: Cross-Verification).

Task: {task_desc}

You have latent information from the previous scanning phase.

Document Section {chunk_info}:
{question}

Instructions:
- Cross-check extracted information against document content
- Verify accuracy of: {focus_areas}
- Identify any missing or ambiguous information
- Resolve inconsistencies
- Refine latent representation (do NOT output final JSON yet)

Continue verification:
"""
    
    elif role == "refiner":
        user_prompt = f"""You are a Document Structuring Agent (Phase 3: Organization).

Task: {task_desc}

You have latent information from previous agents.

Document Section {chunk_info}:
{question}

Instructions:
- Organize all extracted information logically
- Resolve any conflicts between different document sections
- Prepare comprehensive latent summary
- Focus on: {focus_areas}
- Refine latent representation (do NOT output final JSON yet)

Continue organization:
"""
    
    elif role == "judger":
        # Get entity list for DocRED
        entity_list = item.get("entity_list", "")
        docred_entity_section = ""
        if dataset == "docred" and entity_list:
            docred_entity_section = f"""
Entities (use ONLY these):
{entity_list}
"""
        
        user_prompt = f"""Task: {task_desc}

{docred_entity_section}Document:
{question}

{output_constraint}

Instructions:
1. First, analyze the document and reason about relationships (write your thinking)
2. Then output FINAL JSON starting with: {{"relations": [...]}}

Format example:
{{"relations": [{{"head": "IBM Research – Brazil", "relation": "P749", "tail": "IBM Research", "evidence": [0]}}]}}

Rules:
- Use entities [0]-[15] only
- Choose correct P-ID from relation list above
- evidence: sentence indices from 0

Begin your analysis:
"""
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_extraction_prompts_hierarchical(dataset: str, role: str, question: str, item: dict, method=None, args=None):
    """
    Unified extraction prompts for DocRED/CORD/FUNSD/FinER datasets (Hierarchical mode).
    """
    import json
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist. When asked to extract, output ONLY valid JSON without any thinking process or explanatory text."
    
    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"
    
    template = json.loads(item.get("extract_template", "{}"))
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    partition_info = item.get("partition_info", "")
    
    # Dataset-specific instructions
    if dataset == "docred":
        task_desc = "document-level relation extraction using Wikidata property IDs"
        focus_areas = "named entities and their relationships using Wikidata IDs"
        output_constraint = DOCRED_RELATIONS_FULL
    
    elif dataset == "cord":
        task_desc = "receipt/invoice extraction"
        focus_areas = "menu items, prices, totals, tax"
        output_constraint = "Fill the 'menu' array with all items, and populate subtotal/total/tax fields with exact numeric values."
    
    elif dataset == "funsd":
        task_desc = "form understanding"
        focus_areas = "form fields, questions, answers, relationships"
        output_constraint = "Fill 'entities' array with all text elements and their labels, 'relations' array with question-answer pairs."
    
    elif dataset == "finer":
        task_desc = "financial entity recognition"
        focus_areas = "financial domain entities and terms"
        output_constraint = "Fill 'entities' array with all financial entities, their types, and positions in text."
    
    else:
        task_desc = "document extraction"
        focus_areas = "key information"
        output_constraint = "Fill all schema fields with extracted information."
    
    if role == "planner":
        user_prompt = f"""You are Document Partition Reader 1.

Task: {task_desc}

Your assigned: {partition_info}

Document Content:
{question}

Instructions:
- Extract ALL {focus_areas} from this partition only
- Be thorough and accurate
- Store in latent format (no final output yet)

Partition 1 extraction:
"""
    
    elif role == "critic":
        user_prompt = f"""You are Document Partition Reader 2.

Task: {task_desc}

Your assigned: {partition_info}

Document Content:
{question}

Instructions:
- Extract ALL {focus_areas} from this partition only
- Be thorough and accurate
- Store in latent format (no final output yet)

Partition 2 extraction:
"""
    
    elif role == "refiner":
        user_prompt = f"""You are Document Partition Reader 3.

Task: {task_desc}

Your assigned: {partition_info}

Document Content:
{question}

Instructions:
- Extract ALL {focus_areas} from this partition only
- Be thorough and accurate
- Store in latent format (no final output yet)

Partition 3 extraction:
"""
    
    elif role == "judger":
        # Get entity list for DocRED
        entity_list = item.get("entity_list", "")
        
        if dataset == "docred":
            docred_entity_section = f"""
Entities (use ONLY these):
{entity_list}
"""
            user_prompt = f"""Task: {task_desc}

{docred_entity_section}Document:
{question}

{output_constraint}

Instructions:
1. First, analyze the document and reason about relationships (write your thinking)
2. Then output FINAL JSON starting with: {{"relations": [...]}}

Format example:
{{"relations": [{{"head": "IBM Research – Brazil", "relation": "P749", "tail": "IBM Research", "evidence": [0]}}]}}

Rules:
- Use entities [0]-[15] only
- Choose correct P-ID from relation list above
- evidence: sentence indices from 0

You have latent info from all partitions. Begin your analysis:
"""
        elif dataset == "funsd":
            user_prompt = f"""Task: {task_desc}

Document:
{question}

Output Format:
{template_str}

Instructions:
1. Extract ALL form fields, questions, answers, and headers from the document
2. Identify relationships between questions and answers
3. Output valid JSON with 'entities' and 'relations' arrays

Entity labels: question, answer, header, other

Output the extracted information as JSON:
"""
        elif dataset == "cord":
            user_prompt = f"""Task: {task_desc}

Document:
{question}

Output Format:
{template_str}

Instructions:
1. Extract ALL menu items with prices
2. Calculate subtotal, tax, and total amounts
3. Output valid JSON matching the schema

Output the extracted information as JSON:
"""
        elif dataset == "finer":
            user_prompt = f"""Task: {task_desc}

Document:
{question}

Output Format:
{template_str}

Instructions:
1. Extract ALL financial entities from the text
2. Identify entity types and positions
3. Output valid JSON with 'entities' array

Output the extracted information as JSON:
"""
        else:
            user_prompt = f"""Task: {task_desc}

Document:
{question}

{output_constraint}

Output Format:
{template_str}

You have latent info from all partitions. Output the final extraction as JSON:
"""
    
    # Check if item has image (multimodal)
    if "image" in item and item["image"] is not None:
        # Multimodal format: image + text
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},  # PIL Image object
                {"type": "text", "text": user_prompt}
            ]},
        ]
    else:
        # Text-only format
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]


def build_multimodal_extraction_message(role: str, image, text_prompt: str, system_message: str = None):
    """
    Helper function to build multimodal messages with image and text.
    Used for vision-language models like Qwen-VL.
    
    Args:
        role: Agent role (planner/critic/refiner/judger)
        image: PIL Image object
        text_prompt: Text instruction
        system_message: System prompt (optional)
    
    Returns:
        List of message dicts in Qwen-VL format
    """
    if system_message is None:
        system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist."
    
    messages = [{"role": "system", "content": system_message}]
    
    if image is not None:
        # Multimodal: image + text
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt}
            ]
        })
    else:
        # Text-only fallback
        messages.append({
            "role": "user",
            "content": text_prompt
        })
    
    return messages


def build_extraction_prompts_text_mas_sequential(dataset: str, role: str, question: str, context: str, item: dict, method=None, args=None):
    """
    Text-MAS extraction prompts for document extraction (Sequential mode).
    Uses explicit text passing between agents.
    """
    import json
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist. When asked to extract, output ONLY valid JSON without any thinking process or explanatory text."
    
    assert method in ["text_mas"], "this prompt only for text_mas method"
    
    template = json.loads(item.get("extract_template", "{}"))
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    chunk_info = item.get("chunk_info", "")
    
    # Dataset-specific instructions
    if dataset == "docred":
        task_desc = "document-level relation extraction using Wikidata property IDs"
        focus_areas = "named entities and their relationships (using Wikidata relation IDs like P17, P569, P27)"
        output_constraint = DOCRED_RELATIONS_FULL
    elif dataset == "cord":
        task_desc = "receipt/invoice information extraction"
        focus_areas = "menu items, prices, subtotal, total, tax"
        output_constraint = "Fill menu array with items, populate subtotal/total/tax fields."
    elif dataset == "funsd":
        task_desc = "form understanding and key-value extraction"
        focus_areas = "form fields (questions, answers, headers)"
        output_constraint = "Fill entities and relations arrays."
    elif dataset == "finer":
        task_desc = "financial named entity recognition"
        focus_areas = "financial terms, amounts, companies, metrics"
        output_constraint = "Fill entities array with all financial entities."
    else:
        task_desc = "document information extraction"
        focus_areas = "key information"
        output_constraint = "Fill all schema fields."
    
    if role == "planner":
        user_prompt = f"""You are a Document Scanner Agent.

Task: {task_desc}

Document Section {chunk_info}:
{question}

Instructions:
1. Carefully scan the document
2. Identify all relevant information: {focus_areas}
3. List all found items with their values
4. Be thorough - don't miss anything

Output your findings in a structured list format.
"""
    
    elif role == "critic":
        user_prompt = f"""You are a Document Validator Agent.

Task: {task_desc}

Document Section {chunk_info}:
{question}

Previous Agent's Analysis:
{context}

Instructions:
1. Review the previous agent's findings
2. Cross-check against the document content
3. Identify any missing or incorrect information
4. Add corrections and missing items

Output your verified and corrected findings.
"""
    
    elif role == "refiner":
        user_prompt = f"""You are a Document Organizer Agent.

Task: {task_desc}

Document Section {chunk_info}:
{question}

Previous Agents' Analysis:
{context}

Instructions:
1. Organize all verified information
2. Resolve any conflicts between findings
3. Prepare comprehensive summary
4. Focus on: {focus_areas}

Output the organized, complete list of extracted information.
"""
    
    elif role == "judger":
        # Get entity list for DocRED
        entity_list = item.get("entity_list", "")
        docred_entity_section = ""
        if dataset == "docred" and entity_list:
            docred_entity_section = f"""
Entities (use ONLY these):
{entity_list}
"""
        
        user_prompt = f"""Task: {task_desc}

{docred_entity_section}Document:
{question}

{output_constraint}

Instructions:
1. First, analyze the document and reason about relationships (write your thinking)
2. Then output FINAL JSON starting with: {{"relations": [...]}}

Format example:
{{"relations": [{{"head": "IBM Research – Brazil", "relation": "P749", "tail": "IBM Research", "evidence": [0]}}]}}

Rules:
- Use entities [0]-[15] only
- Choose correct P-ID from relation list above
- evidence: sentence indices from 0

Previous agents found:
{context}

Begin your analysis:
"""
    
    # Check if item has image (multimodal)
    if "image" in item and item["image"] is not None:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": user_prompt}
            ]},
        ]
    else:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]


def build_extraction_prompts_text_mas_hierarchical(dataset: str, role: str, question: str, context: str, item: dict, method=None, args=None):
    """
    Text-MAS extraction prompts for document extraction (Hierarchical mode).
    Uses explicit text passing between agents, with parallel partition processing.
    """
    import json
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a document information extraction specialist. When asked to extract, output ONLY valid JSON without any thinking process or explanatory text."
    
    assert method in ["text_mas"], "this prompt only for text_mas method"
    
    template = json.loads(item.get("extract_template", "{}"))
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    partition_info = item.get("partition_info", "")
    
    # Dataset-specific instructions (same as sequential)
    if dataset == "docred":
        task_desc = "document-level relation extraction using Wikidata property IDs"
        focus_areas = "named entities and their relationships (using Wikidata relation IDs like P17, P569, P27)"
        output_constraint = DOCRED_RELATIONS_FULL
    elif dataset == "cord":
        task_desc = "receipt/invoice information extraction"
        focus_areas = "menu items, prices, subtotal, total, tax"
        output_constraint = "Fill menu array with items, populate subtotal/total/tax fields."
    elif dataset == "funsd":
        task_desc = "form understanding and key-value extraction"
        focus_areas = "form fields (questions, answers, headers)"
        output_constraint = "Fill entities and relations arrays."
    elif dataset == "finer":
        task_desc = "financial named entity recognition"
        focus_areas = "financial terms, amounts, companies, metrics"
        output_constraint = "Fill entities array with all financial entities."
    else:
        task_desc = "document information extraction"
        focus_areas = "key information"
        output_constraint = "Fill all schema fields."
    
    if role == "planner":
        user_prompt = f"""You are Document Partition Reader 1.

Task: {task_desc}

Your Partition: {partition_info}

Document Content:
{question}

Instructions:
1. Extract all relevant information from YOUR partition
2. Focus on: {focus_areas}
3. List findings with exact values
4. Note cross-references to other sections

Your partition findings:
"""
    
    elif role == "critic":
        user_prompt = f"""You are Document Partition Reader 2.

Task: {task_desc}

Your Partition: {partition_info}

Document Content:
{question}

Other Partition's Findings:
{context}

Instructions:
1. Extract information from YOUR partition
2. Cross-reference with other partition's findings
3. Add complementary information

Your partition findings:
"""
    
    elif role == "refiner":
        user_prompt = f"""You are Document Partition Reader 3.

Task: {task_desc}

Your Partition: {partition_info}

Document Content:
{question}

All Previous Findings:
{context}

Instructions:
1. Extract remaining information from YOUR partition
2. Merge with previous findings
3. Resolve any conflicts
4. Create unified summary

Merged findings:
"""
    
    elif role == "judger":
        # Get entity list for DocRED
        entity_list = item.get("entity_list", "")
        docred_entity_section = ""
        if dataset == "docred" and entity_list:
            docred_entity_section = f"""
Entities (use ONLY these):
{entity_list}
"""
        
        user_prompt = f"""Task: {task_desc}

{docred_entity_section}Document:
{question}

{output_constraint}

Instructions:
1. First, analyze the document and reason about relationships (write your thinking)
2. Then output FINAL JSON starting with: {{"relations": [...]}}

Format example:
{{"relations": [{{"head": "IBM Research – Brazil", "relation": "P749", "tail": "IBM Research", "evidence": [0]}}]}}

Rules:
- Use entities [0]-[15] only
- Choose correct P-ID from relation list above
- evidence: sentence indices from 0

Partition findings:
{context}

Begin your analysis:
"""
    
    # Check if item has image (multimodal)
    if "image" in item and item["image"] is not None:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": user_prompt}
            ]},
        ]
    else:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]


# ============= LoRA Fine-tuned Model Prompts =============
# 这些prompts与finetune_lora.py中的训练格式完全一致

# 常用DocRED关系(简化版,不需要列出全部96个)
DOCRED_COMMON_RELATIONS = """Common relations:
P17(country), P131(located in), P27(citizenship), P569(birth date), P570(death date), 
P19(birthplace), P20(death place), P69(educated at), P108(employer), P102(political party), 
P40(child), P26(spouse), P22(father), P25(mother), P3373(sibling), P161(cast member),
P57(director), P50(author), P175(performer), P264(record label), P495(country of origin),
P577(publication date), P571(inception), P159(headquarters), P749(parent org), P355(subsidiary)"""


def build_lora_extraction_prompt(dataset: str, question: str, item: dict, args=None):
    """
    LoRA微调模型专用prompt - 与训练格式完全一致
    直接输出JSON,无需多轮agent交互
    """
    entity_list = item.get("entity_list", "")
    
    system_msg = "You are an expert document information extraction system. Extract structured information accurately and output valid JSON only."
    
    if dataset == "docred":
        instruction = f"""Task: Document-level relation extraction.

Entities in this document:
{entity_list}

Extract relations between entities using Wikidata property IDs.
{DOCRED_COMMON_RELATIONS}

Output JSON format:
{{"relations": [{{"head": "Entity Name", "relation": "P17", "tail": "Country Name", "evidence": [0, 1]}}]}}

Rules:
1. head/tail must be entity names from the list above
2. relation must be a valid P-ID
3. evidence is list of sentence indices (0-based) that support this relation"""
    
    elif dataset == "funsd":
        instruction = """Task: Extract form fields and their semantic relationships.

Identify form entities with these labels:
- question: Field labels or prompts ("Name:", "Date of Birth:", "Address:")
- answer: Filled-in values or responses
- header: Section titles or form headers
- other: Other text elements

Identify relations:
- Link questions to their corresponding answers
- Use entity text for matching

Output JSON format:
{"entities": [{"text": "Name:", "label": "question"}, {"text": "John Smith", "label": "answer"}], "relations": [{"head": "Name:", "tail": "John Smith"}]}

Example:
Form text: "Employee Name: Sarah Johnson | Department: Engineering | Salary: $85,000"
Output:
{
  "entities": [
    {"text": "Employee Name:", "label": "question"},
    {"text": "Sarah Johnson", "label": "answer"},
    {"text": "Department:", "label": "question"},
    {"text": "Engineering", "label": "answer"},
    {"text": "Salary:", "label": "question"},
    {"text": "$85,000", "label": "answer"}
  ],
  "relations": [
    {"head": "Employee Name:", "tail": "Sarah Johnson"},
    {"head": "Department:", "tail": "Engineering"},
    {"head": "Salary:", "tail": "$85,000"}
  ]
}"""
    
    elif dataset == "cord":
        instruction = """Task: Extract receipt/invoice information from OCR text.

Extract these fields:
1. num_items: Total number of items purchased (integer)
2. subtotal_price: Price before tax/service charge (string with currency)
3. service_price: Service charge amount (string)
4. tax_price: Tax amount (string)
5. total_price: Final total amount (string)
6. etc: Additional charges or notes (string)

Important:
- Extract exact amounts as they appear (e.g., "$12.50", "25.00")
- If a field is not present, use empty string ""
- num_items should be an integer count

Output JSON format:
{"num_items": 3, "subtotal_price": "25.50", "service_price": "2.00", "tax_price": "2.48", "total_price": "29.98", "etc": ""}

Example:
Input: "Item1 $10.00\nItem2 $15.50\nSubtotal $25.50\nTax $2.48\nTotal $27.98"
Output: {"num_items": 2, "subtotal_price": "25.50", "tax_price": "2.48", "total_price": "27.98", "service_price": "", "etc": ""}"""
    
    elif dataset == "finer":
        instruction = """Task: Fine-grained financial named entity recognition.

Identify and classify financial entities in text.

Entity types:
- PER: Person names (executives, analysts, investors)
- ORG: Organizations (companies, banks, institutions)
- LOC: Locations (countries, cities, regions)
- MONEY: Monetary amounts ("$1M", "100 million dollars")
- DATE: Dates and time periods ("Q3 2023", "March 15")
- PERCENT: Percentage values ("5%", "15.5 percent")
- STOCK: Stock tickers and symbols ("AAPL", "NASDAQ:MSFT")
- METRIC: Financial metrics ("revenue", "profit margin", "EPS")
- PRODUCT: Financial products ("bonds", "derivatives", "mortgage")
- LAW: Financial regulations ("Dodd-Frank", "Basel III")

Output JSON format:
{"entities": [{"text": "Apple Inc.", "type": "ORG", "start": 0, "end": 10}, {"text": "$2.5B", "type": "MONEY", "start": 25, "end": 30}]}

Rules:
- start/end are character positions in original text (0-based)
- text is the exact entity string
- type must be one of the predefined types

Example:
Input: "Apple reported $95.3B revenue in Q1 2024, up 5%."
Output: {"entities": [{"text": "Apple", "type": "ORG", "start": 0, "end": 5}, {"text": "$95.3B", "type": "MONEY", "start": 15, "end": 22}, {"text": "Q1 2024", "type": "DATE", "start": 34, "end": 41}, {"text": "5%", "type": "PERCENT", "start": 46, "end": 48}]}"""
    
    else:
        instruction = "Task: Extract information from the document."
    
    user_content = f"{instruction}\n\nDocument text:\n{question}\n\nExtract and output JSON:"
    
    # 检查是否有图像(多模态)
    if "image" in item and item["image"] is not None:
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": user_content}
            ]},
        ]
    else:
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
