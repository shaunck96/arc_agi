# =======================
# Comprehensive ARC AGI Solver with Test Case Solutions
# =======================
# ---------------------
# 1. Import Libraries
# ---------------------
import openai
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import matplotlib.colors as mcolors
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from functools import lru_cache
from enum import Enum, auto
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import matplotlib.image as mpimg

# ---------------------
# 3. Load Environment Variables
# ---------------------
load_dotenv()  # Load variables from .env file

# ---------------------
# 4. Configure Logging
# ---------------------
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(f"arc_solver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------
# 5. Define Utility Functions
# ---------------------

def plot_grid(grid, ax, title="Grid", color_map='viridis', dpi=300):
    if isinstance(grid, str):
        grid = ast.literal_eval(grid)
    grid = np.array(grid)
    cmap = plt.get_cmap(color_map)
    
    if grid.size > 0:
        norm = mcolors.Normalize(np.min(grid), np.max(grid))
        im = ax.matshow(grid, cmap=cmap, norm=norm)
        
        # Adjust text color dynamically based on cell background
        for (i, j), val in np.ndenumerate(grid):
            color = "white" if norm(val) < 0.5 else "black"
            ax.text(j, i, int(val), ha='center', va='center', fontsize=12, color=color)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, color='white')
    
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

def save_plot(df_row, folder_path, idx, dpi=300):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)  # Larger figure size for high resolution
    fig.patch.set_facecolor('black')
    
    plot_grid(df_row.input, axs[0], title=f"Input {idx + 1}")
    
    if df_row.output is not None:
        plot_grid(df_row.output, axs[1], title=f"Output {idx + 1}")
    else:
        axs[1].text(0.5, 0.5, 'No output', ha='center', va='center', fontsize=14, color='white')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
    
    plt.tight_layout()
    
    # Robust filename handling
    sanitized_key = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in df_row.key)
    filename = f"{sanitized_key}_{idx}.png"
    filepath = os.path.join(folder_path, filename)
    
    plt.savefig(filepath, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return filepath

def convert_grid_to_ascii(grid: List[List[int]]) -> str:
    """
    Convert a 2D grid of integers into an ASCII representation.

    Args:
        grid (List[List[int]]): The input grid.

    Returns:
        str: ASCII representation of the grid.
    """
    try:
        rows = []
        for row in grid:
            row_str = "│ " + " ".join(str(cell) for cell in row) + " │"
            rows.append(row_str)
        grid_ascii = "┌" + "───┬" * (len(grid[0]) - 1) + "───┐\n"
        grid_ascii += "\n".join(rows) + "\n"
        grid_ascii += "└" + "───┴" * (len(grid[0]) - 1) + "───┘"
        return grid_ascii
    except Exception as e:
        logging.error(f"Error converting grid to ASCII: {e}")
        raise ValueError("Failed to convert grid to ASCII.")

def parse_json_response(text: str) -> Dict[str, Optional[float]]:
    """
    Parse the JSON-formatted string from the AI response.

    Args:
        text (str): The text containing JSON data.

    Returns:
        Dict[str, Optional[float]]: Parsed JSON data as a dictionary.
    """
    try:
        # Attempt to find JSON within the text
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in the response.")
        json_str = text[json_start:json_end]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"JSON parse error: {e} - Text: {text}")
        return {"correctness": None, "clarity": None, "completeness": None, "creativity": None}

def format_grid(grid):
    """
    Formats the grid into a string representation for better readability in prompts.
    """
    if isinstance(grid, list):
        return '\n'.join([' '.join(map(str, row)) for row in grid])
    return str(grid)

def validate_grid(grid: List[List[int]]) -> bool:
    if not isinstance(grid, list) or not grid:
        return False
    row_length = len(grid[0])
    for row in grid:
        if not isinstance(row, list) or len(row) != row_length:
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True

# ---------------------
# 6. Configure OpenAI API
# ---------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure you have set this in your .env file

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in the .env file.")

# ---------------------
# 7. Define Prompt Templates
# ---------------------
prompt_templates = {
    "original": """
## Instruction
You are a math puzzle solver. You start with an input grid that undergoes certain transformations to become an output grid.
Generate a clear and concise sequence of steps to achieve the output from the input grid.

**Note:** Please limit your response to a maximum of 200 words.

**Input grid:**
{input_grid}

**Output grid:**
{output_grid}
--

## Output Format
Provide the steps in the following JSON format:

{{
    "steps": "Step 1: ..., Step 2: ..., ..."
}}
--

## Response
""",
    "few_shot": """
## Instruction
You are a math puzzle solver. Below are examples of how input grids are transformed into output grids with detailed steps.

### Example 1:
**Input grid:**
┌───┬───┬───┐
│ 2 1 3 │
│ 4 0 6 │
│ 7 8 5 │
└───┴───┴───┘

**Output grid:**
┌───┬───┬───┐
│ 0 1 3 │
│ 2 8 6 │
│ 4 7 5 │
└───┴───┴───┘

**Steps:**
Step 1: Move the zero to the top row.
Step 2: Swap the positions of 2 and 0.
Step 3: Swap the positions of 8 and 4.

### Example 2:
**Input grid:**
┌───┬───┬───┐
│ 1 2 3 │
│ 4 5 6 │
│ 0 8 7 │
└───┴───┴───┘

**Output grid:**
┌───┬───┬───┐
│ 0 2 3 │
│ 1 5 6 │
│ 4 8 7 │
└───┴───┴───┘

**Steps:**
Step 1: Move the zero to the top row.
Step 2: Swap the positions of 1 and 0.
Step 3: Swap the positions of 4 and 0.

## Task
Now, perform the transformation for the following grids.

**Input grid:**
{input_grid}

**Output grid:**
{output_grid}
--

## Output Format
Provide the steps in the following JSON format:

{{
    "steps": "Step 1: ..., Step 2: ..., ..."
}}
--

## Response
""",
    "detailed": """
## Instruction
You are a highly skilled math puzzle solver specializing in analyzing ARC (Abstraction and Reasoning Corpus) puzzles. Your task is to determine the sequence of transformation steps that convert an input grid into an output grid. Each step should be clearly defined and logically lead from the previous state to the next.

**Guidelines:**
1. **Analyze the Grids:** Examine both input and output grids to identify differences.
2. **Identify Patterns:** Look for patterns such as shifting rows/columns, swapping elements, rotating, flipping, or any arithmetic operations.
3. **Define Each Step:** Clearly describe each transformation step required to achieve the output grid from the input grid.
4. **Maintain Clarity:** Ensure that each step is understandable and unambiguous.
5. **Sequence of Operations:** The steps should be in the correct order, with each step building upon the previous one.
6. **Word Limit:** Please keep your response within 300 words.

**Input grid:**
{input_grid}

**Output grid:**
{output_grid}
--

## Output Format
Provide the steps in the following JSON format:

{{
    "steps": "Step 1: ..., Step 2: ..., ..."
}}
--

## Response
"""
}

# ---------------------
# 8. Define Response Generation and Scoring Functions
# ---------------------

def generate_response(prompt: str, generation_args: dict, max_retries: int = 5, backoff_factor: float = 0.5) -> str:
    """
    Generate a response from OpenAI's ChatCompletion API based on the provided prompt and generation arguments.
    Implements retry logic for transient errors.

    Args:
        prompt (str): The prompt to send to the model.
        generation_args (dict): Dictionary of generation parameters.
        max_retries (int): Maximum number of retries for API calls.
        backoff_factor (float): Factor for exponential backoff.

    Returns:
        str: The generated response text.
    """
    messages = [
        {"role": "system", "content": generation_args.get("system_prompt", "You are an expert ARC puzzle evaluator.")}
    ]
    
    # Incorporate few-shot examples if provided
    if "few_shot_examples" in generation_args and isinstance(generation_args["few_shot_examples"], list):
        messages.extend(generation_args["few_shot_examples"])
    
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=generation_args.get("model", "gpt-4"),
                messages=messages,
                temperature=generation_args.get("temperature", 0.2),
                max_tokens=generation_args.get("max_tokens", 500),
                n=generation_args.get("n", 1),
                stop=generation_args.get("stop", None),
                top_p=generation_args.get("top_p", 1.0),
                frequency_penalty=generation_args.get("frequency_penalty", 0.0),
                presence_penalty=generation_args.get("presence_penalty", 0.0),
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError as e:
            logging.warning(f"Rate limit error on attempt {attempt}/{max_retries}: {e}")
        except openai.error.APIError as e:
            logging.warning(f"API error on attempt {attempt}/{max_retries}: {e}")
        except openai.error.Timeout as e:
            logging.warning(f"Timeout error on attempt {attempt}/{max_retries}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error on attempt {attempt}/{max_retries}: {e}")
            break  # For unexpected errors, do not retry

        sleep_time = backoff_factor * (2 ** (attempt - 1))
        logging.info(f"Retrying after {sleep_time} seconds...")
        time.sleep(sleep_time)

    logging.error(f"Failed to generate response after {max_retries} attempts.")
    return ""

def score_response(response_text: str, scoring_args: dict, max_retries: int = 5, backoff_factor: float = 0.5) -> Dict[str, Optional[float]]:
    """
    Score the generated response using OpenAI's ChatCompletion API.
    Implements retry logic for transient errors.

    Args:
        response_text (str): The response text to be scored.
        scoring_args (dict): Dictionary of scoring parameters.
        max_retries (int): Maximum number of retries for API calls.
        backoff_factor (float): Factor for exponential backoff.

    Returns:
        Dict[str, Optional[float]]: A dictionary containing various scores.
    """
    scoring_prompt = f"""
## Instruction
You are an AI assistant designed to evaluate the quality of responses to ARC (Abstraction and Reasoning Corpus) puzzles. Assess the response based on the following criteria:

1. **Correctness:** How accurately does the response explain the transformation steps from the input grid to the output grid?
2. **Clarity:** Is the explanation clear and easy to understand?
3. **Completeness:** Does the response cover all necessary steps without omissions?
4. **Creativity:** Does the response demonstrate innovative thinking in solving the puzzle?

Provide the scores in the following JSON format:

{{
    "correctness": <score out of 10>,
    "clarity": <score out of 10>,
    "completeness": <score out of 10>,
    "creativity": <score out of 10>
}}

Each score should be an integer between 0 and 10.

## Response
"""

    messages = [
        {"role": "system", "content": scoring_args.get("system_prompt", "You are a helpful assistant for scoring responses.")},
        {"role": "user", "content": scoring_prompt + response_text}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=scoring_args.get("model", "gpt-4"),
                messages=messages,
                temperature=0,  # For deterministic output
                max_tokens=150,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            score_text = response.choices[0].message.content.strip()
            # Extract JSON from response
            score_json = parse_json_response(score_text)
            return score_json
        except openai.error.RateLimitError as e:
            logging.warning(f"Rate limit error on attempt {attempt}/{max_retries}: {e}")
        except openai.error.APIError as e:
            logging.warning(f"API error on attempt {attempt}/{max_retries}: {e}")
        except openai.error.Timeout as e:
            logging.warning(f"Timeout error on attempt {attempt}/{max_retries}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error on attempt {attempt}/{max_retries}: {e}")
            break  # For unexpected errors, do not retry

        sleep_time = backoff_factor * (2 ** (attempt - 1))
        logging.info(f"Retrying scoring after {sleep_time} seconds...")
        time.sleep(sleep_time)

    logging.error(f"Failed to score response after {max_retries} attempts.")
    return {"correctness": None, "clarity": None, "completeness": None, "creativity": None}

# ---------------------
# 9. Define Data Processing Functions
# ---------------------
def load_json_data(filepath: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON data.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def flatten_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten the nested JSON data into a pandas DataFrame.

    Args:
        data (Dict[str, Any]): Nested JSON data.

    Returns:
        pd.DataFrame: Flattened DataFrame.
    """
    records = []
    for key, value in data.items():
        for train_type, train_data in value.items():
            for item in train_data:
                records.append({
                    'key': key,
                    'train_type': train_type,
                    'input': item['input'],
                    'output': item.get('output', None)
                })
    df = pd.DataFrame(records)
    return df

def save_and_load_plots(df: pd.DataFrame, folder_path: str = '/kaggle/working/train'):
    """
    Save grid plots and add file paths to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing grid data.
        folder_path (str): Directory to save plot images.

    Returns:
        pd.DataFrame: DataFrame with added 'file_paths' column.
    """
    os.makedirs(folder_path, exist_ok=True)

    def process_row(row, idx):
        return save_plot(row, folder_path, idx)

    df['file_paths'] = df.apply(lambda row: process_row(row, df.index.get_loc(row.name)), axis=1)
    return df

# ---------------------
# 10. Main Processing Loop for All Training Examples
# ---------------------

def process_all_training_examples(data_filepath: str, output_dir: str = "/kaggle/working/"):
    """
    Process all training examples in the ARC AGI dataset.

    Args:
        data_filepath (str): Path to the ARC AGI training challenges JSON file.
        output_dir (str): Directory to save results.

    Returns:
        pd.DataFrame: DataFrame containing all generation and scoring results.
    """
    # Load JSON data
    data = load_json_data(data_filepath)

    # Flatten data into DataFrame
    df = flatten_data(data)

    # Save plots and add file paths
    df = save_and_load_plots(df)

    # Prepare ASCII representations
    df['ascii_input'] = df['input'].apply(convert_grid_to_ascii)
    df['ascii_output'] = df['output'].apply(convert_grid_to_ascii)

    # Define model configurations and prompt versions
    model_versions = {
        "gpt-4": [500],
        "gpt-4-turbo": [500],
        "gpt-3.5-turbo": [500],
        "gpt-4o-mini": [500],
        "chatgpt-4o-latest": [500],
        "gpt-4o-2024-11-20": [500],
        "gpt-4o": [500]    
    }

    prompt_versions = ["original", "few_shot", "detailed"]

    generation_args_variations = [
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        {
            "temperature": 0.2,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        },
        {
            "temperature": 0.4,
            "top_p": 0.85,
            "frequency_penalty": 0.15,
            "presence_penalty": 0.15
        },
        {
            "temperature": 0.6,
            "top_p": 0.8,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2
        },
        {
            "temperature": 0.8,
            "top_p": 0.75,
            "frequency_penalty": 0.25,
            "presence_penalty": 0.25
        },
        {
            "temperature": 1.0,
            "top_p": 0.7,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        }
    ]

    # Generate all combinations
    combinations = []
    for model, token_limits in model_versions.items():
        for token_limit in token_limits:
            for prompt_version in prompt_versions:
                for gen_args in generation_args_variations:
                    combination = {
                        "model": model,
                        "max_tokens": token_limit,
                        "prompt_version": prompt_version,
                        "generation_args": gen_args
                    }
                    combinations.append(combination)

    # Initialize list to store results
    results = []

    total_combinations = len(combinations)

    # Iterate over all combinations for each training example
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Training Examples"):
        for combo in combinations:
            model = combo["model"]
            max_tokens = combo["max_tokens"]
            prompt_version = combo["prompt_version"]
            gen_args = combo["generation_args"]

            prompt = prompt_templates[prompt_version].format(
                input_grid=row['ascii_input'],
                output_grid=row['ascii_output']
            )

            full_generation_args = {
                "model": model,
                "temperature": gen_args["temperature"],
                "max_tokens": max_tokens,
                "n": 1,
                "top_p": gen_args["top_p"],
                "frequency_penalty": gen_args["frequency_penalty"],
                "presence_penalty": gen_args["presence_penalty"],
                "system_prompt": "You are an expert ARC puzzle analyzer."
            }

            logging.info(f"Generating response for Key={row['key']} with Model={model}, Prompt Version={prompt_version}")

            response_text = generate_response(prompt, full_generation_args)

            if not response_text:
                logging.warning(f"Empty response for Key={row['key']} with combination: {combo}")

            # Scoring the response
            scoring_args = {
                "model": "gpt-4",  # Use GPT-4 for scoring
                "system_prompt": "You are a helpful assistant for scoring ARC puzzle responses."
            }

            score_json = score_response(response_text, scoring_args)

            # Aggregate scores if all are present
            if all(value is not None for value in score_json.values()):
                aggregated_score = (
                    score_json["correctness"] +
                    score_json["clarity"] +
                    score_json["completeness"] +
                    score_json["creativity"]
                ) / 4
            else:
                aggregated_score = None

            # Store the result
            result = {
                "key": row['key'],
                "train_type": row['train_type'],
                "model": model,
                "prompt_version": prompt_version,
                "max_tokens": max_tokens,
                "temperature": gen_args["temperature"],
                "top_p": gen_args["top_p"],
                "frequency_penalty": gen_args["frequency_penalty"],
                "presence_penalty": gen_args["presence_penalty"],
                "response": response_text,
                "correctness": score_json.get("correctness"),
                "clarity": score_json.get("clarity"),
                "completeness": score_json.get("completeness"),
                "creativity": score_json.get("creativity"),
                "aggregated_score": aggregated_score
            }

            results.append(result)

            # To respect rate limits
            time.sleep(1)

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Save detailed results
    detailed_results_path = os.path.join(output_dir, "arc_generation_scoring_results.csv")
    results_df.to_csv(detailed_results_path, index=False)
    logging.info(f"Detailed generation and scoring results saved to '{detailed_results_path}'.")

    # Calculate average scores
    average_scores = results_df.groupby(
        ['model', 'prompt_version', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty']
    )[
        ['correctness', 'clarity', 'completeness', 'creativity', 'aggregated_score']
    ].mean().reset_index()

    # Save average scores
    average_scores_path = os.path.join(output_dir, "arc_average_scores.csv")
    average_scores.to_csv(average_scores_path, index=False)
    logging.info(f"Average scores saved to '{average_scores_path}'.")

    return results_df, average_scores

# ---------------------
# 11. Define Evaluation Functions
# ---------------------
# Few-shot examples for evaluation
few_shot_evaluation = [
    {
        "role": "user",
        "content": """
**Transformation Rule:**
Step 1: Move the zero to the top row.
Step 2: Swap the positions of 2 and 0.
Step 3: Swap the positions of 8 and 4.

**Evaluation:**
The transformation rule correctly moves the zero to the top row and performs the necessary swaps to achieve the desired output grid. The steps are clear and cover all necessary transformations without omissions. However, the evaluation could provide more insight into alternative strategies or optimizations.
"""
    },
    {
        "role": "assistant",
        "content": """
{
    "correctness": 9,
    "clarity": 8,
    "completeness": 7,
    "creativity": 6
}
"""
    },
    {
        "role": "user",
        "content": """
**Transformation Rule:**
Step 1: Rotate the entire grid 90 degrees clockwise.
Step 2: Swap the first and last rows.

**Evaluation:**
The transformation rule rotates the grid as specified and swaps the rows to achieve the output grid. The steps are logical and well-explained. However, it lacks thoroughness in explaining why these specific transformations lead to the desired outcome.
"""
    },
    {
        "role": "assistant",
        "content": """
{
    "correctness": 8,
    "clarity": 9,
    "completeness": 6,
    "creativity": 7
}
"""
    }
]

evaluation_prompt_template = """
## Instruction
You are an expert ARC puzzle evaluator. Given an input grid, an output grid, and a transformation rule, evaluate whether the transformation rule correctly transforms the input grid into the output grid. Provide a detailed evaluation.

**Input Grid:**
{input_grid}

**Output Grid:**
{output_grid}

**Transformation Rule:**
{transformation_rule}

## Evaluation
Provide a detailed evaluation of the transformation rule based on the above grids.

## Output Format
{{
  "reflection": "Your detailed evaluation here",
  "suggestions": "Your suggested corrections here (if any)"
}}

## Response
"""

def evaluate_transformation_rules(detailed_results_df: pd.DataFrame, data_filepath: str, output_filepath: str, few_shot_examples: List[Dict[str, str]]):
    """
    Evaluate transformation rules from the detailed results and save the evaluation scores.

    Args:
        detailed_results_df (pd.DataFrame): DataFrame containing generation and scoring results.
        data_filepath (str): Path to the ARC AGI training challenges JSON file.
        output_filepath (str): Path to save the evaluation scoring results.
        few_shot_examples (List[Dict[str, str]]): Few-shot examples for the evaluator.
    """
    # Load JSON data to get grid representations
    data = load_json_data(data_filepath)
    df_master = flatten_data(data)

    # Merge detailed_results_df with df_master to get grids
    merged_df = pd.merge(detailed_results_df, df_master, on=['key', 'train_type'], how='left')

    # Initialize list to store evaluation results
    evaluation_results = []

    total_rows = len(merged_df)
    for idx, row in tqdm(merged_df.iterrows(), total=total_rows, desc="Evaluating Transformation Rules"):
        input_grid_str = row['input']
        output_grid_str = row['output']
        transformation_rule = row['response']

        # Validate grids
        if not validate_grid(input_grid_str) or not validate_grid(output_grid_str):
            logging.error(f"Invalid grid format for Key={row['key']}. Skipping evaluation.")
            evaluation_results.append({
                "reflection": "Invalid grid format.",
                "suggestions": "None"
            })
            continue

        # Convert grids to ASCII
        ascii_input = convert_grid_to_ascii(input_grid_str)
        ascii_output = convert_grid_to_ascii(output_grid_str)

        # Prepare evaluation prompt
        evaluation_prompt = evaluation_prompt_template.format(
            input_grid=ascii_input,
            output_grid=ascii_output,
            transformation_rule=transformation_rule
        )

        # Define generation arguments for evaluation
        generation_args = {
            "model": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 500,
            "n": 1,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "system_prompt": "You are an expert ARC puzzle evaluator.",
            "few_shot_examples": few_shot_examples
        }

        logging.info(f"Evaluating transformation rule for Key={row['key']}")

        # Generate evaluation response
        evaluation_text = generate_response(evaluation_prompt, generation_args)

        if not evaluation_text:
            logging.warning(f"Empty evaluation response for Key={row['key']}.")
            evaluation_text = "Evaluation not available."

        # Scoring the evaluation
        scoring_args = {
            "model": "gpt-4",
            "system_prompt": "You are a helpful assistant for scoring evaluations."
        }

        score_json = score_response(evaluation_text, scoring_args)

        # Store the evaluation and scores
        evaluation_result = {
            "reflection": evaluation_text,
            "correctness": score_json.get("correctness"),
            "clarity": score_json.get("clarity"),
            "thoroughness": score_json.get("thoroughness"),
            "insightfulness": score_json.get("insightfulness")
        }

        # Aggregate scores if all are present
        if all(value is not None for value in score_json.values()):
            aggregated_score = (
                score_json["correctness"] +
                score_json["clarity"] +
                score_json["thoroughness"] +
                score_json["insightfulness"]
            ) / 4
            evaluation_result["aggregated_score"] = aggregated_score
        else:
            evaluation_result["aggregated_score"] = None

        evaluation_results.append(evaluation_result)

        # To respect rate limits
        time.sleep(1)

    # Create DataFrame from evaluation results
    evaluation_df = pd.DataFrame(evaluation_results)

    # Combine with detailed_results_df
    final_df = pd.concat([merged_df.reset_index(drop=True), evaluation_df.reset_index(drop=True)], axis=1)

    # Save the evaluation results
    final_df.to_csv(output_filepath, index=False)
    logging.info(f"Evaluation and scoring complete. Results saved to '{output_filepath}'.")

    return final_df

# ---------------------
# 12. Define Test Case Solving Functions
# ---------------------
def solve_test_case(sample_input, sample_output, transformation_steps, test_input):
    """
    Apply transformation steps to the test input to generate the test output.

    Args:
        sample_input (List[List[int]]): Sample input grid.
        sample_output (List[List[int]]): Sample output grid.
        transformation_steps (str): Transformation steps derived from the sample.
        test_input (List[List[int]]): Test input grid.

    Returns:
        List[List[int]]: Generated test output grid.
    """
    # This function assumes that transformation_steps is a sequence of steps that can be interpreted and applied programmatically.
    # However, since the transformation_steps are in natural language, applying them programmatically is non-trivial.
    # Therefore, we leverage OpenAI's API to perform the transformation based on the steps.
    
    transformation_prompt = f"""
You are an expert ARC AGI puzzle solver.

You have the following transformation steps derived from a sample puzzle:

{transformation_steps}

Apply these transformation steps to the following test input grid to generate the corresponding output grid.

## Test Input:
{format_grid(test_input)}

## Output Format
{{
  "transformation_steps": "Describe each step clearly.",
  "output_grid": "Provide the resulting grid after applying the transformation steps."
}}

## Response (please respond only in JSON and no additional words in your response)
"""

    messages = [
        {"role": "system", "content": "You are an expert ARC AGI puzzle solver."},
        {"role": "user", "content": transformation_prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0,  # For deterministic output
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        content = response.choices[0].message.content.strip()
        logging.info("Test Case Solution Response:")
        logging.info(content)
        # Parse JSON response using json.loads
        result = json.loads(content)
        return result["transformation_steps"], result["output_grid"]
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error in solve_test_case: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error in solve_test_case: {e}")
        return None, None

# ---------------------
# 13. Define Main Execution
# ---------------------
def main():
    """
    Main function to execute the entire processing, evaluation, and test case solving pipeline.
    """
    # Define file paths
    data_filepath = "/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json"  # Update if different
    detailed_results_path = "/kaggle/working/arc_generation_scoring_results.csv"
    average_scores_path = "/kaggle/working/arc_average_scores.csv"
    evaluation_results_path = "/kaggle/working/arc_evaluation_scoring_results.csv"
    test_case_solutions_path = "/kaggle/working/arc_test_case_solutions.csv"

    # Step 1: Process all training examples
    detailed_results_df, average_scores_df = process_all_training_examples(
        data_filepath=data_filepath,
        output_dir="/kaggle/working/"
    )

    # Step 2: Evaluate transformation rules
    final_evaluation_df = evaluate_transformation_rules(
        detailed_results_df=detailed_results_df,
        data_filepath=data_filepath,
        output_filepath=evaluation_results_path,
        few_shot_examples=few_shot_evaluation
    )

    # Step 3: Solve Test Cases
    # Load JSON data to get test cases
    data = load_json_data(data_filepath)
    df_master = flatten_data(data)

    # Filter test cases
    df_tests = df_master[df_master['train_type'] == 'test'].reset_index(drop=True)
    df_trains = df_master[df_master['train_type'] == 'train'].reset_index(drop=True)

    # Assuming that 'key' can be used to match training and test puzzles
    # If not, adjust the matching logic accordingly
    # For demonstration, we'll iterate over test cases and find their corresponding train cases

    # Initialize list to store test case solutions
    test_case_solutions = []

    for idx, test_row in tqdm(df_tests.iterrows(), total=df_tests.shape[0], desc="Solving Test Cases"):
        key = test_row['key']
        # Find corresponding train case(s)
        corresponding_train = df_trains[df_trains['key'] == key]

        if corresponding_train.empty:
            logging.warning(f"No corresponding training example found for Test Key={key}. Skipping.")
            continue

        # For simplicity, take the first matching training example
        train_row = corresponding_train.iloc[0]

        # Get transformation steps from the evaluated transformations
        # Filter final_evaluation_df for this key
        eval_transforms = final_evaluation_df[final_evaluation_df['key'] == key]

        if eval_transforms.empty:
            logging.warning(f"No evaluated transformation steps found for Train Key={key}. Skipping.")
            continue

        # Select the transformation with the highest aggregated_score
        best_transform = eval_transforms.sort_values(by='aggregated_score', ascending=False).iloc[0]
        transformation_steps = best_transform['response']

        # Solve the test case using the transformation steps
        generated_steps, generated_output = solve_test_case(
            sample_input=train_row['input'],
            sample_output=train_row['output'],
            transformation_steps=transformation_steps,
            test_input=test_row['input']
        )

        if not generated_steps or not generated_output:
            logging.warning(f"Failed to generate solution for Test Key={key}.")
            generated_steps = "Solution not available."
            generated_output = "Solution not available."

        # Validate generated_output
        try:
            # Convert the generated_output back to grid format
            output_grid = ast.literal_eval(generated_output)
            if not validate_grid(output_grid):
                raise ValueError("Generated output grid is invalid.")
        except Exception as e:
            logging.error(f"Invalid generated output for Test Key={key}: {e}")
            generated_output = "Invalid generated output."

        # Store the solution
        solution = {
            "key": key,
            "test_input": test_row['input'],
            "transformation_steps": generated_steps,
            "generated_output": generated_output
        }

        test_case_solutions.append(solution)

        # To respect rate limits
        time.sleep(1)

    # Create DataFrame from test case solutions
    test_case_solutions_df = pd.DataFrame(test_case_solutions)

    # Save test case solutions
    test_case_solutions_df.to_csv(test_case_solutions_path, index=False)
    logging.info(f"Test case solutions saved to '{test_case_solutions_path}'.")

    # Step 4: Evaluate Test Case Solutions (Optional)
    # You can implement similar evaluation functions to assess the quality of test case solutions.

    # Step 5: Analyze and Display Best Results
    final_evaluation_df.dropna(subset=['aggregated_score'], inplace=True)
    best_df = final_evaluation_df[final_evaluation_df['aggregated_score'] > 7]

    print(f"Number of high-scoring transformations: {len(best_df)}")
    print(best_df[['key', 'model', 'prompt_version', 'aggregated_score']])

    # Optional: Save best_df to CSV
    best_df.to_csv("/kaggle/working/arc_best_transformations.csv", index=False)
    logging.info("Best transformations saved to 'arc_best_transformations.csv'.")

    # Optional: Display a sample test case solution
    if not test_case_solutions_df.empty:
        sample_solution = test_case_solutions_df.iloc[0]
        print("\n=== Sample Test Case Solution ===")
        print(f"Key: {sample_solution['key']}")
        print("Transformation Steps:")
        print(sample_solution['transformation_steps'])
        print("Generated Output Grid:")
        print(sample_solution['generated_output'])

if __name__ == "__main__":
    main()
