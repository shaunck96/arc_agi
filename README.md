# Comprehensive ARC AGI Solver Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
3. [Overview of the Script](#overview-of-the-script)
4. [Detailed Breakdown](#detailed-breakdown)
    - [1. Data Loading and Preparation](#1-data-loading-and-preparation)
    - [2. Plot Generation](#2-plot-generation)
    - [3. Prompt Templates](#3-prompt-templates)
    - [4. Response Generation](#4-response-generation)
    - [5. Scoring Responses](#5-scoring-responses)
    - [6. Evaluating Transformation Steps](#6-evaluating-transformation-steps)
    - [7. Solving Test Cases](#7-solving-test-cases)
    - [8. Aggregating and Saving Results](#8-aggregating-and-saving-results)
5. [Best Practices](#best-practices)
6. [Conclusion](#conclusion)

---

## Introduction

The **ARC AGI Solver** is a comprehensive Python script designed to automate the process of solving puzzles from the [Abstraction and Reasoning Corpus (ARC)](https://github.com/fabianp/ARC). This script leverages OpenAI's GPT models to generate transformation steps from input-output grid pairs, evaluate these steps, and apply them to solve corresponding test cases. The solver is built with scalability, robustness, and maintainability in mind, ensuring it can handle a wide range of puzzles efficiently.

---

## Setup and Installation

Before running the ARC AGI Solver, ensure that your environment is correctly set up with all necessary dependencies and configurations.

1. **Clone the Repository and Install Dependencies:**

    ```bash
    # Upgrade pip and install required packages
    pip install --upgrade pip
    pip install openai==0.28
    ```

2. **Environment Variables:**

    - **OpenAI API Key:** The script requires access to OpenAI's API. Store your API key securely using environment variables.

        - Create a `.env` file in your project directory with the following content:

            ```
            OPENAI_API_KEY=your_openai_api_key_here
            ```

        - **Security Note:** Ensure that the `.env` file is added to `.gitignore` to prevent accidental commits of sensitive information.

3. **Directory Structure:**

    - The script assumes a specific directory structure, especially for input and output files. Ensure that paths like `/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json` are correctly set or adjusted based on your environment.

---

## Overview of the Script

The ARC AGI Solver is organized into several key sections, each responsible for different aspects of the puzzle-solving process:

1. **Data Loading and Preparation:** Loads and preprocesses the ARC AGI dataset.
2. **Plot Generation:** Visualizes input and output grids for better understanding.
3. **Prompt Templates:** Defines various prompts used to interact with OpenAI's models.
4. **Response Generation:** Generates transformation steps based on grid pairs.
5. **Scoring Responses:** Evaluates the quality of the generated transformation steps.
6. **Evaluating Transformation Steps:** Assesses the effectiveness of transformation steps.
7. **Solving Test Cases:** Applies transformation steps to solve corresponding test puzzles.
8. **Aggregating and Saving Results:** Compiles and stores all results for analysis.

Each section is modular, allowing for easy maintenance and potential future enhancements.

---

## Detailed Breakdown

### 1. Data Loading and Preparation

**Functionality:**

- **Load JSON Data:** Reads the ARC AGI training challenges from a JSON file.
- **Flatten Data:** Converts the nested JSON structure into a flat pandas DataFrame for easier processing.
- **Grid Validation:** Ensures that each grid is a valid 2D list of integers to prevent runtime errors.

**Key Functions:**

- `load_json_data(filepath: str) -> Dict[str, Any]`: Loads JSON data from the specified file.
- `flatten_data(data: Dict[str, Any]) -> pd.DataFrame`: Flattens the nested JSON data into a pandas DataFrame.
- `validate_grid(grid: List[List[int]]) -> bool`: Validates the structure and content of each grid.

**Usage:**

```python
data = load_json_data(data_filepath)
df = flatten_data(data)
```

### 2. Plot Generation

**Functionality:**

- **Visual Representation:** Generates visual plots of input and output grids to aid in understanding the transformations.
- **Dynamic Text Coloring:** Adjusts text color based on cell background for better readability.

**Key Functions:**

- `plot_grid(grid, ax, title="Grid", color_map='viridis', dpi=300)`: Plots a single grid.
- `save_plot(df_row, folder_path, idx, dpi=300)`: Saves side-by-side plots of input and output grids for each puzzle.

**Usage:**

```python
df = save_and_load_plots(df)
```

### 3. Prompt Templates

**Functionality:**

- **Define Interaction Patterns:** Specifies how prompts are structured when interacting with OpenAI's models.
- **Variety of Approaches:** Includes original, few-shot, and detailed prompt versions to experiment with different response qualities.

**Templates Included:**

- **Original:** Basic prompt asking for transformation steps.
- **Few-Shot:** Provides examples to guide the model.
- **Detailed:** In-depth instructions for generating comprehensive transformation steps.

**Usage:**

```python
prompt = prompt_templates[prompt_version].format(
    input_grid=row['ascii_input'],
    output_grid=row['ascii_output']
)
```

### 4. Response Generation

**Functionality:**

- **Leverage GPT Models:** Uses OpenAI's ChatCompletion API to generate transformation steps based on input-output grid pairs.
- **Retry Logic:** Implements exponential backoff to handle transient API errors gracefully.

**Key Function:**

- `generate_response(prompt: str, generation_args: dict, max_retries: int = 5, backoff_factor: float = 0.5) -> str`: Generates a response from the model with retry capabilities.

**Usage:**

```python
response_text = generate_response(prompt, full_generation_args)
```

### 5. Scoring Responses

**Functionality:**

- **Evaluate Transformation Steps:** Assesses the quality of generated transformation steps based on criteria like correctness, clarity, completeness, and creativity.
- **JSON Formatting:** Ensures that scores are returned in a structured JSON format for easy analysis.

**Key Function:**

- `score_response(response_text: str, scoring_args: dict, max_retries: int = 5, backoff_factor: float = 0.5) -> Dict[str, Optional[float]]`: Scores the response using the model.

**Usage:**

```python
score_json = score_response(response_text, scoring_args)
```

### 6. Evaluating Transformation Steps

**Functionality:**

- **Detailed Assessment:** Provides a comprehensive evaluation of transformation steps, including reflections and suggestions for improvements.
- **Aggregate Scoring:** Combines individual scores into an aggregated metric for easier comparison.

**Key Components:**

- **Few-Shot Evaluation Examples:** Supplies examples to guide the evaluator in scoring.
- **Evaluation Prompt Template:** Structures the prompt for evaluating transformation steps.

**Usage:**

```python
final_evaluation_df = evaluate_transformation_rules(
    detailed_results_df=detailed_results_df,
    data_filepath=data_filepath,
    output_filepath=evaluation_results_path,
    few_shot_examples=few_shot_evaluation
)
```

### 7. Solving Test Cases

**Functionality:**

- **Apply Transformations:** Utilizes the best transformation steps derived from training examples to solve corresponding test puzzles.
- **JSON Parsing:** Extracts transformation steps and output grids from model responses.

**Key Functions:**

- `solve_test_case(sample_input, sample_output, transformation_steps, test_input) -> Tuple[str, List[List[int]]]`: Solves a test case using provided transformation steps.
- `evaluate_transformation_rules(...)`: Assesses and selects the best transformation steps for each test case.

**Usage:**

```python
generated_steps, generated_output = solve_test_case(
    sample_input=train_row['input'],
    sample_output=train_row['output'],
    transformation_steps=transformation_steps,
    test_input=test_row['input']
)
```

### 8. Aggregating and Saving Results

**Functionality:**

- **Compile Results:** Gathers all generation, scoring, evaluation, and test case solution data into structured DataFrames.
- **Save to CSV:** Exports detailed and summarized results for further analysis.

**Key Steps:**

1. **Detailed Results:** `arc_generation_scoring_results.csv` contains all generated transformation steps and their scores.
2. **Average Scores:** `arc_average_scores.csv` summarizes average scores across different model and prompt configurations.
3. **Evaluation Scores:** `arc_evaluation_scoring_results.csv` includes evaluations of transformation steps.
4. **Test Case Solutions:** `arc_test_case_solutions.csv` holds solutions to all test cases.
5. **Best Transformations:** `arc_best_transformations.csv` highlights top-performing transformations based on aggregated scores.

**Usage:**

```python
final_df.to_csv(evaluation_results_path, index=False)
test_case_solutions_df.to_csv(test_case_solutions_path, index=False)
best_df.to_csv("/kaggle/working/arc_best_transformations.csv", index=False)
```

---

