#!/usr/bin/env python3
"""
Step 1: Translate to Selected Symbolic Language
Translates each logic problem to a specified symbolic language
"""

import argparse
import json
import os
import re
from tqdm import tqdm
from typing import Dict, List

from utils.llm_helper import LLMHelper
from utils.dataset_detector import detect_dataset


def load_data(input_file: str) -> List[Dict]:
    """Load input data from JSON file"""
    with open(input_file, 'r') as f:
        return json.load(f)


def save_data(output_file: str, data: List[Dict]):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_role_description(sl: str, item: Dict) -> str:
    """
    Get role description based on SL and dataset detected from item
    
    Args:
        sl: Symbolic language (LP/FOL/SAT)
        item: Problem item to detect dataset from
    
    Returns:
        Role description string
    """
    # Detect dataset from item
    dataset = detect_dataset(item)
    
    # Role descriptions for ProntoQA/ProofWriter (from translate_config.yaml)
    if dataset in ['ProntoQA', 'ProofWriter']:
        if sl == 'LP':
            return """Your final task is to translate the logic problem in natural language into LP logic formulas:
      1. define all the predicates in the problem
      2. parse the problem into logic rules based on the defined predicates
      3. write all the facts mentioned in the problem
      4. parse the question into the logic form (Use && to represent AND, and you cannot use 'NOT' or other negations in LP)
      ----------------
      Example:
      Context: Each jompus is fruity.
      (... more context here ...)
      Rompuses are zumpuses. Alex is a tumpus.

      Question: True or false: Alex is not shy.

      Predicates:
      Jompus($x, bool) ::: Does x belong to Jompus?
      (... more predicates here ...)
      Zumpus($x, bool) ::: Does x belong to Zumpuses?

      Facts:
      Tumpuses(Alex, True)

      Rules:
      Jompus($x, True) >>> Fruity($x, True)
      (... more rules here ...)
      Dumpus($x, True) >>> Rompus($x, True)

      Query:
      Shy(Alex, False)
      ---------------
      Return **only** Predicates, Facts, Rules, and Query.
      Do not include unnecessary symbols such as "**", "numbered lists (like 1., 2., 3., 4.)", " - " etc.
      """
      
        elif sl == 'FOL':
            return """Your final task is to translate the logic problem in natural language into first-order logic formulas. The grammar of first-order logic is defined as follows:

        logical conjunction: expr1 ∧ expr2
        logical disjunction: expr1 ∨ expr2
        logical exclusive disjunction: expr1 ⊕ expr2
        logical negation: ¬expr1
        expr1 implies expr2: expr1 → expr2
        expr1 if and only if expr2: expr1 ↔ expr2
        logical universal quantification: ∀ x
        logical existential quantification: ∃ x
        Output format: logic form ::: description

        ---------------
        Example：
        Context: All people who regularly drink coffee are
        dependent on caffeine.
        (... more context here ...)
        If Rina is not a person dependent on caffeine and a
        student, then Rina is either a person dependent
        on caffeine and a student, or a person dependent
        on caffeine nor a student, or neither a person
        dependent on caffeine nor a student.

        Question: Based on the above information, is the
        following statement true, false, or uncertain?
        Rina is either a person who jokes about being
        addicted to caffeine or is unaware that caffeine
        is a drug.

        Predicates:
        Dependent(x) ::: x is a person dependent on caffeine
        (... more predicates here ...)
        Student(x) ::: x is a student

        Premises:
        ∀x (Drinks(x) → Dependent(x)) ::: All people who
        regularly drink coffee are dependent on caffeine.

        (... more premises here ...)
        ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes
        about being addicted to caffeine is unaware that
        caffeine is a drug.

        Conclusion:
        Jokes(rina) ⊕ Unaware(rina) ::: Rina is either a
        person who jokes about being addicted to caffeine
        or is unaware that caffeine is a drug.
        ---------------
        Return **only** Predicates, Premises, and Conclusion.
        Do not include unnecessary symbols such as "**", "numbered lists (like 1., 2., 3., 4.)", " - " etc.
        """
        
        elif sl == 'SAT':
            return """Your final task is to parse the logic problem in natural language as a SAT problem using Z3 syntax, defining declarations, constraints, and options.
        1. Always include all three section headers in order: # Declarations, # Constraints, # Options
        2. Declarations must follow exact patterns:
        - name = EnumSort([items, ...])  for non-numeric items
        - name = IntSort([numbers, ...])  for numeric items
        - name = Function([types] -> [return_type])
        3. Constraints support:
        - Direct expressions with ==, !=, <=, >=, <, >, Implies(), And(), Or(), Not()
        - ForAll([var:type, ...], expr) and Exists([var:type, ...], expr)
        - Count([var:type], condition)
        - Distinct([var:type], expr)
        4. Options must use predefined functions:
        - is_valid(), is_sat(), is_unsat()
        5. Add explanation with :::
        6. Avoid:
        - Add # in any other places apart from three section headers
        - Add any other unnecessary comment or dashes

      -----------
      Example：
      Context: Bob is cold. Bob is quiet. Bob is red. Bob is smart. Charlie is kind. Charlie is quiet. Charlie is red. Charlie is rough. Dave is cold. Dave is kind. Dave is smart. Fiona is quiet. If something is quiet and cold then it is smart. Red, cold things are round. If something is kind and rough then it is red. All quiet things are rough. Cold, smart things are red. If something is rough then it is cold. All red things are rough. If Dave is smart and Dave is kind then Dave is quiet.
      Question: True or false: Charlie is kind.

      # Declarations
      objects = EnumSort([Bob, Charlie, Dave, Fiona])
      attributes = EnumSort([cold, quiet, red, smart, kind, rough, round])
      has_attribute = Function([objects, attributes] -> [bool])
      
      # Constraints
      has_attribute(Bob, cold) == True ::: Bob is cold.
      has_attribute(Bob, quiet) == True ::: Bob is quiet.
      has_attribute(Bob, red) == True ::: Bob is red.
      has_attribute(Bob, smart) == True ::: Bob is smart.
      has_attribute(Charlie, kind) == True ::: Charlie is kind.
      has_attribute(Charlie, quiet) == True ::: Charlie is quiet.
      has_attribute(Charlie, red) == True ::: Charlie is red.
      has_attribute(Charlie, rough) == True ::: Charlie is rough.
      has_attribute(Dave, cold) == True ::: Dave is cold.
      has_attribute(Dave, kind) == True ::: Dave is kind.
      has_attribute(Dave, smart) == True ::: Dave is smart.
      has_attribute(Fiona, quiet) == True ::: Fiona is quiet.
      ForAll([x:objects], Implies(And(has_attribute(x, quiet) == True, has_attribute(x, cold) == True), has_attribute(x, smart) == True)) ::: If something is quiet and cold then it is smart.
      ForAll([x:objects], Implies(And(has_attribute(x, red) == True, has_attribute(x, cold) == True), has_attribute(x, round) == True)) ::: Red, cold things are round.
      ForAll([x:objects], Implies(And(has_attribute(x, kind) == True, has_attribute(x, rough) == True), has_attribute(x, red) == True)) ::: If something is kind and rough then it is red.
      ForAll([x:objects], Implies(has_attribute(x, quiet) == True, has_attribute(x, rough) == True)) ::: All quiet things are rough.
      ForAll([x:objects], Implies(And(has_attribute(x, cold) == True, has_attribute(x, smart) == True), has_attribute(x, red) == True)) ::: Cold, smart things are red.
      ForAll([x:objects], Implies(has_attribute(x, rough) == True, has_attribute(x, cold) == True)) ::: If something is rough then it is cold.
      ForAll([x:objects], Implies(has_attribute(x, red) == True, has_attribute(x, rough) == True)) ::: All red things are rough.
      Implies(And(has_attribute(Dave, smart) == True, has_attribute(Dave, kind) == True), has_attribute(Dave, quiet) == True) ::: If Dave is smart and Dave is kind then Dave is quiet.
      
      # Options
      is_valid(has_attribute(Charlie, kind) == True) ::: Charlie is kind is True (A).
      is_unsat(has_attribute(Charlie, kind) == True) ::: Charlie is kind is False (B).
      ---------------
      Return **only** # Declarations, # Constraints, # Options.
      Do not include unnecessary symbols such as "**", "numbered lists (like 1., 2., 3., 4.)", " - " etc.
      """
    
    # Role descriptions for LogicalDeduction (from logideduct_translate_config.yaml)
    elif dataset == 'LogicalDeduction':
        if sl == 'LP':
            return """You are given a problem description and a question. The task is to:
      1. define all the predicates in the problem
      2. parse the problem into logic rules based on the defined predicates
      3. write all the facts mentioned in the problem
      4. parse the question into the logic form (Use && to represent AND)
      5. **Finite derivations** :Avoid rules that introduce new Skolem‑like symbols or anonymous constants.
      6. parse the question into the logic form (Use && to represent AND, and you cannot use 'NOT' or other negations)
      ----------------
      Example:
      context: The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a shelf, there are five books: a white book, an orange book, a yellow book, a blue book, and a red book. The yellow book is to the left of the white book. The red book is to the right of the blue book. The yellow book is to the right of the orange book. The blue book is to the right of the white book.
      question: Which of the following is true?
      options: 
      A) The white book is the second from the right.
      B) The orange book is the second from the right.
      C) The yellow book is the second from the right.
      D) The blue book is the second from the right.
      E) The red book is the second from the right.

      Predicates:
      Book($x, bool)                  ::: $x is one of the five books.
      LeftOf($x, $y, bool)            ::: Book $x is strictly to the left of book $y.
      RightOf($x, $y, bool)           ::: Book $x is strictly to the right of book $y.
      RightMost($x, bool)             ::: Book $x is the right‑most book on the shelf.
      SecondFromRight($x, bool)       ::: Book $x is the second book from the right.

      Facts:
      Book(white,  True)              ::: The white book.
      Book(orange, True)              ::: The orange book.
      Book(yellow, True)              ::: The yellow book.
      Book(blue,   True)              ::: The blue book.
      Book(red,    True)              ::: The red book.
      LeftOf(yellow, white,  True)    ::: The yellow book is to the left of the white book.
      RightOf(red,   blue,   True)    ::: The red book is to the right of the blue book.
      RightOf(yellow, orange, True)   ::: The yellow book is to the right of the orange book.
      RightOf(blue,  white,  True)    ::: The blue book is to the right of the white book.

      Rules:
      LeftOf($a, $b, True) >>> RightOf($b, $a, True) ::: If $a is left of $b, then $b is right of $a.
      RightOf($a, $b, True) >>> LeftOf($b, $a, True) ::: If $a is right of $b, then $b is left of $a.
      RightOf($a, $b, True) && RightOf($b, $c, True) >>> RightOf($a, $c, True) ::: Right‑of is transitive.
      RightOf($b, white,  True) && RightOf($b, orange, True) && RightOf($b, yellow, True) && RightOf($b, blue,   True)  >>> RightMost($b, True) ::: A book that is to the right of all the other four is the right‑most book.
      RightMost($rm, True) && RightOf($rm, $s, True) && RightOf($s, white,  True) && RightOf($s, orange, True) && RightOf($s, yellow, True) >>> SecondFromRight($s, True) ::: The book immediately left of the right‑most—and still right of the remaining three—is second from the right.

      Query:
      SecondFromRight(white,  True)  ::: Option A
      SecondFromRight(orange, True)  ::: Option B
      SecondFromRight(yellow, True)  ::: Option C
      SecondFromRight(blue,   True)  ::: Option D
      SecondFromRight(red,    True)  ::: Option E
      ---------------
      Return **only** Predicates, Facts, Rules, and Query.
      Do not include unnecessary symbols such as "**", "numbered lists (like 1., 2., 3., 4.)", " - " etc.
      """
    
      
        elif sl == 'FOL':
            return """Task Description: Given a problem description and a question. The task is to parse the problem and the question into first-order logic formulas. The grammar of first-order logic is defined as follows:

        variable names: always use single/multi-character with underscores as variable names such as x, y, cow, blue_jay. Never use Dollar Sign Variables.
        logical conjunction: expr1 ∧ expr2
        logical disjunction: expr1 ∨ expr2
        logical exclusive disjunction: expr1 ⊕ expr2
        logical negation: ¬expr1
        expr1 implies expr2: expr1 → expr2
        expr1 if and only if expr2: expr1 ↔ expr2
        logical universal quantification: ∀ x
        logical existential quantification: ∃ x
        Output format: logic form ::: description

        ---------------
        Example：
        context: The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nA fruit stand sells five fruits: mangoes, kiwis, plums, pears, and watermelons. The kiwis are less expensive than the plums. The pears are the third-most expensive. The kiwis are the second-cheapest. The watermelons are the most expensive.
        question: Which of the following is true?
        options: 
        A) The mangoes are the third-most expensive.
        B) The kiwis are the third-most expensive.
        C) The plums are the third-most expensive.
        D) The pears are the third-most expensive.
        E) The watermelons are the third-most expensive.

        Predicates:
        Rank(fruit, pos) ::: fruit has price position pos, where pos ∈ {one,two,three,four,five}; one = most expensive, five = cheapest.
        Cheaper(x, y) ::: x is cheaper (less expensive) than y.

        Premises:
        Rank(watermelon, one) :::Watermelons are the most expensive
        Rank(pears, three) ::: Pears are the third‑most expensive
        Rank(kiwis, four) ::: Kiwis are the second‑cheapest
        Cheaper(kiwis, plums) ::: Kiwis are cheaper than plums
        ∀F ∀P ∀Q ((Rank(F,P) ∧ Rank(F,Q)) → (P = Q)) ::: One rank per fruit
        ∀P ∀F ∀G ((Rank(F,P) ∧ Rank(G,P)) → (F = G)) ::: One fruit per rank
        Rank(mangoes, one) ∨ Rank(mangoes, two) ∨ Rank(mangoes, three) ∨ Rank(mangoes, four) ∨ Rank(mangoes, five) ::: each still‑unknown fruit occupies some rank 
        Rank(plums, one) ∨ Rank(plums, two) ∨ Rank(plums, three) ∨ Rank(plums, four) ∨ Rank(plums, five) ::: each still‑unknown fruit occupies some rank 
        ∀X ∀Y (Rank(X, one) ∧ Rank(Y, two) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, one) ∧ Rank(Y, three) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, one) ∧ Rank(Y, four) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, one) ∧ Rank(Y, five) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, two) ∧ Rank(Y, three) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, two) ∧ Rank(Y, four) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, two) ∧ Rank(Y, five) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, three) ∧ Rank(Y, four) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, three) ∧ Rank(Y, five) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Rank(X, four) ∧ Rank(Y, five) → Cheaper(Y, X)) ::: "higher rank → more expensive" (10 ordered pairs) 
        ∀X ∀Y (Cheaper(X, Y) → ¬Cheaper(Y, X)) ::: "cheaper" is asymmetric

        Conclusion:
        Rank(mangoes, three) ::: Option A
        Rank(kiwis, three) ::: Option B
        Rank(plums, three) ::: Option C
        Rank(pears, three) ::: Option D
        Rank(watermelon, three) ::: Option E
        ---------------
        Return **only** Predicates, Premises, and Conclusion.
        Do not include unnecessary symbols such as "**", "numbered lists (like 1., 2., 3., 4.)", " - " etc.
        """
        
        elif sl == 'SAT':
            return """Task Description: You are given a problem description.
      The task is to parse the problem as a SAT problem using Z3 syntax, defining declarations, constraints, and options.
        1. Always include all three section headers in order: # Declarations, # Constraints, # Options
        2. Declarations must follow exact patterns:
        - name = EnumSort([items, ...])  for non-numeric items
        - name = IntSort([numbers, ...])  for numeric items
        - name = Function([types] -> [return_type])
        3. Constraints support:
        - Direct expressions with ==, !=, <=, >=, <, >, Implies(), And(), Or(), Not()
        - ForAll([var:type, ...], expr) and Exists([var:type, ...], expr)
        - Count([var:type], condition)
        - Distinct([var:type], expr)
        4. Options must use predefined functions:
        - is_valid(), is_sat(), is_unsat()
        5. Add explanation with :::
        6. Avoid:
        - Add # in any other places apart from three section headers
        - Add any other unnecessary comment or dashes

      -----------
      Example：
      context: The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a shelf, there are five books: a white book, an orange book, a yellow book, a blue book, and a red book. The yellow book is to the left of the white book. The red book is to the right of the blue book. The yellow book is to the right of the orange book. The blue book is to the right of the white book.
      question: Which of the following is true?
      options: 
            A) The white book is the second from the right.
            B) The orange book is the second from the right.
            C) The yellow book is the second from the right.
            D) The blue book is the second from the right.
            E) The red book is the second from the right.
      
      # Declarations
      objects = EnumSort([White, Orange, Yellow, Blue, Red])
      positions = IntSort([1, 2, 3, 4, 5])
      pos = Function([objects] -> [positions])

      # Constraints
      Distinct([b:objects], pos(b)) ::: Each book occupies a unique position
      pos(Yellow) < pos(White) ::: The yellow book is to the left of the white book.
      pos(Red) > pos(Blue) ::: The red book is to the right of the blue book.
      pos(Yellow) > pos(Orange) ::: The yellow book is to the right of the orange book.
      pos(Blue) > pos(White) ::: The blue book is to the right of the white book.

      # Options
      is_valid(pos(White) == 4)  ::: A) The white book is the second from the right.
      is_valid(pos(Orange) == 4)  ::: B) The orange book is the second from the right.
      is_valid(pos(Yellow) == 4)  ::: C) The yellow book is the second from the right.
      is_valid(pos(Blue) == 4)  ::: D) The blue book is the second from the right.
      is_valid(pos(Red) == 4)  ::: E) The red book is the second from the right.
      ---------------
      Return **only** # Declarations, # Constraints, # Options.
      Do not include unnecessary symbols such as "**", "numbered lists (like 1., 2., 3., 4.)", " - " etc.
      """
    
    return ""


# COMPLETE PROMPT TEMPLATE
def build_complete_prompt_template(item: Dict) -> str:
    """
    Build complete prompt template with placeholders based on detected dataset
    
    Args:
        item: Problem item to detect dataset from
    
    Returns:
        Complete prompt template string
    """
    dataset = detect_dataset(item)
    
    if dataset == 'LogicalDeduction':
        # For LogicalDeduction, include options
        return """You are given a logic problem including a context and a question as follows:
Context: ${context}
Question: ${question}
Options: ${options}

${role_description}"""
    else:
        # For ProntoQA and ProofWriter
        return """You are given a logic problem in natural language including a context and a question as follows:
Context: ${context}
Question: ${question}

${role_description}"""


def substitute_placeholders(template: str, item: Dict, role_description: str) -> str:
    """
    Substitute placeholders in template with actual values
    
    Args:
        template: Template string with placeholders
        item: Data item with context, question, options
        role_description: Role description for the specific SL
    
    Returns:
        Filled template string
    """
    result = template
    
    context = item.get('context', '')
    result = result.replace('${context}', context)
    
    question = item.get('question', '')
    result = result.replace('${question}', question)
    
    if '${options}' in result:
        options = item.get('options', [])
        if options:
            options_text = '\n'.join(options) if isinstance(options, list) else str(options)
            result = result.replace('${options}', options_text)
    
    result = result.replace('${role_description}', role_description)
    
    return result


def translate_problem(llm_helper: LLMHelper, item: Dict, sl: str) -> str:
    """
    Translate a single problem to symbolic language using complete prompt template
    
    Args:
        llm_helper: LLM helper instance
        item: Problem item with context, question, options
        sl: Target symbolic language (LP/FOL/SAT)
    
    Returns:
        Translation in symbolic language
    """
    # Get role description and prompt template based on detected dataset
    role_description = get_role_description(sl, item)
    prompt_template = build_complete_prompt_template(item)
    
    filled_prompt = substitute_placeholders(prompt_template, item, role_description)
    
    try:
        system_prompt = "You are an expert in symbolic logic translation. Follow the instructions carefully and provide accurate translations."
        
        translation = llm_helper.generate(
            system_prompt=system_prompt,
            user_prompt=filled_prompt,
            return_json=False
        )
        cleaned_output = translation.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        cleaned_output = re.sub(r"\n-", "\n", cleaned_output) 
        cleaned_output = cleaned_output.replace("**", "")
        
        return cleaned_output.strip()
        
    except Exception as e:
        print(f"Error in translation: {e}")
        return f"Translation failed: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Step 1: Translate logic problems to selected symbolic language')
    parser.add_argument('--input_file', type=str, default='data/LogicalDeduction/dev.json',
                       help='Input JSON file path')
    parser.add_argument('--output_file', type=str, default='results/deepseek/translation/result.json',
                       help='Output JSON file path')
    parser.add_argument('--sl_selection', type=str, default='LP',
                       choices=['LP', 'FOL', 'SAT'],
                       help='Symbolic language to use for all problems (LP/FOL/SAT)')
    parser.add_argument('--openai_api_key', type=str, default='sk-81c7577b681e483dbbb2b5e466db3688',
                       help='OpenAI API key')
    parser.add_argument('--openai_base_url', type=str, default='https://api.deepseek.com/v1',
                       help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default='deepseek-chat', # deepseek-chat, gpt-4-0613
                       help='Model name')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for generation (default: 0.0)')
    parser.add_argument('--max_tokens', type=int, default=4000,
                       help='Maximum tokens for generation (default: 4000)')
    
    args = parser.parse_args()
    
    llm_helper = LLMHelper(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} problems")
    print(f"Using symbolic language: {args.sl_selection}")

    results = []
    translation_stats = {'success': 0, 'failed': 0}

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)

    for item in tqdm(data, desc="Translating problems"):
        # Add the selected SL to the item
        sl = args.sl_selection

        translation = translate_problem(llm_helper, item, sl)

        result_item = item.copy()
        result_item['SL'] = sl  # Add SL field for compatibility with downstream steps
        result_item['translation'] = {sl: translation}

        if "Translation failed" in translation:
            translation_stats['failed'] += 1
        else:
            translation_stats['success'] += 1

        results.append(result_item)
        
        # Save results after each sample (overwrite the file)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinal save to {args.output_file}...")
    save_data(args.output_file, results)
    
    print("\n=== Translation Statistics ===")
    print(f"Total problems processed: {len(results)}")
    print(f"Symbolic language used: {args.sl_selection}")
    print(f"Successful translations: {translation_stats['success']}")
    print(f"Failed translations: {translation_stats['failed']}")

    print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()