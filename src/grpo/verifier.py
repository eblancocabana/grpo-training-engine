"""
Rule-Based Verifier for GRPO.
Validates mathematical answers from generated text.
"""
import re
from typing import Tuple, Optional


class RuleBasedVerifier:
    """
    Verifies mathematical answers using deterministic rules.
    
    Extracts answers from generated text and compares with ground truth.
    Assigns binary rewards: 1.0 for correct, 0.0 for incorrect.
    """
    
    def __init__(self):
        """Initialize verifier with regex patterns."""
        # Pattern to extract content between <answer> tags
        self.answer_tag_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        
        # Pattern to extract boxed answer: \boxed{...}
        # Handles nested braces by matching balanced curly braces
        self.boxed_pattern = re.compile(r'\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}', re.DOTALL)
        
        # Pattern to extract "The answer is X" or "Final Answer: X"
        self.final_answer_pattern = re.compile(
            r'(?:the answer is|final answer:?|answer:?|####|therefore.*?is|thus.*?is)\s*\$?([\d,\.]+)',
            re.IGNORECASE
        )
        
        # Pattern to extract equation answer: "= X" or "is X" or "is equal to X"
        # Matches: = 670, is $11, is equal to 150, etc.
        self.equation_pattern = re.compile(
            r'(?:=|\bis\b|\bis\s+equal\s+to)\s*\$?(-?[\d,]+(?:\.\d+)?)',
            re.IGNORECASE
        )
        
        # Pattern to extract think block content
        # The <think> tag is in the prompt, model generates content + </think>
        # So we grab everything before the closing </think> tag
        self.think_pattern = re.compile(r'^(.*?)</think>', re.DOTALL)
        
        # Pattern to extract numbers
        # Matches numbers like 1,000.50, -5, 42
        # Must contain at least one digit
        self.number_pattern = re.compile(r'-?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?')
    
    def extract_answer_from_tags(self, text: str) -> Optional[str]:
        """
        Extract answer from <answer> tags or \\boxed{}.
        
        Args:
            text: Generated text
            
        Returns:
            Extracted answer or None
        """
        # Try <answer> tags
        matches = self.answer_tag_pattern.findall(text)
        if matches:
            return matches[-1].strip()
            
        # Try \boxed{}
        matches = self.boxed_pattern.findall(text)
        if matches:
            boxed_content = matches[-1].strip()
            # Extract the first number from boxed content
            # Boxed content may contain LaTeX like "1250 \text{ fluid ounces}"
            number_matches = self.number_pattern.findall(boxed_content)
            if number_matches:
                return number_matches[0]
            return boxed_content
            
        return None
    
    def extract_equation_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from equation patterns like "= X" or "is X".
        
        Args:
            text: Generated text
            
        Returns:
            Extracted number from equation or None
        """
        matches = self.equation_pattern.findall(text)
        if matches:
            # Return the last (most recent) equation result
            return matches[-1].strip()
        return None
    
    def extract_answer_from_think(self, text: str) -> Optional[str]:
        """
        Extract answer from <think> block using equation patterns.
        
        Args:
            text: Generated text
            
        Returns:
            Extracted answer from think block or None
        """
        think_matches = self.think_pattern.findall(text)
        if think_matches:
            think_block = think_matches[-1]  # Last think block
            # Look for equation patterns inside think block
            equation_result = self.extract_equation_answer(think_block)
            if equation_result:
                return equation_result
        return None
    
    def extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extract final answer using cascading priority.

        Priority order (FIXED based on log analysis):
        1. Tags (<answer>, \\boxed{}) - Most explicit
        2. LAST number in <think> block - Think block has complete reasoning
        3. "Answer is X" patterns - Explicit statements
        4. LAST number in last 150 chars - If no think block
        5. Equation patterns (= X) - Last resort

        Analysis showed 66% failure rate because:
        - Model truncates final output before completing answer
        - Think block contains complete reasoning with final number
        - We were grabbing intermediate calculations (33, 66) instead of final answers (121)

        Args:
            text: Generated text

        Returns:
            Extracted answer or None
        """
        # Priority 1: Try <answer> tags and \boxed{}
        answer = self.extract_answer_from_tags(text)
        if answer:
            return answer

        # Priority 2: Extract answer from <think> block
        # When the model performs verification ("let me check: 42 - 10 = 32"),
        # the last number is from verification, not the answer. Try explicit
        # answer patterns first, then equation patterns, then last number.
        think_matches = self.think_pattern.findall(text)
        if think_matches:
            think_block = think_matches[-1]
            # 2a: Explicit "the answer is X" inside think block
            answer_match = self.final_answer_pattern.search(think_block)
            if answer_match:
                return answer_match.group(1).strip()
            # 2b: Equation pattern "= X" (last occurrence)
            equation_result = self.extract_equation_answer(think_block)
            if equation_result:
                return equation_result
            # 2c: Last resort — last number in think block
            numbers_in_think = self.number_pattern.findall(think_block)
            if numbers_in_think:
                return numbers_in_think[-1].strip()

        # Priority 3: Try "Final Answer:" / "The answer is" patterns
        match = self.final_answer_pattern.search(text)
        if match:
            return match.group(1).strip()

        # Priority 4: LAST number in last 150 chars of main text
        # Only if we don't have a think block
        last_section = text[-150:] if len(text) > 150 else text
        numbers = self.number_pattern.findall(last_section)
        if numbers:
            return numbers[-1].strip()

        # Priority 5: Try equation patterns
        equation_answer = self.extract_equation_answer(text)
        if equation_answer:
            return equation_answer

        # Priority 6: Fall back to last number in full text
        numbers = self.number_pattern.findall(text)
        if numbers:
            return numbers[-1].strip()

        return None
    
    def normalize_number(self, text: Optional[str]) -> Optional[float]:
        """
        Normalize number text to float.
        
        Args:
            text: Number as string (may contain commas)
            
        Returns:
            Float value or None
        """
        if text is None:
            return None
        
        cleaned = text.replace(',', '').replace(' ', '')
        cleaned = cleaned.replace('\\\\$', '').replace('$', '')
        cleaned = cleaned.replace('\\\\%', '').replace('%', '')
        cleaned = cleaned.replace('\\', '').replace('!', '')
        cleaned = cleaned.rstrip('.')
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def verify(
        self,
        generated_text: str,
        ground_truth: str
    ) -> Tuple[float, dict]:
        """
        Verify if generated answer matches ground truth.
        
        Args:
            generated_text: Model-generated text
            ground_truth: Ground truth answer
            
        Returns:
            Tuple of (reward, info_dict)
            reward: 1.0 if correct, 0.0 if incorrect
            info: Dictionary with extraction details
        """
        extracted = self.extract_final_answer(generated_text)
        truth_extracted = ground_truth.strip()
        
        info = {
            'generated_text_preview': generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
            'extracted_answer': extracted,
            'ground_truth_answer': truth_extracted,
        }
        
        generated_num = self.normalize_number(extracted)
        truth_num = self.normalize_number(truth_extracted)
        
        info['generated_number'] = generated_num
        info['ground_truth_number'] = truth_num
        
        if generated_num is None or truth_num is None:
            reward = 0.0
            info['match'] = False
            info['error'] = "Could not extract valid numbers"
        elif abs(generated_num - truth_num) < 1e-3:
            reward = 1.0
            info['match'] = True
        else:
            reward = 0.0
            info['match'] = False
        
        return reward, info
    
    def verify_batch(
        self,
        generated_texts: list,
        ground_truths: list
    ) -> Tuple[list, list]:
        """
        Verify a batch of generated answers.
        
        Args:
            generated_texts: List of generated texts
            ground_truths: List of ground truth answers
            
        Returns:
            Tuple of (rewards_list, info_list)
        """
        rewards = []
        infos = []
        
        for gen, truth in zip(generated_texts, ground_truths):
            reward, info = self.verify(gen, truth)
            rewards.append(reward)
            infos.append(info)
        
        return rewards, infos

