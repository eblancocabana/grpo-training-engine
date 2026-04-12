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
            r'(?:=|\bequals\b|\bequal\s+to\b|\bis\b|\bis\s+equal\s+to)\s*\$?(-?[\d,]+(?:\.\d+)?)',
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

        # Common LaTeX formatting noise inside answers
        self.latex_spacing_pattern = re.compile(r'\\[,!;:\s]+')
        self.latex_text_pattern = re.compile(r'\\text\{([^{}]*)\}')
        self.sentence_split_pattern = re.compile(r'(?<=[.!?])\s+|\n+')
        self.incomplete_trailing_pattern = re.compile(
            r'(?:\\boxed\{[^}]*|\\frac\{[^}]*|\\frac\{[^}]*\}\{[^}]*|=|\*\*|\\\[$)\s*$'
        )

    def _extract_last_match(self, pattern: re.Pattern, text: str) -> Optional[str]:
        """Return the last capture group match from a regex pattern."""
        matches = pattern.findall(text)
        if not matches:
            return None
        last_match = matches[-1]
        if isinstance(last_match, tuple):
            last_match = last_match[-1]
        return str(last_match).strip()

    def _clean_answer_text(self, text: str) -> str:
        """Remove lightweight LaTeX formatting noise before number extraction."""
        cleaned = self.latex_text_pattern.sub(r' \1 ', text)
        cleaned = self.latex_spacing_pattern.sub('', cleaned)
        cleaned = cleaned.replace('\\$', '$').replace('\\%', '%')
        cleaned = cleaned.replace('{', '').replace('}', '')
        return cleaned.strip()

    def _extract_number_from_text(self, text: str) -> Optional[str]:
        """Extract the last number from lightly cleaned text."""
        cleaned = self._clean_answer_text(text)
        matches = self.number_pattern.findall(cleaned)
        if matches:
            return matches[-1].strip()
        normalized = self.normalize_number(cleaned)
        if normalized is not None:
            if float(normalized).is_integer():
                return str(int(normalized))
            return str(normalized)
        return None

    def _looks_incomplete(self, text: str) -> bool:
        """Heuristic guard against truncated completions being mis-scored."""
        stripped = text.rstrip()
        if not stripped:
            return False
        return bool(self.incomplete_trailing_pattern.search(stripped))

    def _tail_segments(self, text: str, max_segments: int = 1) -> list[str]:
        """Return the last few non-empty sentence-like chunks from text."""
        segments = [
            segment.strip()
            for segment in self.sentence_split_pattern.split(text.strip())
            if segment and segment.strip()
        ]
        if max_segments <= 0:
            return []
        return segments[-max_segments:]

    def _extract_tail_equation_answer(self, text: str) -> Optional[str]:
        """
        Extract a high-confidence equation result from the tail of a reasoning block.

        This is intentionally narrow: only the final sentence-like chunk
        are searched. Training reward should not mine arbitrary intermediate numbers
        from the body of the chain of thought.
        """
        for segment in reversed(self._tail_segments(text)):
            answer = self._extract_last_match(self.final_answer_pattern, segment)
            if answer:
                return answer

            equation_answer = self.extract_equation_answer(segment)
            if equation_answer:
                return equation_answer
        return None
    
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
            number_match = self._extract_number_from_text(boxed_content)
            if number_match:
                return number_match
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

        Priority order:
        1. Tags (<answer>, \\boxed{}) - Most explicit
        2. Explicit final-answer markers outside </think>
        3. High-confidence explicit/equation answers near the end of <think>
        4. Explicit final-answer markers anywhere
        5. Last number in the trailing section when no think block exists
        6. Equation patterns in full text as a final fallback

        The verifier intentionally avoids "last random number in <think>" because
        that mis-scores truncated or partially completed responses by rewarding
        intermediate quantities as final answers.

        Args:
            text: Generated text

        Returns:
            Extracted answer or None
        """
        # Priority 1: Try <answer> tags and \boxed{}
        answer = self.extract_answer_from_tags(text)
        if answer:
            return answer

        # Priority 2: Prefer an explicit final answer outside the closing think block.
        # This prevents verification arithmetic inside <think> from outranking the
        # actual final answer stated after reasoning.
        post_think_text = text.rsplit("</think>", 1)[1] if "</think>" in text else text
        answer = self._extract_last_match(self.final_answer_pattern, post_think_text)
        if answer:
            return answer

        # Priority 3: Extract answer from <think> block
        # Avoid scanning the full chain of thought for arbitrary numbers. Use only
        # explicit answer markers or equation-style results near the end.
        think_matches = self.think_pattern.findall(text)
        if think_matches:
            think_block = think_matches[-1]
            answer = self._extract_last_match(self.final_answer_pattern, think_block)
            if answer:
                return answer

            tail_window = think_block[-200:] if len(think_block) > 200 else think_block
            equation_result = self.extract_equation_answer(tail_window)
            if equation_result:
                return equation_result

            if self._looks_incomplete(text):
                return None

        # Priority 4: Try "Final Answer:" / "The answer is" patterns anywhere.
        answer = self._extract_last_match(self.final_answer_pattern, text)
        if answer:
            return answer

        # Priority 5: LAST number in last 150 chars of main text
        # Only if we don't have a think block
        if not think_matches:
            last_section = text[-150:] if len(text) > 150 else text
            number = self._extract_number_from_text(last_section)
            if number:
                return number

        # Priority 6: Try equation patterns
        equation_answer = self.extract_equation_answer(text)
        if equation_answer:
            return equation_answer

        return None

    def extract_training_answer(self, text: str) -> Optional[str]:
        """
        Extract an answer conservatively for training reward.

        The training path prioritizes precision over recall: a missing reward is
        safer than rewarding a random intermediate number from a partially finished
        chain of thought.
        """
        answer = self.extract_answer_from_tags(text)
        if answer:
            return answer

        post_think_text = text.rsplit("</think>", 1)[1] if "</think>" in text else text
        answer = self._extract_last_match(self.final_answer_pattern, post_think_text)
        if answer:
            return answer

        think_matches = self.think_pattern.findall(text)
        if think_matches:
            think_block = think_matches[-1]
            answer = self._extract_tail_equation_answer(think_block)
            if answer:
                return answer
            return None

        if self._looks_incomplete(text):
            return None

        answer = self._extract_last_match(self.final_answer_pattern, text)
        if answer:
            return answer

        return self._extract_tail_equation_answer(text)
    
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
        extracted = self.extract_training_answer(generated_text)
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
