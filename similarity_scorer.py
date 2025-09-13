 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolated similarity scoring algorithm from algo.py

This script contains the similarity scoring functions that calculate:
1. Jaccard similarity (based on 2-character n-grams)
2. Jaro-Winkler similarity 
3. Acronym similarity score
4. Overall weighted similarity score

Usage:
    from similarity_scorer import SimilarityScorer
    
    scorer = SimilarityScorer()
    results = scorer.calculate_similarity("信息科学与技术学院", "信息科学技术学院")
    print(results)
"""

import re
from typing import Dict, List, Tuple, NamedTuple


class SimilarityResult(NamedTuple):
    """Container for similarity calculation results"""
    jaccard_score: float
    jaro_winkler_score: float
    acronym_score: float
    overall_score: float


class SimilarityScorer:
    """Isolated similarity scoring algorithm"""
    
    def __init__(self):
        # Text preprocessing constants
        self.STOP_ORG_SUFFIX = [
            "学院", "书院", "研究院", "系", "部", "所", "中心", "办公室", "实验教学中心",
            "团委", "校区", "分校", "分院", "院", "学部", "学院（筹）"
        ]
        
        self.BRACKETS_PATTERN = re.compile(r"[\(（][^\)）]*[\)）]")
        
        self.NOISE_PATTERNS = [
            re.compile(p) for p in [
                r"(?:某某大学|某大学|本校|我校)",
                r"大学",
                r"校区.*$",
            ]
        ]
        
        self.CORE_TOKENS_PRIORITY = [
            "信息", "科学", "技术", "计算机", "软件", "网络", "人工智能", "数据",
            "经济", "管理", "金融", "会计", "统计",
            "数学", "物理", "化学", "生物", "材料", "土木", "建筑",
            "医", "药", "法", "文", "史", "哲", "教育", "外国语", "外语",
        ]
        
        self.SYNONYM_MAP = {
            "信科": "信息科学与技术",
            "计科": "计算机科学与技术",
            "计软": "计算机",
            "经管": "经济管理",
            "外院": "外国语",
            "马院": "马克思主义",
        }

    def normalize_text(self, s: str) -> str:
        """Normalize text by removing noise and standardizing format"""
        if s is None:
            return ""
        s = str(s).strip()
        if not s:
            return s
        
        # Remove content in brackets
        s = self.BRACKETS_PATTERN.sub("", s)
        
        # Synonym replacement
        for k, v in self.SYNONYM_MAP.items():
            s = s.replace(k, v)
        
        # Remove noise patterns
        for pat in self.NOISE_PATTERNS:
            s = pat.sub("", s)
        
        # Remove organizational suffixes
        for suf in self.STOP_ORG_SUFFIX:
            if s.endswith(suf):
                s = s[: -len(suf)]
        
        # Remove whitespace and punctuation
        s = re.sub(r"[\s\u3000\t\r\n]+", "", s)
        s = re.sub(r"[·•·\.\-/\\_]+", "", s)
        
        # Remove connecting words
        s = re.sub(r"[与和及]", "", s)
        return s

    def core_tokens(self, s: str) -> List[str]:
        """Extract core tokens from text using longest match on priority words"""
        s = self.normalize_text(s)
        if not s:
            return []
        
        tokens: List[Tuple[int, str]] = []
        used = [False] * len(s)
        
        # Longest match on core tokens
        for w in sorted(self.CORE_TOKENS_PRIORITY, key=len, reverse=True):
            start = 0
            while True:
                i = s.find(w, start)
                if i == -1:
                    break
                if not any(used[i : i + len(w)]):
                    tokens.append((i, w))
                    for k in range(i, i + len(w)):
                        used[k] = True
                start = i + 1
        
        # Add remaining characters
        for i, ch in enumerate(s):
            if not used[i]:
                tokens.append((i, ch))
        
        tokens.sort(key=lambda x: x[0])
        
        # Remove duplicates while preserving order
        seen = set()
        ordered = []
        for _, t in tokens:
            if t not in seen:
                ordered.append(t)
                seen.add(t)
        return ordered

    def acronym(self, s: str) -> str:
        """Generate acronym from core tokens"""
        toks = self.core_tokens(s)
        return "".join(t[0] for t in toks) if toks else ""

    def char_ngrams(self, s: str, n: int = 2) -> List[str]:
        """Generate character n-grams"""
        s = self.normalize_text(s)
        return [s[i : i + n] for i in range(max(0, len(s) - n + 1))] if s else []

    def jaccard(self, a: List[str], b: List[str]) -> float:
        """Calculate Jaccard similarity coefficient"""
        A, B = set(a), set(b)
        if not A and not B:
            return 1.0
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def jaro_winkler(self, s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
        """Calculate Jaro-Winkler similarity"""
        s1, s2 = self.normalize_text(s1), self.normalize_text(s2)
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        match_distance = max(len1, len2) // 2 - 1
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0

        # Find matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if s2_matches[j]:
                    continue
                if s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        transpositions //= 2

        # Calculate Jaro similarity
        jaro = (
            (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
        )

        # Calculate common prefix length
        prefix = 0
        for i in range(min(max_l, len1, len2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        return jaro + prefix * p * (1 - jaro)

    def acronym_score(self, a: str, b: str) -> float:
        """Calculate acronym similarity score"""
        ac_a, ac_b = self.acronym(a), self.acronym(b)
        if not ac_a or not ac_b:
            return 0.0
        ratio = min(len(ac_a), len(ac_b)) / max(len(ac_a), len(ac_b))
        return ratio * self.jaro_winkler(ac_a, ac_b)

    def calculate_similarity(self, a: str, b: str) -> SimilarityResult:
        """
        Calculate all similarity scores between two strings
        
        Args:
            a: First string to compare
            b: Second string to compare
            
        Returns:
            SimilarityResult containing individual and overall scores
        """
        # Calculate individual similarity scores
        jac = self.jaccard(self.char_ngrams(a, 2), self.char_ngrams(b, 2))
        jw = self.jaro_winkler(a, b)
        ac = self.acronym_score(a, b)
        
        # Calculate weighted overall score (weights from original algo.py)
        overall = 0 * jac + 0.9 * jw + 0.1 * ac
        
        return SimilarityResult(
            jaccard_score=jac,
            jaro_winkler_score=jw,
            acronym_score=ac,
            overall_score=overall
        )

    def similarity(self, a: str, b: str) -> float:
        """
        Calculate overall similarity score (for compatibility with original function)
        
        Args:
            a: First string to compare
            b: Second string to compare
            
        Returns:
            Overall weighted similarity score
        """
        return self.calculate_similarity(a, b).overall_score


def main():
    """Command-line interface for the SimilarityScorer"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 similarity_scorer.py \"string1\" \"string2\" [\"string3\" ...]")
        print("\nCompares similarity between pairs of strings.")
        print("If more than 2 strings provided, compares all possible pairs.")
        print("\nExample:")
        print("  python3 similarity_scorer.py \"机械电子工程\" \"机械工程及自动化\"")
        print("  python3 similarity_scorer.py \"信息学院\" \"计算机学院\" \"软件学院\"")
        sys.exit(1)
    
    scorer = SimilarityScorer()
    strings = sys.argv[1:]  # Get all strings after script name
    
    if len(strings) == 2:
        # Simple case: compare two strings
        s1, s2 = strings
        print(f"Comparing: \"{s1}\" vs \"{s2}\"")
        print("=" * 60)
        print(f"Normalized: \"{scorer.normalize_text(s1)}\" vs \"{scorer.normalize_text(s2)}\"")
        print()
        
        result = scorer.calculate_similarity(s1, s2)
        print("Similarity Scores:")
        print(f"  Jaccard (2-grams):  {result.jaccard_score:.3f}")
        print(f"  Jaro-Winkler:       {result.jaro_winkler_score:.3f}")
        print(f"  Acronym:            {result.acronym_score:.3f}")
        print(f"  Overall (weighted): {result.overall_score:.3f}")
        
    else:
        # Multiple strings: compare all pairs
        print(f"Comparing {len(strings)} strings - all pairs:")
        print("=" * 80)
        
        # Header
        print(f"{'String 1':<25} {'String 2':<25} {'Jaccard':<8} {'Jaro-W':<8} {'Acronym':<8} {'Overall':<8}")
        print("-" * 80)
        
        for i in range(len(strings)):
            for j in range(i + 1, len(strings)):
                s1, s2 = strings[i], strings[j]
                result = scorer.calculate_similarity(s1, s2)
                
                # Truncate long strings for display
                s1_display = s1[:22] + "..." if len(s1) > 25 else s1
                s2_display = s2[:22] + "..." if len(s2) > 25 else s2
                
                print(f"{s1_display:<25} {s2_display:<25} {result.jaccard_score:<8.3f} {result.jaro_winkler_score:<8.3f} {result.acronym_score:<8.3f} {result.overall_score:<8.3f}")


if __name__ == "__main__":
    main()
