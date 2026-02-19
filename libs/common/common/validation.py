import re

BLOCKLIST = {
    "sexual": [r"\bnsfw\b", r"\bexplicit\b", r"\bsex\b", r"\bnude\b"],
    "minors": [r"\bchild\b", r"\bloli\b", r"\bminor\b", r"\bunderage\b"],
    "hate": [r"\bgenocide\b", r"\bnazi\b", r"\brace war\b"],
    "self_harm": [r"\bsuicide\b", r"\bself-harm\b", r"\bcut myself\b"],
    "text_overlay": [r"\bwatermark\b", r"\blogo\b", r"\bsubtitle\b", r"\btext overlay\b", r"\bcaption\b"],
}


class PromptValidationError(ValueError):
    pass


def validate_prompt(prompt: str) -> None:
    lower = prompt.lower()
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                raise PromptValidationError(f"Prompt rejected: detected disallowed category '{category}'.")
