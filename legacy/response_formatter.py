import logging
import re

from telegram.constants import ParseMode

# Import logging_config to ensure DEBUG level is available
import logging_config  # noqa: F401

# MarkdownV2 special characters that must be escaped when not part of formatting
MARKDOWNV2_SPECIAL_CHARS = set("_*[]()~`>#+-=|{}.!")


def _escape_special_chars(text: str) -> str:
    """Escape MarkdownV2 special characters that aren't part of formatting."""
    # Don't escape chars already escaped
    text = re.sub(r'\\(.)', r'\\\1', text)  # Avoid double-escaping

    result = []
    i = 0
    while i < len(text):
        char = text[i]

        # Check if already escaped
        if i > 0 and text[i - 1] == '\\':
            result.append(char)
            i += 1
            continue

        # Check for formatting patterns (don't escape these)
        # *text*, _text_, `code`, [link](url), etc.
        if char in MARKDOWNV2_SPECIAL_CHARS:
            # Look ahead to see if this is part of valid formatting
            is_formatting = False

            if char == '*' and i + 1 < len(text) and text[i + 1] != '*':
                is_formatting = True  # Single * for bold
            elif char == '_' and i + 1 < len(text) and text[i + 1] != '_':
                is_formatting = True  # Single _ for italic
            elif char == '`':
                is_formatting = True  # Backtick for code
            elif char == '[':
                is_formatting = True  # Link opening
            elif char == ']' and i + 1 < len(text) and text[i + 1] == '(':
                is_formatting = True  # Link closing

            if not is_formatting:
                result.append('\\')

        result.append(char)
        i += 1

    return ''.join(result)


def normalize_to_markdownv2(text: str) -> str:
    """
    Convert various markdown/HTML artifacts to valid MarkdownV2.
    - Strips HTML tags, converts to MarkdownV2 equivalents
    - Fixes common markdown variations
    - Escapes special characters
    """
    if not text:
        return text

    # Convert HTML tags to MarkdownV2
    text = re.sub(r'<b>(.*?)</b>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<strong>(.*?)</strong>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<i>(.*?)</i>', r'_\1_', text, flags=re.DOTALL)
    text = re.sub(r'<em>(.*?)</em>', r'_\1_', text, flags=re.DOTALL)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
    text = re.sub(r'<a href=["\']?(.*?)["\']?>(.*?)</a>', r'[\2](\1)', text, flags=re.DOTALL)

    # Convert markdown variations to standard MarkdownV2
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)  # **bold** → *bold*
    text = re.sub(r'__(.*?)__', r'_\1_', text)      # __italic__ → _italic_

    # Escape special characters
    text = _escape_special_chars(text)

    return text


def validate_markdownv2(text: str) -> tuple[bool, str]:
    """
    Validate MarkdownV2 syntax.
    Returns: (is_valid, error_message)

    Key rules:
    - * and _ must be balanced (even count of unescaped occurrences)
    - ` must be balanced
    - [ and ] must be balanced
    - Underscores inside URLs and code blocks don't count (skip them)
    """
    if not text:
        return True, ""

    # Skip validation for text containing unbalanced code blocks or links
    # These are too complex to validate without a proper parser
    # Instead, use a simpler heuristic: if has balanced basic structure, tentatively accept

    def count_unescaped_outside_code_urls(char: str) -> int:
        """Count unescaped char, but skip those inside backticks and URLs."""
        count = 0
        i = 0
        in_code = False
        in_url = False

        while i < len(text):
            # Track if we're inside backticks (inline code)
            if text[i] == '`' and (i == 0 or text[i-1] != '\\'):
                in_code = not in_code
                i += 1
                continue

            # Track if we're inside parentheses (URLs in links)
            if text[i] == '(' and (i == 0 or text[i-1] != '\\') and not in_code:
                in_url = True
                i += 1
                continue
            if text[i] == ')' and (i == 0 or text[i-1] != '\\') and not in_code and in_url:
                in_url = False
                i += 1
                continue

            # Only count char if not in code or URL
            if text[i] == char and not in_code and not in_url:
                # Check if it's escaped (preceded by odd number of backslashes)
                num_backslashes = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    num_backslashes += 1
                    j -= 1
                # If even number of backslashes (including 0), the char is unescaped
                if num_backslashes % 2 == 0:
                    count += 1
            i += 1
        return count

    # Check balanced asterisks (bold)
    asterisk_count = count_unescaped_outside_code_urls('*')
    if asterisk_count % 2 != 0:
        return False, f"Unbalanced asterisks (*) — found {asterisk_count} unescaped"

    # Check balanced underscores (italic)
    underscore_count = count_unescaped_outside_code_urls('_')
    if underscore_count % 2 != 0:
        return False, f"Unbalanced underscores (_) — found {underscore_count} unescaped"

    # Check balanced backticks (code)
    backtick_count = count_unescaped_outside_code_urls('`')
    if backtick_count % 2 != 0:
        return False, f"Unbalanced backticks (`) — found {backtick_count} unescaped"

    # Check for balanced brackets in links
    bracket_count = count_unescaped_outside_code_urls('[')
    close_bracket_count = count_unescaped_outside_code_urls(']')
    if bracket_count != close_bracket_count:
        return False, f"Unbalanced brackets — [{bracket_count} vs ]{close_bracket_count}"

    return True, ""


async def format_response(bot_text: str) -> tuple[str, str]:
    """
    Format and validate bot response for Telegram.

    Returns:
        (formatted_text, parse_mode) where parse_mode is either MARKDOWN_V2 or None (plain text)

    Strategy:
    1. Normalize to MarkdownV2 (convert HTML, fix markdown variations)
    2. Validate syntax
    3. If invalid, fall back to plain text (safest option)
    4. Log any conversions/fallbacks
    """
    if not bot_text:
        return "", None

    logging.info(f"[FORMAT] Input bot_text (len={len(bot_text)})")
    logging.debug(f"[FORMAT] Raw input:\n{bot_text}")

    try:
        # Step 1: Normalize
        normalized = normalize_to_markdownv2(bot_text)
        logging.info(f"[FORMAT] After normalize (len={len(normalized)})")
        if len(normalized) != len(bot_text):
            logging.debug(f"[FORMAT] Normalized text:\n{normalized}")

        # Step 2: Validate
        is_valid, error_msg = validate_markdownv2(normalized)
        logging.info(f"[FORMAT] Validation result: is_valid={is_valid}, error='{error_msg}'")

        if is_valid:
            logging.info(f"[FORMAT] Sending as MarkdownV2 (len={len(normalized)})")
            return normalized, ParseMode.MARKDOWN_V2
        else:
            logging.warning(f"[FORMAT] MarkdownV2 validation failed: {error_msg}")
            # Strip all markdown/HTML for plain text fallback
            plain = re.sub(r'[*_`\[\]()]', '', bot_text)
            logging.info(f"[FORMAT] Fallback to plain text (len={len(plain)})")
            logging.debug(f"[FORMAT] Plain text:\n{plain}")
            return plain, None

    except Exception as e:
        logging.error(f"[FORMAT] Exception: {e}", exc_info=True)
        # Fail safe: return plain text
        plain = re.sub(r'[*_`\[\]()]', '', bot_text)
        logging.info(f"[FORMAT] Fallback to plain text due to exception (len={len(plain)})")
        return plain, None
