import logging
import re

from telegram.constants import ParseMode

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
    """
    if not text:
        return True, ""

    # Check for unescaped special characters (basic check)
    # This is a simplified validation; Telegram's parser is more lenient

    # Check for balanced brackets in links
    link_pattern = r'\[.*?\]\(.*?\)'
    links = re.findall(link_pattern, text)
    for link in links:
        if link.count('[') != link.count(']'):
            return False, f"Unbalanced brackets in link: {link}"
        if link.count('(') != link.count(')'):
            return False, f"Unbalanced parentheses in link: {link}"

    # Check for properly formatted code blocks (backticks)
    backtick_count = text.count('`') - text.count(r'\`')
    if backtick_count % 2 != 0:
        return False, "Unbalanced backticks"

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

    try:
        # Step 1: Normalize
        normalized = normalize_to_markdownv2(bot_text)

        # Step 2: Validate
        is_valid, error_msg = validate_markdownv2(normalized)

        if is_valid:
            logging.info("Response formatted as MarkdownV2 (valid)")
            return normalized, ParseMode.MARKDOWN_V2
        else:
            logging.warning(f"MarkdownV2 validation failed: {error_msg} — falling back to plain text")
            # Strip all markdown/HTML for plain text fallback
            plain = re.sub(r'[*_`\[\]()]', '', bot_text)
            return plain, None

    except Exception as e:
        logging.error(f"Response formatting error: {e} — falling back to plain text")
        # Fail safe: return plain text
        plain = re.sub(r'[*_`\[\]()]', '', bot_text)
        return plain, None
