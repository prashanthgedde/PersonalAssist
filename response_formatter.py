import logging
import re

logger = logging.getLogger(__name__)

TELEGRAM_HTML_TAGS = ["b", "i", "u", "code", "pre", "a"]


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_to_html(text: str) -> str:
    """
    Convert Markdown to Telegram-compatible HTML.
    Limited to: <b>, <i>, <code>, <pre>, <a>
    """
    result = text

    result = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r'<a href="\2">\1</a>', result)

    result = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", result)
    result = re.sub(r"__([^_]+)__", r"<b>\1</b>", result)
    result = re.sub(r"_([^_]+)_", r"<i>\1</i>", result)

    result = re.sub(r"`([^`]+)`", r"<code>\1</code>", result)

    result = re.sub(r"```(\w*)\n?([\s\S]*?)```", r"<pre>\2</pre>", result)

    return result


def format_as_html(text: str, sources: list[dict] = None, metadata: dict = None) -> str:
    """
    Format the response as Telegram-compatible HTML.

    Args:
        text: The main response text
        sources: List of {"title": str, "url": str} dicts
        metadata: Dict with tool_calls, total_latency_ms, etc.

    Returns:
        HTML-formatted string
    """
    html_text = markdown_to_html(text)

    parts = [html_text]

    if sources:
        parts.append("")
        parts.append("<b>Sources:</b>")
        for source in sources:
            title = escape_html(source.get("title", "Untitled"))
            url = escape_html(source.get("url", ""))
            if url:
                parts.append(f'• <a href="{url}">{title}</a>')
            else:
                parts.append(f"• {title}")

    if metadata:
        meta_parts = []

        tool_calls = metadata.get("tool_calls", 0)
        if tool_calls > 0:
            meta_parts.append(f"Tools: {tool_calls}")

        total_latency = metadata.get("total_latency_ms", 0)
        if total_latency > 0:
            meta_parts.append(f"Time: {total_latency / 1000:.1f}s")

        iteration_count = metadata.get("iteration_count", 0)
        if iteration_count > 0:
            meta_parts.append(f"Iterations: {iteration_count}")

        if meta_parts:
            parts.append("")
            parts.append(f"<i>{' | '.join(meta_parts)}</i>")

    return "\n".join(parts)


def format_response(
    bot_text: str, sources: list = None, metadata: dict = None
) -> tuple[str, str]:
    """
    Main entry point - returns (formatted_text, parse_mode).
    For HTML, returns (html_content, "HTML").
    """
    html_content = format_as_html(bot_text, sources, metadata)
    return html_content, "HTML"
