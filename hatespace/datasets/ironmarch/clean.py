# TODO document this module

import html
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import html2text

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

html_parser = html2text.HTML2Text()
html_parser.body_width = 0
html_parser.ignore_links = False
html_parser.ignore_images = False


def html2text(text: str) -> str:
    return html_parser.handle(text)


URL_TOKEN = "<URL>{}"
EMAIL_TOKEN = "<EMAIL>{}"

SPACELIKE_CHARACTERS = [
    "\u0020",
    "\u00a0",
    "\u1680",
    "\u180e",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u200b",
    "\u202f",
    "\u205f",
    "\u3000",
    "\ufeff",
]

match_email = re.compile(r"([\w\-\+.\"]+@[\w-]+.[\w-]+)", re.MULTILINE)
match_web_url = re.compile(
    r"""<?((?:ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9\u00a1-\uffff0\-\./(%)_]{2,}\.[a-zA-Z0-9\u00a1-\uffff0\-\./(%)_?=#!+&;]{2,}[a-zA-Z0-9\u00a1-\uffff0\-\./(%)_]*)>?""",
    re.MULTILINE,
)
match_web_protocol = re.compile(r"^(https?|fttp)")
match_markdown_image = re.compile(r"!\[([\s\S]*?)\]\([\s\S]*?\)", re.MULTILINE)
match_mailto_link = re.compile(r"(?<!!)\[([\s\S]*?)\]\(mailto:[\s\S]*?\)", re.MULTILINE)
match_any_link = re.compile(r"(?<!!)\[([\s\S]*?)\]\(([\s\S]*?)\)", re.MULTILINE)
match_duplicate_newlines = re.compile(r"[\s\>]{1,}\n", re.MULTILINE)
match_numbered_item = re.compile(r"^[0-9][ \.)]*[\.)]", re.MULTILINE)
match_multiple_spaces = re.compile(r"[ ]+", re.MULTILINE)
match_surrounded_by_colons = re.compile(r"^:.+:$", re.MULTILINE)


def convert_url_to_token(url: str) -> str:
    if not re.match(match_web_protocol, url):
        url = "http://" + url
    net_location = urlparse(url).netloc.split(".")
    if len(net_location) < 3:
        domain = net_location[0]
    else:
        domain = ".".join(net_location[1:-1])
    return URL_TOKEN.format(domain)


def convert_email_to_token(email: str) -> str:
    return EMAIL_TOKEN.format(email.split("@")[0])


def convert_urls_to_tokens(string: str) -> str:
    def match_function(match: re.Match) -> str:
        return convert_url_to_token(match.group(1))

    return re.sub(match_web_url, match_function, string)


def convert_emails_to_tokens(string: str) -> str:
    def match_function(match: re.Match) -> str:
        return convert_email_to_token(match.group(1))

    return re.sub(match_email, match_function, string)


def replace_images_with_alt_text(string: str) -> str:
    def remove_extra_characters(match: re.Match) -> str:
        output = match.group(1).replace("\)", ")")
        if re.match(match_surrounded_by_colons, output):
            output = output[1:-1]
        return output

    return re.sub(match_markdown_image, remove_extra_characters, string)


def remove_text_hyperlink_annotation(string: str) -> str:
    def web_link_match_function(match: re.Match) -> str:
        return match.group(1) + " " + convert_url_to_token(match.group(2))

    def email_link_match_function(match: re.Match) -> str:
        return convert_email_to_token(match.group(1))

    output_string = re.sub(match_mailto_link, email_link_match_function, string)
    return re.sub(match_any_link, web_link_match_function, output_string)


def remove_duplicate_newlines(string: str) -> str:
    return re.sub(match_duplicate_newlines, "\n", string)


def remove_duplicate_internal_whitespace(string: str) -> str:
    return re.sub(match_multiple_spaces, " ", string)


def replace_spacelike_characters(string: str) -> str:
    for character in SPACELIKE_CHARACTERS:
        string = string.replace(character, " ")
    return string


def replace_numbered_items(string: str) -> str:
    return re.sub(match_numbered_item, "-", string)


CLEANING_FUNCTIONS = [
    replace_spacelike_characters,
    replace_images_with_alt_text,
    remove_text_hyperlink_annotation,
    convert_urls_to_tokens,
    convert_emails_to_tokens,
    replace_numbered_items,
    remove_duplicate_newlines,
    remove_duplicate_internal_whitespace,
]

# TODO: html2text will remove all of the newlines from within a tag. Somehow prevent this, since those newlines have conceptual organization
def format_post(post: str) -> str:
    post = post.strip()
    if post is None or post == "":
        raise ValueError(f"Post is empty")
    original_post = post
    if bool(BeautifulSoup(post, "html.parser").find()):
        post = html2text(post)
    post = html.unescape(post)
    for function in CLEANING_FUNCTIONS:
        post = function(post)
    post = post.strip()
    if post == "":
        raise ValueError(f"Post was empty after cleaning: {repr(original_post)}")
    return post


if __name__ == "__main__":
    example = """Hi Rostislav,  
danke für den Willkommensgruss. Savitri Devi habe ich natürlich im original hier. 
Falls Du gute Texte in Deutsch suchst empfehle ich Dir  
https://cernunninsel.wordpress.com/  
&gt;tfw you missed that you got 1488 likes
Heil Dir!"""
    print(format_post(example))
