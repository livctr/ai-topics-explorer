from datetime import datetime
import regex as re


def _get_lbl_from_name(names):
    """Tuple (last_name, first_name, middle_name) => String 'first_name [middle_name] last_name'."""
    return [
        name[1] + ' ' + name[0] if name[2] == '' \
        else name[1] + ' ' + name[2] + ' ' + name[0]
        for name in names
    ]


def remove_unmatched(text: str, open_sym: str, close_sym: str) -> str:
    """
    Remove unmatched occurrences of open_sym and close_sym in text.
    """
    stack = []
    indices_to_remove = set()
    
    # First pass: mark unmatched closing symbols.
    for i, ch in enumerate(text):
        if ch == open_sym:
            stack.append(i)
        elif ch == close_sym:
            if stack:
                stack.pop()
            else:
                indices_to_remove.add(i)
    # Any remaining open symbols in the stack are unmatched.
    indices_to_remove.update(stack)
    
    # Build new text without the unmatched symbols.
    new_text = "".join(ch for i, ch in enumerate(text) if i not in indices_to_remove)
    return new_text

def textify(text: str) -> str:
    # 1. Replace tabs and newlines with spaces and collapse extra spaces.
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    
    # 2. Replace LaTeX commands for italics, bold, sans-serif, and math-sans with their content.
    text = re.sub(r"\\(?:textit|textbf|textsf|mathsf)\{([^}]+)\}", r"\1", text)
    
    # 3. Replace "\%" with "%" and "\times" with " times".
    text = text.replace("\\%", "%").replace("\\times", " times")
    
    # 4. Remove any remaining dollar signs.
    text = text.replace("$", "")
    
    # 5. Remove any \url{...} commands (and their contents).
    text = re.sub(r"\\url\{[^}]*\}", "", text)
    
    # 6. Remove square brackets and whatever is inside them.
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # 7. Remove unmatched parentheses and curly braces.
    text = remove_unmatched(text, "(", ")")
    text = remove_unmatched(text, "{", "}")
    
    # 8. Remove the last sentence if it includes a link (ignoring case).
    sentences = re.split(r'\.\s*', text.strip())
    if sentences:
        last_sentence = sentences[-1].strip()
        if "http" in last_sentence.lower():
            sentences = sentences[:-1]
        text = ". ".join(s for s in sentences if s)
        if text and text[-1] not in ".!?":
            text += "."
    
    return text.strip()


class EntryExtractor:

    @staticmethod
    def extract_id(entry_data):
        """Extracts the arXiv ID from the entry data."""
        return entry_data.get("id")

    @staticmethod
    def extract_submitter(entry_data):
        """Extracts the submitter from the entry data."""
        return entry_data.get("submitter")

    @staticmethod
    def extract_authors(entry_data):
        """
        Extracts and formats author names from the entry data.

        Returns:
            A list of formatted names in the format 'first_name [middle_name] last_name'.
        """
        authors_parsed = entry_data.get("authors_parsed", [])
        return _get_lbl_from_name(authors_parsed)
    
    @staticmethod
    def extract_title(entry_data, max_chars: int = 250):
        """Extracts the title from the entry data."""
        title = entry_data.get("title").strip()
        if max_chars is not None and len(title) > max_chars:
            front = title[:max_chars // 2]
            back = title[-max_chars // 2:]
            title = front + " ... " + back
        return title

    @staticmethod
    def extract_abstract(entry_data, max_chars: int = 2000):
        """Extracts the abstract from the entry data."""
        abstract = entry_data.get("abstract").strip()
        abstract = textify(abstract)
        if max_chars is not None and len(abstract) > max_chars:
            front = abstract[:max_chars // 2]
            back = abstract[-max_chars // 2:]
            abstract = front + " ... " + back
        return abstract

    @staticmethod
    def extract_num_authors(entry_data):
        """Extracts the number of authors from the entry data."""
        authors_parsed = entry_data.get("authors_parsed", [])
        return len(authors_parsed)
    
    @staticmethod
    def extract_categories(entry_data):
        """Extracts the categories from the entry data."""
        return entry_data.get("categories").split(" ")
    
    @staticmethod
    def extract_date(entry_data, first_version: bool = True):
        """Extracts the date from the entry data as a datetime"""

        versions = entry_data.get("versions", [])

        # Select the appropriate version
        version_info = versions[0] if first_version else versions[-1]
        created_str = version_info.get("created")
        if not created_str:
            return False  # No creation date available

        # Parse the date string (e.g., "Sat, 7 Apr 2007 20:23:54 GMT")
        version_date = datetime.strptime(created_str, "%a, %d %b %Y %H:%M:%S %Z")
        return version_date


class PaperFilter:

    @staticmethod
    def is_cs(entry_data):
        """Returns True if the entry is categorized under CS."""
        categories = EntryExtractor.extract_categories(entry_data)
        return any(re.match(r"cs\.[a-zA-Z]{2}", cat) for cat in categories)

    @staticmethod
    def inside_date_range(entry_data, start: datetime, end: datetime, first_version: bool = True):
        """
        Returns True if the paper was submitted between start and end dates.
        Since papers may have multiple versions, `first_version` controls
        whether we consider version 1 (True) or the last version (False).
        """
        try:
            version_date = EntryExtractor.extract_date(entry_data, first_version)
        except IndexError:
            print(f"IndexError on extracting date from {entry_data}")
            return False
        except ValueError:
            print(f"ValueError on extracting date from {entry_data}")
            return False
        # Check if the version date is within the given range
        return start <= version_date <= end
    
    @staticmethod
    def one_author_in_set(entry_data, author_set):
        """Returns True if at least one author is in the provided set."""
        authors = EntryExtractor.extract_authors(entry_data)
        return any(author in author_set for author in authors)
