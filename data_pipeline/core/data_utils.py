from datetime import datetime
import regex as re


def _get_lbl_from_name(names):
    """Tuple (last_name, first_name, middle_name) => String 'first_name [middle_name] last_name'."""
    return [
        name[1] + ' ' + name[0] if name[2] == '' \
        else name[1] + ' ' + name[2] + ' ' + name[0]
        for name in names
    ]


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
    