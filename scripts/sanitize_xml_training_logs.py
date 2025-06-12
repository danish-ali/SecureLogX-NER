import json
import re

def strip_xml_tags_and_adjust(text, entities):
    """
    Remove XML tags from text and adjust entity spans accordingly.

    Parameters:
    - text (str): The original text with XML tags.
    - entities (list of dicts): List of entities with 'start', 'end', and 'label'.

    Returns:
    - tuple: Cleaned text and adjusted entities.
    """
    clean_text = ""
    adjusted_entities = []
    tag_pattern = re.compile(r"</?[^>]+>")
    current_pos = 0
    offset = 0

    tag_matches = list(tag_pattern.finditer(text))
    cursor = 0

    for match in tag_matches:
        tag_start, tag_end = match.span()
        clean_text += text[cursor:tag_start]
        offset += tag_end - tag_start
        cursor = tag_end

    clean_text += text[cursor:]

    # Adjust entity spans
    for ent in entities:
        original = text[ent["start"]:ent["end"]]
        adjusted_start = len(tag_pattern.sub("", text[:ent["start"]]))
        adjusted_end = adjusted_start + len(original)
        adjusted_entities.append({
            "start": adjusted_start,
            "end": adjusted_end,
            "label": ent["label"]
        })

    return clean_text, adjusted_entities

def sanitize_xml_logs(input_file, output_file):
    """
    Process the input JSON file to remove XML tags and adjust entity spans,
    then save the cleaned data to the output JSON file.

    Parameters:
    - input_file (str): Path to the input JSON file containing XML logs.
    - output_file (str): Path to save the cleaned JSON data.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        xml_logs = json.load(f)

    cleaned_logs = []

    for log in xml_logs:
        cleaned_text, cleaned_entities = strip_xml_tags_and_adjust(log["text"], log["entities"])
        cleaned_logs.append({
            "text": cleaned_text,
            "entities": cleaned_entities
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_logs, f, indent=2)

if __name__ == "__main__":
    input_path = "xml_ner_logs.json"  # Update with your input file path
    output_path = "xml_ner_logs_cleaned.json"  # Update with your desired output file path
    sanitize_xml_logs(input_path, output_path)
    print(f"Sanitized logs saved to {output_path}")
