import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from cat.mad_hatter.decorators import tool, hook

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Add phrases to matcher
matcher = PhraseMatcher(nlp.vocab)

@tool()
def extract_important_terms(tool_input, cat):
    """Extracts the most important verbs, nouns, and phrases from the provided text."""
    text = tool_input["text"]

    # Process the text with Spacy
    doc = nlp(text)

    # Initialize lists to store important verbs, nouns, and phrases
    important_verbs_list = []
    important_nouns_list = []
    important_phrases_list = []

    # Iterate through the tokens in the document
    for token in doc:
        # Check if token is a verb
        if token.pos_ == "VERB":
            important_verbs_list.append(token.text)

        # Check if token is a noun
        elif token.pos_ == "NOUN":
            important_nouns_list.append(token.text)

    # Use phrase matcher to find important phrases
    matches = matcher(doc)
    for match_id, start, end in matches:
        phrase = Span(doc, start, end)
        important_phrases_list.append(phrase.text)

    return {
        "important_verbs": important_verbs_list,
        "important_nouns": important_nouns_list,
        "important_phrases": important_phrases_list
    }

@hook
def analyze_text(hook_input, cat):
    """Analyzes the text and extracts important terms."""
    text = hook_input["text"]

    # Extract important terms using the tool
    extracted_terms = extract_important_terms({"text": text})

    # Log the extracted terms
    cat.log.info("Important Verbs:", extracted_terms["important_verbs"])
    cat.log.info("Important Nouns:", extracted_terms["important_nouns"])
    cat.log.info("Important Phrases:", extracted_terms["important_phrases"])

    # Return the extracted terms
    return extracted_terms
