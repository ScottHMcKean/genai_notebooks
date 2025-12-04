"""DsPy Signatures for Information Extraction"""

import dspy


class QueryRewrite(dspy.Signature):
    """Rewrite user query for better retrieval results"""

    query = dspy.InputField(desc="Original user query")
    rewritten_query = dspy.OutputField(desc="Enhanced query optimized for search")


class GenerateSummary(dspy.Signature):
    """Generate a concise, Google-like summary from context"""

    query = dspy.InputField(desc="User's information request")
    context = dspy.InputField(desc="Retrieved documents and passages")
    summary = dspy.OutputField(desc="Concise 2-3 sentence summary answering the query")


class ExtractKeyPoints(dspy.Signature):
    """Extract key points from context"""

    query = dspy.InputField(desc="User's information request")
    context = dspy.InputField(desc="Retrieved documents and passages")
    key_points = dspy.OutputField(desc="List of 3-5 key points as bullet points")


class ExtractEntities(dspy.Signature):
    """Extract structured entities from context"""

    context = dspy.InputField(desc="Retrieved documents and passages")
    entities_json = dspy.OutputField(
        desc="JSON array of entities with name, type, value, and confidence fields"
    )

