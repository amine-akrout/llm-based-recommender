"""
This module contains utility functions for the recommender service.
"""

from langchain_community.query_constructors.chroma import (
    ChromaTranslator as BaseChromaTranslator,
)
from langchain_core.structured_query import Comparator, Comparison


class CustomChromaTranslator(BaseChromaTranslator):
    def __init__(self):
        super().__init__()
        # `allowed_comparators` is a list; convert to set, then add `LIKE`, then back to list
        if self.allowed_comparators is None:
            self.allowed_comparators = []
        allowed_comparator_set = set(self.allowed_comparators)
        allowed_comparator_set.add(Comparator.LIKE)
        self.allowed_comparators = list(allowed_comparator_set)

    def visit_comparison(self, comparison: Comparison):
        """
        Chroma does NOT allow '$contains' or substring filtering out-of-the-box.
        We'll interpret `LIKE(attribute, value)` as an array-membership check:
            {"attribute": {"$in": [value]}}
        For this to work, your metadata must store `attribute` as a list of items.
        """
        if comparison.comparator == Comparator.LIKE:
            # Use $in to check if 'comparison.value' is in the attribute's list.
            return {comparison.attribute: {"$in": [comparison.value]}}
        # Otherwise, do default logic for eq, gt, gte, lt, lte, etc.
        return super().visit_comparison(comparison)


ATTRIBUTE_INFO = [
    {
        "name": "Product Details",
        "description": "Details about the product",
    },
    {
        "name": "Brand Name",
        "description": "Name of the brand",
    },
    {
        "name": "Available Sizes",
        "description": (
            "Sizes available for the product (stored as a comma-separated string, e.g., 'small, medium, large'). "
            "Use the `like` operator to check if a size is included. "
            'Example: `like("Available Sizes", "xl")` to find products that have XL in their size options.'
        ),
    },
    {
        "name": "Product Price",
        "description": "Price of the product. Use `lt`, `lte`, `gt`, or `gte` for filtering.",
    },
]

DOC_CONTENT = "A detailed description of an e-commerce product, including its features, benefits, and specifications."


def get_metadata_info():
    return ATTRIBUTE_INFO, DOC_CONTENT
