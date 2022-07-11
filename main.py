#!/usr/bin/env python

"""
CLI for the link-prediction package
"""

import click
import json
import joblib
from link_prediction.datasets.rico import RicoDataPoint, ViewHierarchy


@click.command()
@click.option(
    "-s",
    "--source-screen",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    prompt="Path to source screen's view hierarchy",
    help="The path to the source screen's view hierarchy (refer to RICO dataset).",
)
@click.option(
    "-e",
    "--source-element-id",
    type=str,
    required=True,
    prompt="Source element ID (pointer)",
    help="The source element id (pointer; refer to RICO dataset).",
)
@click.option(
    "-t",
    "--target-screen",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    prompt="Path to source screen's view hierarchy",
    help="The path to the target screen's view hierarchy (refer to RICO dataset).",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(
        [
            "PageContainsLabel",
            "LargestTextElementsContainLabel",
            "LabelTextSimilarity",
            "TextSimilarity",
            "TextSimilarityNeighbors",
            "TextOnly",
            "LayoutOnly",
        ],
        case_sensitive=False,
    ),
    required=True,
    show_default=True,
    default="PageContainsLabel",
    prompt="Link prediction model",
    help="The link prediction model to use.",
)
def predict(source_screen, source_element_id, target_screen, model):
    """Outputs a label (i.e., 'LINK' or 'NON-LINK') and score (i.e., 'linkability score'
    ranging from 0 to 1) for a link between the SOURCE SCREEN and
    the TARGET SCREEN originating from the node with the SOURCE ELEMENT ID
    using the link prediction MODEL.
    """
    source_view_hierarchy = ViewHierarchy.from_file(source_screen)
    target_view_hierarchy = ViewHierarchy.from_file(target_screen)
    data_point = RicoDataPoint(
        source=RicoDataPoint.RawSourceData(
            id=source_view_hierarchy.request_id,
            element_id=source_element_id,
        ),
        target=RicoDataPoint.RicoScreen(id=target_view_hierarchy.request_id),
        application_name=None,
        trace_id=None,
        data_type=None,
        source_view_hierarchy=source_view_hierarchy,
        target_view_hierarchy=source_view_hierarchy,
    )
    clf = joblib.load(f"models/{model}Classifier.joblib")
    predicted_label = clf.predict([data_point])[0]
    predicted_score = clf.decision_function([data_point])[0]
    print(
        f"{'LINK' if predicted_label == 1 else 'NON-LINK'} (score={round(predicted_score, 3)})"
    )


if __name__ == "__main__":
    predict()
