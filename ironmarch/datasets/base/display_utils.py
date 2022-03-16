"""Diplay utilities for rootflow datasets

Attributes:
    TAB_SIZE: Length of the tab for dataset display functions
    TAB: A string of `TAB_SIZE` space characters.
"""

from typing import Any, List, Mapping, Sequence, Tuple, Union

import textwrap

TAB_SIZE = 4
TAB = "".join([" " for i in range(TAB_SIZE)])


def truncate_with_elipse(string: str, max_width: int) -> str:
    """Truncates a string, placing an elipse at the end if the string is truncated.

    Args:
        string (str): String to truncate.
        max_width (int): Width to truncate to.

    Returns:
        str: The truncated string.
    """
    if len(string) > max_width:
        string = string[: max_width - 3]
        string += "..."
    return string


def format_data_element(element: Any) -> str:
    """Formats a data element for console display"""
    if isinstance(element, str):
        return element.strip()
    elif isinstance(element, float):
        return f"{element:.3f}"
    elif isinstance(element, type):
        return element.__name__
    else:
        return str(element)


def format_docstring(docstring: str, display_width: int, indent: bool = False) -> str:
    """Formats a docstring

    Formats a docstring to conform to a specific display width, optionally indenting
    the entire string.

    Args:
        docstring (str): The docstring to format.
        display_width (int): Width to wrap the docstring.
        indent (:obj:`bool`, optional): Whether to indent the docstring.

    Returns:
        str: Formatted docstring.
    """
    if docstring is None:
        docstring = "(No Description)"
    if indent:
        display_width -= TAB_SIZE
    formatted_docstring = ""
    docstring_lines = [line.strip() for line in docstring.split("\n")]
    for line in docstring_lines:
        wrapped_lines = textwrap.wrap(line, width=display_width)
        for line in wrapped_lines:
            if indent:
                formatted_docstring += TAB
            formatted_docstring += line + "\n"
    return formatted_docstring[:-1]


def format_statistics(
    statistics: dict, display_width: int, indent: bool = False
) -> str:
    """Formats dataset statistics

    Formats a given dictionary to be printed in a pseudo-yaml manner. (Instead of
    always breaking, single values will stay on the same line). The function will
    do so recursively, through multiple layers of dictionaries and lists. Lines which
    are longer than the specified width will be wrapped. The values of the dictionary
    will be formatted using the `format_data_element` function, which will, for example
    limit the precision of floating point values, among other things.

    Args:
        statistics (dict): A dict which we would like to format.
        display_width (int): The width at which we would like to start wrapping.
        indent (bool): Whether to begin the formatting with an indent.

    Returns:
        str: The formatted statistics.
    """

    def _format_statistics(statistics: Union[dict, list], display_width: int):
        if isinstance(statistics, Sequence) and not isinstance(statistics, str):
            line_lists = [
                _format_statistics(item, display_width - TAB_SIZE)
                for item in statistics
            ]

            lines = []
            modified_tab = "-" + TAB[1:]
            for line_list in line_lists:
                first_line = modified_tab + line_list[0]
                lines += [
                    first_line,
                    *[TAB + line for line in line_list[1:]],
                ]
                lines += lines
            return lines
        elif isinstance(statistics, Mapping):
            lines = []
            for key, value in statistics.items():
                if isinstance(value, Mapping):
                    lines.append(f"{key}:")
                    nested_lines = _format_statistics(value, display_width - TAB_SIZE)
                    lines += [TAB + line for line in nested_lines]
                else:
                    if isinstance(value, Sequence) and not isinstance(value, str):
                        element_example = value[0]
                        if isinstance(element_example, Mapping):
                            lines.append(f"{key}:")
                            nested_lines = _format_statistics(
                                value, display_width - TAB_SIZE
                            )
                            lines += [TAB + line for line in nested_lines]
                        else:
                            value = [format_data_element(element) for element in value]
                            lines += textwrap.wrap(
                                f"{key}: {value}", width=display_width
                            )
                    else:
                        value = format_data_element(value)
                        lines += textwrap.wrap(f"{key}: {value}", width=display_width)
            return lines

    formatted_stats_string = ""
    if indent:
        lines = _format_statistics(statistics, display_width - TAB_SIZE)
    else:
        lines = _format_statistics(statistics, display_width)
    for line in lines:
        if indent:
            formatted_stats_string += TAB
        formatted_stats_string += line + "\n"
    return formatted_stats_string[:-1]


def flatten_example(example: dict) -> list:
    """Expands the data and target components of a dataset example.

    If the data or target fields of a given example are either a list or a dict, they
    will be expanded out into a flat list.
    """
    id, data, target = example["id"], example["data"], example["target"]
    flat_example = [id]

    if isinstance(data, Sequence) and not isinstance(data, str):
        flat_example += data
    elif isinstance(data, Mapping):
        flat_example += data.values()
    else:
        flat_example.append(data)

    if isinstance(target, Mapping):
        flat_example += target.values()
    else:
        flat_example.append(target)

    return flat_example


def get_flat_column_names(example: dict) -> List[str]:
    """Flattens the data and target fields, returning names for each.

    Returns a list of column names for the flattened version of the dataset example.
    If the data or target fields are a dictionary, the keys are returned. If the data
    or field is a list, the function will return feature_{i} for each expanded value.
    """
    _, data, target = example["id"], example["data"], example["target"]

    column_names = ["id"]
    if isinstance(data, Sequence) and not isinstance(data, str):
        column_names += [f"feature_{i}" for i in range(len(data))]
    elif isinstance(data, Mapping):
        column_names += data.keys()
    elif isinstance(data, str):
        column_names.append("text")
    else:
        column_names.append("data")

    if isinstance(target, Mapping):
        column_names += target.keys()
    else:
        column_names.append("target")

    return column_names


# TODO Does not handle the case where we have an extreme number of columns for the
# flattened examples. It may be necessary to implement some sort of column truncation.
# (i.e. After n data columns, an elipse, then after n target columns, another)

# TODO Currently does not support None targets. Should remove the target column in
# the case that it is an unlabeled dataset.
def format_examples_tabular(
    examples: List[dict], table_width: int, indent: bool = False
) -> str:
    """Formats examples into a tabular display string.

    Returns the given list of examples in a tabular format, dynamically resizing
    columns and truncating table values where necessary. The function will
    flatten the given examples if the `"data"` or `"target"`s are dicts or lists,
    prefering to increase the number of columns instead of printing lists within
    the table.

    Args:
        examples List[dict]: The data examples to format.
        table_width (int): Desired width of the table in number of characters.
        indent (:obj:`bool`, optional): Whether to indent the entire table.

    Returns:
        str: The formatted examples.
    """
    if indent:
        table_width = table_width - TAB_SIZE
    column_names = get_flat_column_names(examples[0])
    examples = [flatten_example(example) for example in examples]

    num_columns = len(column_names)
    column_seperator = " "
    column_width = (
        table_width - (len(column_seperator) * (num_columns))
    ) // num_columns

    formatted_examples_string = ""

    divider_string = ""
    if indent:
        formatted_examples_string += TAB
    for column_header in column_names:
        column_header = truncate_with_elipse(column_header, column_width)
        formatted_examples_string += (
            f"{column_header:<{column_width}}" + column_seperator
        )
        divider_string += "".join(["-" for i in range(column_width)]) + column_seperator
    formatted_examples_string += "\n"
    if indent:
        formatted_examples_string += TAB
    formatted_examples_string += divider_string + "\n"

    for example in examples:
        example_string = ""
        for example_element in example:
            column_formatted_element = truncate_with_elipse(
                format_data_element(example_element), column_width
            )
            example_string += (
                f"{column_formatted_element:<{column_width}}" + column_seperator
            )
        if indent:
            formatted_examples_string += TAB
        formatted_examples_string += example_string + "\n"

    return formatted_examples_string[:-1]
