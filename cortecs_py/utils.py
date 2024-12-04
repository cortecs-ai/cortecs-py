def convert_model_name(name: str, to_hf_format: bool = True) -> str:
    """
    Converts a model name between `hf_name` and `model_id` formats.

    Args:
        name (str): The model name to convert.
        to_hf_format (bool): If True, converts `model_id` (with `--`) to `hf_name` (with `/`).
                             If False, converts `hf_name` (with `/`) to `model_id` (with `--`).

    Returns:
        str: The converted model name.
    """
    if to_hf_format:
        return name.replace("--", "/")
    else:
        return name.replace("/", "--")
