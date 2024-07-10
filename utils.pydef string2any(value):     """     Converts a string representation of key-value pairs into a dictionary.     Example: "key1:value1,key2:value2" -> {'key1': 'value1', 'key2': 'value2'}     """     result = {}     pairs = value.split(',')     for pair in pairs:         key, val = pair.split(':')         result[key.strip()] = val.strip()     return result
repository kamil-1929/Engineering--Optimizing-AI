def string2any(value):
    """
    Converts a string representation of key-value pairs into a dictionary.
    Example: "key1:value1,key2:value2" -> {'key1': 'value1', 'key2': 'value2'}
    """
    result = {}
    pairs = value.split(',')
    for pair in pairs:
        key, val = pair.split(':')
        result[key.strip()] = val.strip()
    return result
