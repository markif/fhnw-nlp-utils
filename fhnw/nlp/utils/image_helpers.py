
def image_to_base64(image_path):
    """Base64 conversion of an image.

    Parameters
    ----------
    image_path : str
        The url/path of the image
        
    Returns
    -------
    string
        The base64 encoded image
    """
    import base64
    import urllib
    from urllib.parse import urlparse
    from mimetypes import guess_type
    
    mime_type, _ = guess_type(image_path)
    # Default to png
    if mime_type is None:
        mime_type = 'image/png'

    # check if local file
    url_parsed = urlparse(image_path)
    if url_parsed.scheme in ('file', ''):
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    else:
        image_file = urllib.request.urlopen(image_path)
        image_data = image_file.read()   
    
    base64_encoded_data = base64.b64encode(image_data).decode('utf-8')
    
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"
